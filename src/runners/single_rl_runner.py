import collections
import csv
import os
import random
import sys

import chex
from flax.core import freeze, unfreeze
from hydra.utils import to_absolute_path

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# =========================================================
import jax

print("--- JAX 环境诊断 ---")
try:
    print(f"JAX 默认后端: {jax.default_backend()}")
    print(f"JAX 可用设备: {jax.devices()}")
except Exception as e:
    print(f"无法获取 JAX 设备信息: {e}")
print("----------------------\n")
# =========================================================

import jax.numpy as jnp
import numpy as np
import distrax
from tqdm import tqdm
from flax.training import checkpoints
from datetime import datetime
import hydra
from omegaconf import DictConfig

# 你自己的模块
from src.envs.sat_env import SatEnv
from src.models.ac_gnn import ACGNN
from src.learners import single_rl_learner
from src.utils.data_parser import parse_cnf
from src.utils.metrics_tools import flatten_metrics, mean_std
import logging
logging.getLogger("absl").setLevel(logging.WARNING)
# 导入优化所需的额外模块
import flax.struct
from typing import NamedTuple, Any
from functools import partial

# ========================
# 超参：循环式训练 & 评估
# ========================
TRAIN_STEPS_PER_CYCLE = 3000   # 每个循环用于训练的数据步数（rollout steps）
NUM_ENVS = 4  # <-- 新增：并行环境的数量
EVAL_EPISODES_PER_CYCLE = 50  # 每个循环评估的episodes数量
# NUM_CYCLES = 500  # 循环次数
SAVE_BEST_EVERY_CYCLE = True  # 每个循环结束根据eval最优另存best.ckpt
LOG_FILE = "train_eval_log.txt"


def load_cnf_problems(cnf_data_dir: str):
    cnf_fnames = sorted([f for f in os.listdir(cnf_data_dir) if f.endswith('.cnf')])
    problems = []
    print(f"Found {len(cnf_fnames)} SAT instances.")
    for fname in tqdm(cnf_fnames, desc="LOADING CNF PROBLEMS"):
        cnf_path = os.path.join(cnf_data_dir, fname)
        num_vars, num_clauses, clauses = parse_cnf(cnf_path)
        problems.append({
            "num_vars": num_vars,
            "num_clauses": num_clauses,
            "clauses": clauses,
        })
    return problems


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # jax 本身用 PRNGKey，不是全局随机状态
    key = jax.random.PRNGKey(seed)
    return key


# ==============================================================================
#                      【改造後的核心並行函數】
# ==============================================================================
@flax.struct.dataclass
class VmappedRolloutCarryState:
    """用於在 lax.scan 中傳遞並行環境狀態的容器。"""
    train_state: flax.training.train_state.TrainState
    vmapped_env_state: Any
    vmapped_obs: Any
    key: jax.random.PRNGKey
    # 追踪每个并行环境的回合数
    ep_counts: chex.Array

@partial(jax.jit, static_argnames=("network", "vmapped_reset_fn", "vmapped_step_fn", "num_steps", "num_envs"))
def collect_rollouts(
        initial_carry_state: VmappedRolloutCarryState,
        problems,
        network: ACGNN,
        vmapped_reset_fn,
        vmapped_step_fn,
        num_steps: int,
        num_envs: int
):
    """
    【JAX 並行優化版】
    使用 jax.lax.scan 在時間維度上迭代，而在每個時間步內部，
    所有環境 (num_envs) 都被並行處理。
    """

    def _one_step_rollout(carry, _):
        """定義單步操作，但這次是針對一批 (vmapped) 環境"""
        ep_counts = carry.ep_counts  # <-- 解包新增的计数器
        train_state = carry.train_state
        vmapped_env_state = carry.vmapped_env_state
        vmapped_obs = carry.vmapped_obs
        key = carry.key

        # a. 動作採樣 (模型原生支持 batch 輸入)
        key, act_key = jax.random.split(key)
        logits, value = network.apply({'params': train_state.params}, vmapped_obs['agent_0'])
        pi = distrax.Categorical(logits=logits)
        actions = pi.sample(seed=act_key)
        log_probs = pi.log_prob(actions)
        # b. 環境並行步進
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        actions_dict = {"agent_0": actions}
        next_vmapped_obs, next_vmapped_env_state, rewards, dones, infos = vmapped_step_fn(
            step_keys, vmapped_env_state, actions_dict
        )

        # c. 處理完成的回合 (自動重置)
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        N = problems["num_vars"].shape[0]
        key, idx_key = jax.random.split(key)
        new_problem_indices = jax.random.randint(idx_key, (num_envs,), 0, N)
        new_problems = jax.tree_util.tree_map(lambda x: x[new_problem_indices], problems)
        obs_after_reset, state_after_reset = vmapped_reset_fn(reset_keys, new_problems)

        done_mask = dones["__all__"]
        new_ep_counts = ep_counts + done_mask.astype(jnp.int32)

        def _reset_if_done(old, new):
            """根據 done_mask 選擇性地替換 PyTree 的葉節點"""
            # old.ndim 是舊狀態陣列的維度數, e.g., A_pos 是 3
            # 我們需要為 done_mask 增加 (ndim - 1) 個新維度
            # 例如，對於 A_pos (ndim=3)，mask shape 變為 (16, 1, 1)
            mask_shape = done_mask.shape + (1,) * (old.ndim - 1)
            reshaped_mask = jnp.reshape(done_mask, mask_shape)
            return jnp.where(reshaped_mask, new, old)

        final_next_env_state = jax.tree_util.tree_map(
            _reset_if_done, next_vmapped_env_state, state_after_reset
        )
        final_next_obs = jax.tree_util.tree_map(
            _reset_if_done, next_vmapped_obs, obs_after_reset
        )

        # d. 存儲 Transition
        transition = single_rl_learner.Transition(
            done=dones["__all__"],
            action=actions,
            value=value,
            reward=rewards["agent_0"],
            log_prob=log_probs,
            obs=vmapped_obs['agent_0'],  # 存储的是步进前的 obs (s_t)
            info=infos
        )

        next_carry = VmappedRolloutCarryState(
            train_state=train_state,
            vmapped_env_state=final_next_env_state,  # 使用重置后的状态
            vmapped_obs=final_next_obs,  # 使用重置后的观测
            key=key,
            ep_counts=new_ep_counts
        )

        return next_carry, transition

    # 使用 jax.lax.scan 執行整個 rollout 循環
    final_carry, traj_batch = jax.lax.scan(
        _one_step_rollout, initial_carry_state, None, length=num_steps
    )

    # 計算最後一個狀態的價值
    _, last_val = network.apply({'params': final_carry.train_state.params}, final_carry.vmapped_obs['agent_0'])

    return final_carry, traj_batch, last_val


@partial(jax.jit, static_argnames=("network", "env", "max_steps"))
def evaluate_policy(
        key, env, network, params, problems,
        problem_indices,  #
        max_steps: int
):
    """
    【新版】评估策略：对指定的一批问题进行评估。
    """

    def _run_one_episode(key, problem_idx):
        """内部函数，运行单个回合，逻辑保持不变"""
        problem = jax.tree_util.tree_map(lambda x: x[problem_idx], problems)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, problem)

        def _one_step_eval(carry, _):
            obs, state, key = carry
            logits, _ = network.apply({'params': params}, obs['agent_0'])
            action = jnp.argmax(logits, axis=-1)
            key, step_key = jax.random.split(key)
            next_obs, next_state, reward, done, info = env.step_env(
                step_key, state, {"agent_0": action}
            )
            return (next_obs, next_state, key), (reward, done)

        _, (rewards, dones) = jax.lax.scan(
            _one_step_eval, (obs, state, key), None, length=max_steps
        )

        first_done_idx = jnp.argmax(dones["__all__"])
        ep_len = jnp.where(jnp.any(dones["__all__"]), first_done_idx + 1, max_steps)
        step_indices = jnp.arange(max_steps)
        mask = step_indices < ep_len
        ep_return = jnp.sum(rewards["agent_0"] * mask)
        terminal_reward = jnp.sum(rewards["agent_0"] * dones["__all__"])
        is_solved = terminal_reward > 0
        return is_solved, ep_return, ep_len

    # --- 核心修改 ---
    # 使用传入的 problem_indices，而不是随机生成
    keys = jax.random.split(key, problem_indices.shape[0])

    # vmap 现在会遍历我们提供的所有问题索引
    solves, ep_returns, ep_lens = jax.vmap(_run_one_episode)(keys, problem_indices)

    eval_stats = {
        "eval_solve_rate": jnp.mean(solves),
        "eval_avg_len": jnp.mean(ep_lens),
        "eval_avg_return": jnp.mean(ep_returns),
        "eval_episodes": len(solves),
    }
    # 返回更新后的 key 和统计数据
    return jax.random.split(key)[0], eval_stats


def reset_heads_to_init(train_state, network, env, problems_raw, seed=42, reset_actor=True, reset_critic=True):
    """
    (此函數保持不變)
    """
    init_key = jax.random.PRNGKey(seed)
    dummy_obs, _ = env.reset(init_key, problems_raw[0])
    blank_params = network.init(init_key, dummy_obs['agent_0'])['params']
    p = unfreeze(train_state.params)
    np_ = unfreeze(blank_params)
    if reset_actor:
        if 'actor_dense_1' in p:
            p['actor_dense_1'] = np_['actor_dense_1']
            if 'actor_dense_2' in p:
                p['actor_dense_2'] = np_['actor_dense_2']
        if 'actor_output' in p:
            p['actor_output'] = np_['actor_output']
    if reset_critic:
        if 'critic_dense_1' in p:
            p['critic_dense_1'] = np_['critic_dense_1']
        if 'critic_dense_2' in p:
            p['critic_dense_2'] = np_['critic_dense_2']
        if 'critic_output' in p:
            p['critic_output'] = np_['critic_output']
    new_params = freeze(p)
    new_opt_state = train_state.tx.init(new_params)
    train_state = train_state.replace(params=new_params, opt_state=new_opt_state, step=0)
    print("✅ Reinitialized actor/critic heads successfully.")
    return train_state


@hydra.main(version_base=None, config_path="../../configs", config_name="single_rl_ppo_config.yaml")
def train(hydra_config: DictConfig):
    global config
    config = hydra_config

    # 0) RNG & 数据
    key = set_global_seeds(config.SEED)
    problems_raw = load_cnf_problems(config.ENV_PARAMS.CNF_DATA_DIR) # shape( num_ problems_raw )
    problems = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *problems_raw)
    # 1) 环境/模型
    num_vars = problems_raw[0]["num_vars"]
    num_clauses = problems_raw[0]["num_clauses"]
    env = SatEnv(num_vars, num_clauses, max_clause_len=config.ENV_PARAMS.WRAPPER_PARAMS.max_clause_len,
                 c_bonus=config.ENV_PARAMS.WRAPPER_PARAMS.c_bonus,
                 alpha=config.ENV_PARAMS.WRAPPER_PARAMS.alpha
                 ,max_steps=config.ENV_PARAMS.WRAPPER_PARAMS.max_steps)

    # --- 改造點 1：創建向量化的環境函數 ---
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, 0))
    vmapped_step = jax.vmap(env.step_env, in_axes=(0, 0, 0))

    network = ACGNN(
        hidden_dim=config.MODEL_PARAMS.HIDDEN_DIM,
        num_message_passing_step=config.MODEL_PARAMS.NUM_MESSAGE_PASSING_STEP
    )

    # 2) 初始化 TrainState
    key, net_key = jax.random.split(key)
    dummy_obs, _ = env.reset(jax.random.PRNGKey(0), problems_raw[0])
    train_state = single_rl_learner.create_train_state(
        model=network,
        key=net_key,
        config=config,
        dummy_input=dummy_obs['agent_0']
    )

    # 加载model（只恢复 params）
    if config.TRAIN_PARAMS.RESUME_CKPT_PATH is not None:
        resume_dir_abs = to_absolute_path(config.TRAIN_PARAMS.RESUME_CKPT_PATH)

        try:
            print(f"🔄 Attempting to resume from checkpoint: {resume_dir_abs}")

            # (1) 恢復完整的 TrainState (包含 params, opt_state, step)
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=resume_dir_abs,
                target=train_state,  # 這裡 target 只是為了結構匹配
                prefix="cycle_3000"  # 或者 "cycle_"
            )

            # (2) 只保留恢復的 params，拋棄舊的 opt_state 和 step
            #    這就隱式地重置了優化器，因為我們將使用下面新初始化的 opt_state
            train_state = train_state.replace(params=restored_state.params)
            print("✅ Successfully restored model parameters (GNN body).")

            # (3) 重置 Actor/Critic 的頭部，並重新初始化優化器狀態
            train_state = reset_heads_to_init(
                train_state=train_state,
                network=network,
                env=env,
                problems_raw=problems_raw,
                seed=config.SEED,  # 使用固定的種子保證重置的一致性
                reset_actor=True,
                reset_critic=True
            )
            print("🔧 Reinitialized Actor/Critic heads and Optimizer state for the new curriculum stage.")

        except Exception as e:
            print(f"⚠️ Resume failed ({e}), starting training from scratch.")

    # 3) 日志 & Checkpoint (保持不變)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = os.path.abspath(f"experiments/single_rl/{time_str}/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {ckpt_dir}")

    log_path = os.path.join(os.path.dirname(ckpt_dir), LOG_FILE)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Train/Eval Log | start_time={time_str} | seed={config.SEED}\n")
        f.write("# Fields: cycle | "
                "train_loss_total train_loss_value train_loss_policy train_entropy | "
                "train_value_mean train_value_std train_reward_mean train_reward_std | "
                "train_ep_num train_solve_rate train_avg_len train_avg_return | "
                "eval_solve_rate eval_avg_len eval_avg_return eval_episodes\n")

    ppo_update = single_rl_learner.make_update(config.PPO_PARAMS)
    best_solve = -1.0
    best_len = float('inf')

    # 4) 循環式訓練-评估

    pbar = tqdm(range(1, config.TRAIN_PARAMS.NUM_CYCLES + 1), desc="Cycles")
    steps_per_env = TRAIN_STEPS_PER_CYCLE // NUM_ENVS

    for cycle_idx in pbar:
        key, p_idx_key, reset_key, rollout_key, update_key = jax.random.split(key, 5)

        # 1. 为本次 rollout 随机选择一批初始问题
        initial_problem_indices = jax.random.randint(p_idx_key, (NUM_ENVS,), 0, len(problems_raw))
        initial_problems = jax.tree_util.tree_map(
            lambda *x: jnp.stack(x), *[problems_raw[i] for i in initial_problem_indices]
        )



        # 2. 用新的 key 重置所有环境，确保初始状态的随机性
        reset_keys = jax.random.split(reset_key, NUM_ENVS)
        vmapped_obs, vmapped_state = vmapped_reset(reset_keys, initial_problems)

        # 3. 为本次 rollout 构建一个全新的、干净的初始 carry 容器
        #    注意：train_state 是从上一个循环继承的，因为它包含了需要持续学习的模型权重
        carry = VmappedRolloutCarryState(
            train_state=train_state,
            vmapped_env_state=vmapped_state,
            vmapped_obs=vmapped_obs,
            key=rollout_key,
            ep_counts=jnp.ones(NUM_ENVS, dtype=jnp.int32)
        )

        # ===== 1. 并行收集数据 (Collect Rollouts) =====
        # 现在 collect_rollouts 函数接收的是一个干净的初始状态
        ep_counts_before = jnp.sum(carry.ep_counts)
        carry, traj_batch, last_val = collect_rollouts(
            carry, initial_problems, network, vmapped_reset, vmapped_step,
            num_steps=steps_per_env, num_envs=NUM_ENVS,
        )
        ep_counts_after = jnp.sum(carry.ep_counts)
        # 本次 rollout 中总共开始过的回合数 = 结束时的总数 - 开始时的总数 + 初始环境数
        # (因为初始的 NUM_ENVS 个回合也算在内)
        total_episodes_started = (ep_counts_after - ep_counts_before) + NUM_ENVS

        # 这里的3只是判断 是否正的完成了
        solved_in_train = jnp.sum((traj_batch.reward > 3) & traj_batch.done)
        train_solve_rate = jax.lax.cond(
            total_episodes_started > 0,
            lambda: solved_in_train / total_episodes_started,
            lambda: 0.0
        )
        # 从返回的 carry 中获取更新后的 key，以便下一次循环使用
        key = carry.key

        # ===== 2. 更新模型 (PPO Update) =====
        key, update_key = jax.random.split(key)
        # 这里的 carry.train_state 是 rollout 结束时的 train_state，但实际上它在 rollout 期间没有改变
        # 我们使用这个 train_state 来进行更新
        updated_train_state, metrics = ppo_update(
            network, carry.train_state, traj_batch, last_val, update_key
        )

        # 将更新后的 train_state 保存下来，用于下一个循环
        train_state = updated_train_state

        # ===== 3. 计算并记录训练统计数据 (Logging) =====
        metrics_py = jax.device_get(metrics)
        total_loss = np.mean(metrics_py[0])
        vloss = np.mean(metrics_py[1])  # <-- 修正：直接取 metrics_py[1]
        aloss = np.mean(metrics_py[2])  # <-- 修正：直接取 metrics_py[2]
        ent = np.mean(metrics_py[3])

        val_mean, val_std = mean_std(traj_batch.value)
        rew_mean, rew_std = mean_std(traj_batch.reward)

        # final_env_state = carry.vmapped_env_state
        # num_solved_in_batch = jnp.sum(final_env_state.prev_unsat_ratio == 0.0)
        # train_solve_rate = num_solved_in_batch / NUM_ENVS
        tr_stats_py = jax.device_get({
            "loss_total": total_loss, "loss_value": vloss, "loss_policy": aloss, "entropy": ent,
            "value_mean": val_mean, "value_std": val_std, "reward_mean": rew_mean, "reward_std": rew_std,
            "train_ep_num": total_episodes_started,  # 仍然可以记录完成的回合数
            "train_solve_rate": train_solve_rate,
        })

        # 训练状态打印
        train_line = [
            f"[Train] loss {tr_stats_py['loss_total']:.3f}",
            f"V {tr_stats_py['loss_value']:.3f}", f"π {tr_stats_py['loss_policy']:.3f}",
            f"H {tr_stats_py['entropy']:.3f}",
            f"V̂ {tr_stats_py['value_mean']:.3f}±{tr_stats_py['value_std']:.3f}",
            f"R {tr_stats_py['reward_mean']:.3f}±{tr_stats_py['reward_std']:.3f}",
            f"Ep {tr_stats_py['train_ep_num']}", f"Solve {tr_stats_py['train_solve_rate'] * 100:.1f}%",
        ]
        print("\n" + " | ".join(train_line))

        ev_stats_py = {"eval_solve_rate": 0.0, "eval_avg_len": float('inf'),
                       "eval_avg_return": 0.0, "eval_episodes": 0}

        # ===== 4. 评估策略 (Evaluation) =====
        if tr_stats_py['train_solve_rate'] >= 0.70:  #
            key, sub_key = jax.random.split(key)
            problem_indices = jax.random.randint(sub_key, (EVAL_EPISODES_PER_CYCLE,), 0, len(problems_raw))
            key, ev_stats = evaluate_policy(
                key, env, network, train_state.params, problems,
                problem_indices=problem_indices,
                # num_episodes=EVAL_EPISODES_PER_CYCLE,
                max_steps=config.ENV_PARAMS.WRAPPER_PARAMS.max_steps
            )
            ev_stats_py = jax.device_get(ev_stats)

            eval_line = [f"[Eval ] Solve {ev_stats_py['eval_solve_rate'] * 100:.1f}%",
                         f"Len {ev_stats_py['eval_avg_len']:.2f}",
                         f"Return {ev_stats_py['eval_avg_return']:+.4f}",
                         f"Eps {ev_stats_py['eval_episodes']}"]
            print(" | ".join(eval_line))

            ev_solve, ev_len = ev_stats_py["eval_solve_rate"], ev_stats_py["eval_avg_len"]
            should_save = ev_solve > best_solve or (abs(ev_solve - best_solve) < 1e-9 and ev_len < best_len)

            if SAVE_BEST_EVERY_CYCLE and should_save:
                best_solve, best_len = ev_solve, ev_len
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir, target=train_state, step=cycle_idx,
                    prefix="best_eval_", overwrite=True
                )
                print(f"✅ New best (solve={best_solve * 100:.2f}%, len={best_len:.2f}). Saved.")
        else:
            print(f"| [Eval ] Skipped. Train solve rate {tr_stats_py['train_solve_rate'] * 100:.1f}% <= 80%")

        # 保存周期性 checkpoint
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir, target=train_state, step=cycle_idx, prefix="cycle_", keep=3
        )

        # 写入日志文件
        with open(log_path, "a", encoding="utf-8") as f:
            log_data = {**tr_stats_py, **ev_stats_py, "train_avg_len": 0.0, "train_avg_return": 0.0}  # 补全缺失的训练指标
            f.write(
                f"{cycle_idx} | "
                f"{log_data['loss_total']:.6f} {log_data['loss_value']:.6f} {log_data['loss_policy']:.6f} {log_data['entropy']:.6f} | "
                f"{log_data['value_mean']:.6f} {log_data['value_std']:.6f} {log_data['reward_mean']:.6f} {log_data['reward_std']:.6f} | "
                f"{log_data['train_ep_num']} {log_data['train_solve_rate']:.6f} {log_data['train_avg_len']:.6f} {log_data['train_avg_return']:.6f} | "
                f"{log_data['eval_solve_rate']:.6f} {log_data['eval_avg_len']:.6f} {log_data['eval_avg_return']:.6f} {log_data['eval_episodes']}\n"
            )

        # 更新进度条
        pbar.set_postfix({
            "train_solve%": f"{tr_stats_py['train_solve_rate'] * 100:.1f}",
            "eval_solve%": f"{ev_stats_py['eval_solve_rate'] * 100:.1f}",
            "best_solve%": f"{best_solve * 100:.1f}"
        })

    # 5) 保存最终权重
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir, target=carry.train_state, step=config.TRAIN_PARAMS.NUM_CYCLES,
        prefix="final_model_", overwrite=True
    )
    print(f"✅ Final model saved to {ckpt_dir}")
    print(f"📄 Full TXT log saved to {log_path}")


if __name__ == "__main__":
    train()