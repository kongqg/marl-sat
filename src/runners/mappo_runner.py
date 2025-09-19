# 文件: mappo_runner.py
import flax
import jax
import jax.numpy as jnp
import numpy as np
import os
import random
from tqdm import tqdm
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from flax.training import checkpoints
from flax.training.train_state import TrainState
from functools import partial
import time
import optax

from src.envs.multi_agent_sat_env import SATEnv
from src.learners.mappo_gnn_sat_learner import SATDataWrapper, GNN_ActorCritic, make_train_cycle, RunnerState
from src.utils.data_parser import parse_cnf, load_cnf_problems
from src.utils.model_init import load_bc_and_inject


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


@partial(jax.jit, static_argnames=("env", "network", "max_steps"))
def evaluate_policy(key, env, network, params, problem_clauses, max_steps):
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(problem_clauses, reset_key)

    def _one_step_eval(carry, _):
        env_state, obs, key = carry
        global_state = obs[1]  # obs is a tuple (local_obs, global_state)
        pi, _ = network.apply({'params': params}, global_state, env.agent_vars, env.action_mask)
        if env.action_mode == 0:  # single_flip
            # pi.logits 的 shape 是 (num_agents, max_vars_per_agent + 1)
            # argmax 直接得到每个 agent 的最佳动作索引 (一个整数)
            actions_array = jnp.argmax(pi.logits, axis=-1)  # shape: (num_agents,)
            actions_dict = {agent: actions_array[i] for i, agent in enumerate(env.agents)}
        else:  # multi_flip
            # pi.logits 的 shape 是 (num_agents, max_vars_per_agent, 2)
            # 对每个变量的 "不翻转 vs 翻转" logit 取 argmax
            actions_array = jnp.argmax(pi.logits, axis=-1)  # shape: (num_agents, max_vars_per_agent)
            actions_dict = {agent: actions_array[i, :] for i, agent in enumerate(env.agents)}
        key, step_key = jax.random.split(key)
        next_obs, next_state, reward, done, info = env.step(key=step_key, state=env_state, actions=actions_dict)
        return (next_state, next_obs, key), (done, info, next_state.env_state.variable_assignments)

    _, (dones, infos, all_assignments) = jax.lax.scan(_one_step_eval, (env_state, obs, key), None, length=max_steps)

    # --- 以下是修正后的正确逻辑 ---
    # 1. 真实地检查问题是否在任何一步被解决过
    was_ever_solved = jnp.any(infos['solved'])

    # 2. 找到第一个解决的步骤索引，这个索引只有在 was_ever_solved 为 True 时才有意义
    first_solve_step_index = jnp.argmax(infos['solved'])

    # 3. 根据该索引获取解
    solution = all_assignments[first_solve_step_index]

    # 4. 只有在真正解决时才使用 solution，否则使用占位符
    no_solution_placeholder = jnp.zeros_like(solution)
    solution_assignments = jnp.where(was_ever_solved, solution, no_solution_placeholder)

    # 5. 根据 was_ever_solved 来计算真实的解决步数
    steps_to_solve = jnp.where(was_ever_solved, first_solve_step_index + 1, max_steps)

    # 6. 返回真实的解决状态、步数和解
    return was_ever_solved, steps_to_solve, solution_assignments


@hydra.main(version_base=None, config_path="../../configs", config_name="MAPPO_CONFIG.yaml")
def main(hydra_config: DictConfig):
    config = OmegaConf.to_container(hydra_config, resolve=True)

    # --- 1. 初始化和數據加載 ---
    seed = config.get('SEED', 42)
    key = set_global_seeds(seed)
    problems_raw = load_cnf_problems(config.get('CNF_DATA_DIR', 'data'))
    if not problems_raw:
        return

    # 【新】将问题列表转换为 JAX PyTree，方便传入 JIT 函数
    # problems现在是一个字典，每个键对应一个包含所有问题数据的数组
    """
        # problems 的样子:
        {
          'num_vars':    jnp.array([50, 50, 50, ...]),          # shape: (问题总数,)
          'num_clauses': jnp.array([212, 212, 212, ...]),       # shape: (问题总数,)
          'clauses':     jnp.array([<array_1>, <array_2>, ...]) # shape: (问题总数, 每个问题的子句数, 文字数)
        }
    """

    print(f"[*] 原始問題總數: {len(problems_raw)}")
    # 設置隨機數種子以保證劃分的可複現性
    np_rng = np.random.RandomState(seed)
    indices = np.arange(len(problems_raw))
    np_rng.shuffle(indices)

    split_idx = int(len(problems_raw) * 0.8)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_problems_raw = [problems_raw[i] for i in train_indices]
    eval_problems_raw = [problems_raw[i] for i in eval_indices]

    print(f"[*] 數據集劃分完畢: {len(train_problems_raw)} 個用於訓練 (80%), {len(eval_problems_raw)} 個用於評估 (20%)")

    # 將訓練集轉換為 JAX PyTree，方便傳入 JIT 函數
    numeric_train_problems_raw = [
        {"num_vars": p["num_vars"], "num_clauses": p["num_clauses"], "clauses": p["clauses"]}
        for p in train_problems_raw
    ]
    train_problems_tree = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *numeric_train_problems_raw)
    # <--- 核心修改：數據集劃分 END --->
    # --- 2. 准备环境、模型和训练状态 (只执行一次) ---
    flat_config = {**config.get('environment', {}), **config.get('network', {}), **config.get('training', {})}

    # 创建基础环境和包装器
    env = SATEnv(
        num_vars=flat_config["NUM_VARS"],
        num_clauses=flat_config["NUM_CLAUSES"],
        max_steps=flat_config["MAX_STEPS"],
        action_mode=flat_config["action_mode"],
        r_clause=flat_config["rewards"]["R_CLAUSE"],
        r_sat=flat_config["rewards"]["R_SAT"],
        gamma=flat_config["GAMMA"],  # 若 SATEnv 构造函数支持
        vars_per_agent=flat_config["VARS_PER_AGENT"],
    )
    env = SATDataWrapper(env)

    # 【新】创建向量化的环境函数
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, 0))  # (problem_clauses, key)

    # 创建网络
    network = GNN_ActorCritic(
        gnn_hidden_dim=flat_config['GNN_HIDDEN_DIM'],
        gnn_num_message_passing_steps=flat_config['GNN_NUM_MESSAGE_PASSING_STEPS'],
        num_agents=env.num_agents,
        max_vars_per_agent=env.max_vars_per_agent,
        action_mode=flat_config["action_mode"]
    )

    # 初始化网络参数和 TrainState
    key, _rng_init, _rng_reset = jax.random.split(key, 3)
    # 使用第一个问题作为模板来获取 dummy_input 的形状
    (dummy_local_obs, dummy_global_state), dummy_wrapper_state = env.reset(
        problems_raw[0]['clauses'], _rng_reset
    )
    dummy_clause_masks = dummy_wrapper_state.env_state.agent_clause_masks
    dummy_neighbor_masks = dummy_wrapper_state.env_state.agent_neighbor_masks
    network_params = network.init(
        _rng_init,
        dummy_global_state,  # <--- 之前是 obs["gnn_input"]
        env.agent_vars,
        env.action_mask,
    )

    print("🔍 類型檢查:")
    print(f"  - type(network_params): {type(network_params)}")
    if 'params' in network_params:
        print(f"  - type(network_params['params']): {type(network_params['params'])}")


    params_for_state = flax.core.freeze(network_params['params'])

    def create_lr_schedule(config):
        num_updates = config.get("NUM_UPDATES", 1)  # 獲取總更新次數

        # 檢查是否啟用退火
        if config.get("ANNEAL_LR", False):
            print("📈 啟用學習率線性退火...")
            lr_start = config.get("LEARNING_RATE", 3e-4) * config.get("LR_START_FACTOR", 1.0)
            lr_end = config.get("LR_END_FLOOR", 1e-5)

            # 使用 optax 的 linear_schedule 創建退火函數
            # 它會在 num_updates 步內從 lr_start 線性衰減到 lr_end
            lr_schedule = optax.linear_schedule(
                init_value=lr_start,
                end_value=lr_end,
                transition_steps=num_updates
            )
            print(f"   - 初始學習率: {lr_start:.6f}")
            print(f"   - 最終學習率: {lr_end:.6f}")
            print(f"   - 衰減步數: {num_updates}")
            return lr_schedule
        else:
            print("📉 未啟用學習率退火，使用固定學習率。")
            return config.get("LEARNING_RATE", 3e-4)

    # 根據配置創建學習率或學習率調度器
    learning_rate_or_schedule = create_lr_schedule(flat_config)

    tx = optax.adam(learning_rate_or_schedule)
    # train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    train_state = TrainState.create(apply_fn=network.apply, params=params_for_state, tx=tx)

    train_state_for_run = train_state  # 默认使用新初始化的状态

    continue_path = config.get('loading', {}).get('continue_rl_run_path')
    inject_path = config.get('loading', {}).get('inject_bc_model_path')

    if continue_path:
        # 场景一：恢复之前的 RL 训练
        print(f"🔄 正在尝试从 RL 检查点恢复训练: '{continue_path}'")
        ckpt_dir_to_load = os.path.join(continue_path, "checkpoints")
        try:
            loaded_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir_to_load, target=train_state,
                                                          prefix="latest_model_", step=0)
            if loaded_state:
                reset_optimizer = config.get('loading', {}).get('RESET_OPTIMIZER', False)
                if reset_optimizer:
                    train_state_for_run = train_state.replace(params=loaded_state.params)
                    print("✅ RL 检查点加载成功！模型参数已恢复，优化器已重置。")
                else:
                    train_state_for_run = loaded_state
                    print("✅ RL 检查点加载成功！模型参数和优化器状态均已恢复。")
            else:
                print(f"⚠️ 警告: 从 '{ckpt_dir_to_load}' 加载 RL 检查点失败。将从头开始训练。")
        except Exception as e:
            print(f"⚠️ 警告: 加载 RL 检查点时发生错误: {e}。将从头开始训练。")

    elif inject_path:
        # 场景二：从 BC 预训练模型注入权重
        print(f"🔄 正在尝试从 BC 预训练模型注入权重: '{inject_path}'")
        try:
            bc_checkpoint = checkpoints.restore_checkpoint(ckpt_dir=inject_path, target={'params': train_state.params},
                                                           prefix="bc_model_")
            if bc_checkpoint:
                bc_params = bc_checkpoint['params']
                new_params_mutable = train_state.params.unfreeze()

                # 注入 GNN Encoder 参数
                new_params_mutable['encoder'] = bc_params['encoder']
                print("  ✅ GNN Encoder 参数已注入。")

                # 注入 Actor 相关参数
                actor_param_keys = [k for k in bc_params.keys() if 'actor' in k or 'agent_id_embedding' in k]
                for param_key in actor_param_keys:  # <--- 已修正
                    new_params_mutable[param_key] = bc_params[param_key]
                print(f"  ✅ Actor 相关参数 ({len(actor_param_keys)} 个) 已注入。")

                print("  ℹ️ Critic 参数已保持随机初始化。")
                final_params = flax.core.freeze(new_params_mutable)

                # 使用注入的参数和全新的优化器状态创建 TrainState
                train_state_for_run = train_state.replace(params=final_params)
                print("🛠️  优化器状态已自动重置。")
                print("🎉 BC 预训练模型迁移成功！")
            else:
                print(f"⚠️ 警告: 从 '{inject_path}' 加载 BC 检查点失败。将从头开始训练。")
        except Exception as e:
            print(f"⚠️ 警告: 加载 BC 检查点时发生错误: {e}。将从头开始训练。")
    else:
        # 场景三：从头开始训练
        print("ℹ️ 未配置任何加载路径，将从头开始训练。")



    # --- 3. 获取 JIT 编译的单轮训练函数 ---
    print("🛠️  正在 JIT 編譯單輪訓練函數...")
    start_time = time.time()
    # 调用新的 make_train_cycle 函数
    train_cycle_fn = make_train_cycle(flat_config, env, network, vmapped_reset)
    print(f"✅ JIT 編譯完成！耗時: {time.time() - start_time:.2f} 秒")

    # --- 4. 日誌和 Checkpoint 設置 ---
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.get('SAVE_DIR', 'experiments'), time_str)
    run_dir = os.path.abspath(run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"💾 实验结果將保存在: {run_dir}")

    log_file_path = os.path.join(run_dir, "training_metrics.txt")

    # --- 5. 执行训练循环 ---
    print(f"\n🎉 === 开始训练，总共 {flat_config['NUM_UPDATES']} 轮更新 ===")
    start_time = time.time()




    # 准备初始的 runner_state
    key, _rng = jax.random.split(key)
    # 随机选择一批初始问题来开始
    initial_indices = jax.random.randint(_rng, (flat_config["NUM_ENVS"],), 0, len(train_problems_raw))
    initial_problems = jax.tree_util.tree_map(lambda x: x[initial_indices], train_problems_tree)

    reset_keys = jax.random.split(_rng, flat_config["NUM_ENVS"])
    (initial_local_obs, initial_global_state), initial_env_state = vmapped_reset(initial_problems['clauses'], reset_keys)

    runner_state = RunnerState(
        train_state=train_state_for_run,
        env_state=initial_env_state,
        last_local_obs=initial_local_obs,
        last_global_state=initial_global_state,
        rng=key
    )
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        # 寫入日誌文件的標頭 (現在不需要 epoch 這一列了)
        header = (
            "update,"
            "mean_return,solve_rate,avg_unsat_clauses,avg_solve_steps,explained_variance,"
            "value_loss,actor_loss,policy_entropy\n"
        )
        log_file.write(header)

        pbar = tqdm(range(flat_config['NUM_UPDATES']), desc="训练进度")
        for update_idx in pbar:
            # 執行一輪"数据收集 + 模型更新"
            runner_state, metrics = train_cycle_fn(runner_state, train_problems_tree, jnp.array(update_idx))
            metrics_cpu = jax.device_get(metrics)


            # 1. 準備指標，直接取最後一個 epoch 的損失值
            final_epoch_metrics = {
                "update": update_idx + 1,
                "return": metrics_cpu['mean_episodic_return'],
                "solve_rate%": metrics_cpu['solve_rate'] * 100,
                "unsat_C": metrics_cpu['avg_unsatisfied_clauses'],
                "solve_steps": metrics_cpu['avg_steps_to_solve'],
                "expl_var": metrics_cpu['explained_variance'],
                # 直接索引 [-1] 來獲取數組的最後一個元素，並取平均值
                "v_loss": metrics_cpu['epoch_value_losses'][-1].mean(),
                "a_loss": metrics_cpu['epoch_actor_losses'][-1].mean(),
                "entropy": metrics_cpu['epoch_entropies'][-1].mean(),
                "ent_coef": metrics_cpu['current_ent_coef'],
            }

            # 2. 更新進度條顯示
            pbar_display = {
                "update": final_epoch_metrics['update'],
                "return": f"{final_epoch_metrics['return']:.1f}",
                "solve%": f"{final_epoch_metrics['solve_rate%']:.1f}",
                "unsat_C": f"{final_epoch_metrics['unsat_C']:.1f}",
                "v_loss": f"{final_epoch_metrics['v_loss']:.3f}",
                "entropy": f"{final_epoch_metrics['entropy']:.2f}"
            }
            pbar.set_postfix(pbar_display)

            # 3. 寫入日誌文件
            log_line = (
                f"{final_epoch_metrics['update']},"
                f"{final_epoch_metrics['return']:.4f},{final_epoch_metrics['solve_rate%'] / 100:.4f},"
                f"{final_epoch_metrics['unsat_C']:.4f},{final_epoch_metrics['solve_steps']:.4f},"
                f"{final_epoch_metrics['expl_var']:.4f},"
                f"{final_epoch_metrics['v_loss']:.4f},{final_epoch_metrics['a_loss']:.4f},"
                f"{final_epoch_metrics['entropy']:.4f}\n"
            )
            log_file.write(log_line)

            if (update_idx + 1) % config.get('evaluation', {}).get('EVAL_INTERVAL', 10) == 0:
                pbar.write("-" * 50)
                pbar.write(f"🔬 在線評估 (使用驗證集) [Update #{update_idx + 1}]")

                # 從驗證集中隨機採樣一小批問題
                eval_batch_size = config.get('evaluation', {}).get('EVAL_BATCH_SIZE', 32)
                if len(eval_problems_raw) < eval_batch_size:
                    eval_batch = eval_problems_raw
                else:
                    eval_batch = random.sample(eval_problems_raw, k=eval_batch_size)

                eval_solved_count = 0
                temp_final_state = runner_state.train_state
                eval_keys_batch = jax.random.split(runner_state.rng, len(eval_batch))

                for j, problem in enumerate(eval_batch):
                    solved, _, _ = evaluate_policy(
                        eval_keys_batch[j], env, network, temp_final_state.params,
                        problem['clauses'], flat_config['MAX_STEPS']
                    )
                    if solved:
                        eval_solved_count += 1

                true_solve_rate = eval_solved_count / len(eval_batch)
                pbar.write(f"✅ 驗證集求解率 ({len(eval_batch)}個樣本): {true_solve_rate:.2%}")
                pbar.write("-" * 50)

                pbar_display["eval_solve%"] = f"{true_solve_rate:.1%}"
                pbar.set_postfix(pbar_display)

            try:
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    target=runner_state.train_state,
                    step=0,
                    prefix="latest_model_",
                    overwrite=True
                )
            except PermissionError as e:
                pbar.write(f"⚠️  警告: 保存檢查點失敗，權限被拒絕。將繼續訓練...")
                pbar.write(f"   錯誤詳情: {e}")


    end_time = time.time()
    print(f"🏁 === 训练结束！总耗时: {end_time - start_time:.2f} 秒 ===")

    # --- 6. 保存最终模型 ---
    print(f"💾 最終模型已保存至: {ckpt_dir}")

    # --- 7. 训练后评估 ---
    eval_model_template = TrainState.create(apply_fn=network.apply, params=params_for_state, tx=tx)
    final_train_state = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=eval_model_template,
        step=0,
        prefix="latest_model_"
    )

    print(f"✅ 已加载最新的模型")
    print("\n🔬 === 开始测试整个数据集 ===")

    # --- 7a. 设置评估和解决方案日志记录 ---
    solutions_file_path = os.path.join(run_dir, "test_solutions.txt")
    print(f"📄 将解决方案记录到: {solutions_file_path}")

    solved_count = 0
    total_steps_to_solve = 0
    # 使用包含名称的原始问题列表
    eval_problems = eval_problems_raw
    key, eval_key_base = jax.random.split(runner_state.rng)
    eval_keys = jax.random.split(eval_key_base, len(eval_problems))

    # --- 7b. 运行评估并将日志记录到文件 ---
    with open(solutions_file_path, 'w', encoding='utf-8') as solutions_file:
        pbar_eval = tqdm(enumerate(eval_problems), total=len(eval_problems), desc="評估進度")

        for i, problem in pbar_eval:
            # 获取对应的 JAX 随机密钥
            eval_key = eval_keys[i]

            solved, steps_to_solve, solution = evaluate_policy(
                eval_key, env, network, final_train_state.params,
                problem['clauses'], flat_config['MAX_STEPS']
            )

            if solved:
                solved_count += 1
                total_steps_to_solve += steps_to_solve

                # 将布尔解转换为 0/1 字符串
                solution_np = np.array(solution)
                solution_str = "".join(solution_np.astype(np.int32).astype(str))

                # 使用 pbar_eval.write 打印到控制台，以免扰乱进度条
                pbar_eval.write(f"✅ 問題 {problem['name']} 已解決！步數: {steps_to_solve}")

                # 将结果写入解决方案文件
                solution_line = f"Problem: {problem['name']}, Solved: True, Steps: {steps_to_solve}, Solution: {solution_str}\n"
                solutions_file.write(solution_line)
            else:
                # 将失败结果写入解决方案文件
                solution_line = f"Problem: {problem['name']}, Solved: False\n"
                solutions_file.write(solution_line)

            # 更新进度条的后缀显示
            pbar_eval.set_postfix(
                {"solve_rate": f"{solved_count / (i + 1):.2%}"}
            )

    # --- 7c. 打印最终摘要 ---
    avg_steps = total_steps_to_solve / solved_count if solved_count > 0 else 0
    solve_rate = solved_count / len(eval_problems)
    print(f"\n✅ Final Solve Rate on {len(problems_raw)} problems: {solve_rate:.2%}")
    print(f"📊 Average steps to solve (for solved problems): {avg_steps:.2f}")


if __name__ == "__main__":
    main()