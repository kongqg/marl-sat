import os
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax import struct
from flax.training import checkpoints, train_state
from tqdm import tqdm
from typing import List, Dict, Tuple
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf

from src.runners.single_rl_runner import set_global_seeds
from src.envs.multi_agent_sat_env import SATEnv
from src.learners.mappo_gnn_sat_learner import GNN_ActorCritic, SATDataWrapper
from src.utils.data_parser import load_cnf_problems
from src.utils.graph_constructor import create_static_graph, GNNInput
from functools import partial


@struct.dataclass
class ExpertSample:
    problem_clauses: jnp.ndarray
    expert_solution: jnp.ndarray


def load_expert_data(cnf_dir: str, sol_dir: str) -> List[ExpertSample]:
    # ... (此函數保持不變) ...
    print(f"[*] Loading expert data from {cnf_dir} and {sol_dir}...")
    problems_raw = load_cnf_problems(cnf_dir)
    problem_dict = {p['name']: p for p in problems_raw}
    expert_data = []
    solution_files = sorted([f for f in os.listdir(sol_dir) if f.endswith('.sol')])
    for sol_file in solution_files:
        cnf_name = sol_file.replace('.sol', '.cnf')
        if cnf_name in problem_dict:
            problem = problem_dict[cnf_name]
            with open(os.path.join(sol_dir, sol_file), 'r') as f:
                line = f.readline().strip()
                solution_list = [int(v) for v in line.split()]
                solution = np.array([max(0, v) for v in solution_list])
            expert_data.append(ExpertSample(
                problem_clauses=jnp.array(problem['clauses']),
                expert_solution=jnp.array(solution)
            ))
    print(f"[+] Loaded {len(expert_data)} expert problem-solution pairs.")
    return expert_data


# <--- 核心修改: 新的專家標籤生成函數 (實施方案 1) --->
def compute_joint_labels_parallel_greedy(env: SATEnv,
                                         clauses: np.ndarray,
                                         assignments: np.ndarray,
                                         tau: float) -> np.ndarray:
    """
    為每個 agent 並行地計算貪心動作，生成一個聯合動作標籤。
    【已修正索引映射 Bug】
    """
    num_agents = env.num_agents
    labels = []

    _, base_unsat = env._calculate_satisfaction_explicit(assignments, clauses)

    for i in range(num_agents):
        valid_mask = np.array(env.action_mask[i])

        # <--- 核心修正 START --->
        # 1. 獲取有效動作在 0..max_vars_per_agent-1 範圍內的原始索引
        valid_local_indices = np.flatnonzero(valid_mask)
        # 2. 根據原始索引獲取對應的全局變數 ID
        local_vars_global_indices = np.array(env.agent_vars[i][valid_local_indices])
        # <--- 核心修正 END --->

        best_delta = 0.0
        best_local_idx = env.max_vars_per_agent  # 預設為 no-op

        if len(local_vars_global_indices) > 0:
            # 遍歷有效動作
            for j, global_var_idx in enumerate(local_vars_global_indices):
                temp_assignments = np.copy(assignments)
                temp_assignments[global_var_idx] ^= 1
                _, new_unsat = env._calculate_satisfaction_explicit(temp_assignments, clauses)

                delta = float(new_unsat - base_unsat)
                if delta < best_delta:
                    best_delta = delta
                    # <--- 核心修正 START --->
                    # 3. 記錄的不再是壓縮索引 j，而是原始的本地索引
                    best_local_idx = valid_local_indices[j]
                    # <--- 核心修正 END --->

        if best_delta < tau:
            labels.append(best_local_idx)
        else:
            labels.append(env.max_vars_per_agent)

    return np.array(labels, dtype=np.int32)


def preprocess_and_save_data(expert_data: List[ExpertSample],
                             env: SATEnv,
                             config: dict,
                             save_path: str):
    bc_config = config.get('bc_training', {})
    num_samples_per_expert = bc_config.get('NUM_SAMPLES_PER_EXPERT', 5)
    corruption_level = bc_config.get('CORRUPTION_LEVEL', 3)
    tau_improve = bc_config.get('TAU_IMPROVE', 0.0)

    if os.path.exists(save_path):
        print(f"[*] Preprocessed data found at {save_path}. Skipping generation.")
        return

    print(f"[*] Starting offline data preprocessing with 'parallel_joint' mode...")
    all_gnn_inputs = []
    all_expert_actions = {agent: [] for agent in env.agents}
    key = jax.random.PRNGKey(0)

    pbar = tqdm(expert_data, desc="Preprocessing Expert Samples")
    for sample in pbar:
        for _ in range(num_samples_per_expert):
            problem_clauses = np.array(sample.problem_clauses)
            expert_solution = np.array(sample.expert_solution)

            key, subkey = jax.random.split(key)
            vars_to_flip = jax.random.choice(subkey, env.num_vars, shape=(corruption_level,), replace=False)
            corrupted_assignments = np.copy(expert_solution)
            corrupted_assignments[vars_to_flip] = 1 - corrupted_assignments[vars_to_flip]

            # <--- 核心修改: 使用新的並行專家來生成聯合標籤 --->
            joint_labels = compute_joint_labels_parallel_greedy(env,
                                                                problem_clauses,
                                                                corrupted_assignments,
                                                                tau=tau_improve)

            for i_agent, agent in enumerate(env.agents):
                all_expert_actions[agent].append(joint_labels[i_agent])

            # (GNN Input 的構建邏輯保持不變)
            dummy_state, static_graph = env.reset_for_bc(jnp.array(problem_clauses), jnp.array(corrupted_assignments))
            wrapper = SATDataWrapper(env)
            gnn_input = wrapper._state_to_gnn_input(dummy_state, static_graph)
            all_gnn_inputs.append(gnn_input)

    print("[*] Aggregating preprocessed data...")
    batch_gnn_inputs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_gnn_inputs)
    batch_expert_actions = {k: jnp.array(v) for k, v in all_expert_actions.items()}
    print(f"[*] Saving preprocessed data to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump((batch_gnn_inputs, batch_expert_actions), f)
    print("[+] Preprocessing finished.")


# <--- 核心修改: 評估時每個 agent 並行決策 (實施方案 3) --->
@partial(jax.jit, static_argnames=("env", "network", "max_steps"))
def evaluate_policy_as_solver(key, env: SATDataWrapper, network: GNN_ActorCritic, params, problem_clauses, max_steps):
    key, reset_key = jax.random.split(key)
    (local_obs, global_state), env_state = env.reset(problem_clauses, reset_key)

    def _one_step_eval(carry, _):
        env_state, global_state, key = carry
        actor_fn = partial(network.apply, method=GNN_ActorCritic.apply_actor)
        pi = actor_fn({'params': params}, global_state, env.agent_vars, env.action_mask)

        # --- 【實施方案 3】: 每個 agent 獨立 argmax，實現並行決策 ---
        # 準備完整的動作掩碼 (包括 no-op)
        full_mask = jnp.concatenate([env.action_mask, jnp.ones((env.num_agents, 1), dtype=bool)], axis=-1)
        # 應用掩碼
        masked_logits = jnp.where(full_mask, pi.logits, -jnp.inf)
        # 每個 agent 獨立取最優動作
        actions_array = jnp.argmax(masked_logits, axis=1)  # Shape: (A,)

        # actions_dict = {agent: actions_array[i] for i, agent in enumerate(env.agents)}

        key, step_key = jax.random.split(key)
        actions_dict = {agent: actions_array[i] for i, agent in enumerate(env.agents)}
        (next_local_obs, next_global_state), next_state, reward, done, info = env.step(key=step_key, state=env_state,
                                                                                       actions=actions_dict)
        return (next_state, next_global_state, key), (done, info, next_state.env_state.variable_assignments)

    _, (dones, infos, all_assignments) = jax.lax.scan(_one_step_eval, (env_state, global_state, key), None,
                                                      length=max_steps)
    was_ever_solved = jnp.any(infos['solved'])
    first_solve_step_index = jnp.argmax(infos['solved'])
    solution = all_assignments[first_solve_step_index]
    no_solution_placeholder = jnp.full_like(solution, -1)
    solution_assignments = jnp.where(was_ever_solved, solution, no_solution_placeholder)
    steps_to_solve = jnp.where(was_ever_solved, first_solve_step_index + 1, max_steps)
    return was_ever_solved, steps_to_solve, solution_assignments


@hydra.main(version_base=None, config_path="../../configs", config_name="MAPPO_CONFIG.yaml")
def main(hydra_config: DictConfig):
    config = OmegaConf.to_container(hydra_config, resolve=True)

    # --- 1. 配置和初始化 ---
    train_config = config.get('training', {})
    env_config = config.get('environment', {})

    SAVE_PATH = os.path.abspath("models/bc_pretrained")
    PREPROCESSED_DATA_PATH = os.path.join(SAVE_PATH, "preprocessed_bc_data.pkl")
    LOG_FILE_PATH = os.path.join(SAVE_PATH, "bc_training_log.txt")
    SOLUTIONS_LOG_PATH = os.path.join(SAVE_PATH, "solver_solutions_log.txt")

    os.makedirs(SAVE_PATH, exist_ok=True)
    env = SATEnv(num_vars=env_config["NUM_VARS"], num_clauses=env_config["NUM_CLAUSES"],
                 max_steps=env_config["MAX_STEPS"], action_mode=env_config["action_mode"])
    key = set_global_seeds(config.get('SEED', 42))
    key, net_key = jax.random.split(key)

    network = GNN_ActorCritic(
        gnn_hidden_dim=config['network']['GNN_HIDDEN_DIM'],
        gnn_num_message_passing_steps=config['network']['GNN_NUM_MESSAGE_PASSING_STEPS'],
        num_agents=env.num_agents, max_vars_per_agent=env.max_vars_per_agent,
        action_mode=env_config["action_mode"]
    )
    dummy_clauses = jnp.ones((env_config["NUM_CLAUSES"], 3), dtype=jnp.int32)
    dummy_assignments = jnp.zeros((env_config["NUM_VARS"],), dtype=jnp.int32)
    dummy_state, static_graph = env.reset_for_bc(dummy_clauses, dummy_assignments)
    wrapper = SATDataWrapper(env)
    dummy_gnn_input = wrapper._state_to_gnn_input(dummy_state, static_graph)
    params = network.init(net_key, dummy_gnn_input, env.agent_vars, env.action_mask)['params']
    tx = optax.adam(train_config['LEARNING_RATE'])
    ts = train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # --- 2. 載入或生成預處理資料 ---
    expert_data = load_expert_data(config['CNF_DATA_DIR'], "data/uf20-91-answer")
    preprocess_and_save_data(expert_data, env, config, PREPROCESSED_DATA_PATH)
    print(f"[*] Loading preprocessed data from {PREPROCESSED_DATA_PATH}...")
    with open(PREPROCESSED_DATA_PATH, 'rb') as f:
        all_gnn_inputs, all_expert_actions = pickle.load(f)
    print("[+] Preprocessed data loaded.")
    num_total_samples = jax.tree_util.tree_leaves(all_gnn_inputs)[0].shape[0]

    # --- 3. 定義訓練步驟 ---
    # <--- 核心修改: 聯合交叉熵損失 (實施方案 2) --->
    @partial(jax.jit, static_argnames=("network", "env"))
    def train_step(ts: train_state.TrainState, batch_gnn_inputs: GNNInput, batch_labels: Dict,
                   network: nn.Module, env: SATEnv):
        def loss_fn(params):
            labels_array = jnp.stack([batch_labels[agent] for agent in env.agents], axis=1)  # Shape: (B, A)
            actor_fn = partial(network.apply, method=GNN_ActorCritic.apply_actor)
            pi = jax.vmap(actor_fn, in_axes=(None, 0, None, None))(
                {'params': params}, batch_gnn_inputs, env.agent_vars, env.action_mask
            )

            # pi 是一個 (B, A) 的 Categorical 分佈
            # labels_array 也是 (B, A) 的標籤
            # log_prob 會返回 (B, A) 的對數機率
            log_probs = pi.log_prob(labels_array)

            # 直接對所有 agent 的 log_prob 取平均，計算聯合交叉熵
            joint_cross_entropy_loss = -jnp.mean(log_probs)
            return joint_cross_entropy_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(ts.params)
        ts = ts.apply_gradients(grads=grads)
        return ts, loss

    # --- 4. 執行訓練迴圈 ---
    print("\n[*] Starting Behavioral Cloning pre-training (parallel_joint mode)...")
    with open(LOG_FILE_PATH, 'w') as log_file:
        log_file.write("epoch,loss\n")
        pbar = tqdm(range(train_config['NUM_UPDATES']))  # 使用 config 中的 epochs
        for epoch in pbar:
            permutation = np.random.permutation(num_total_samples)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, num_total_samples, train_config['MINIBATCH_SIZE']):
                indices = permutation[i:i + train_config['MINIBATCH_SIZE']]
                batch_gnn_inputs = jax.tree_util.tree_map(lambda x: x[indices], all_gnn_inputs)
                batch_expert_actions = jax.tree_util.tree_map(lambda x: x[indices], all_expert_actions)
                ts, loss = train_step(ts, batch_gnn_inputs, batch_expert_actions, network, env)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            pbar.set_description(f"Epoch {epoch + 1}/{train_config['NUM_UPDATES']} | Loss: {avg_loss:.4f}")
            log_file.write(f"{epoch + 1},{avg_loss:.6f}\n")

    # --- 5. 儲存預訓練模型 ---
    checkpoints.save_checkpoint(
        ckpt_dir=SAVE_PATH,
        target={'params': ts.params},
        step=train_config['NUM_UPDATES'],
        prefix="bc_model_",
        overwrite=True
    )
    print(f"\n[+] Pre-training finished. Model saved to {SAVE_PATH}")
    print(f"[+] Training log saved to {LOG_FILE_PATH}")

    # --- 6. 執行求解器評估 ---
    # (此部分與上一版幾乎相同，僅決策邏輯已在函數內部更新)
    print("\n[*] Starting evaluation of the BC model as a complete solver...")
    final_model_state = checkpoints.restore_checkpoint(
        ckpt_dir=SAVE_PATH,
        target={'params': ts.params},
        step=train_config['NUM_UPDATES'],
        prefix="bc_model_"
    )
    if final_model_state is None:
        print(f"❌ Error: Checkpoint not found at {SAVE_PATH}. Cannot perform evaluation.")
        return

    final_params = final_model_state['params']
    print("[+] Final trained model loaded for evaluation as a solver.")
    print(f"[*] Solutions will be saved to: {SOLUTIONS_LOG_PATH}")
    eval_env = SATDataWrapper(env)
    problems_raw = load_cnf_problems(config['CNF_DATA_DIR'])
    solved_count = 0
    total_steps_for_solved = 0
    key, eval_key_base = jax.random.split(key)
    eval_keys = jax.random.split(eval_key_base, len(problems_raw))
    with open(SOLUTIONS_LOG_PATH, 'w') as log_file:
        header = f"# Format: CNF_File, Solved_Status, Steps_To_Solve, Is_Solution_Valid, Predicted_Solution\n"
        log_file.write(header)
        pbar_eval = tqdm(enumerate(problems_raw), total=len(problems_raw), desc="Solving Problems")
        for i, problem in pbar_eval:
            problem_name = problem['name']
            problem_clauses = jnp.array(problem['clauses'])
            solved, steps, solution = evaluate_policy_as_solver(
                eval_keys[i], eval_env, network, final_params,
                problem_clauses, max_steps=env.max_steps
            )
            solution_np = np.array(solution)
            is_valid_solution = False
            if solved:
                solved_count += 1
                total_steps_for_solved += steps
                _, num_unsatisfied = env._calculate_satisfaction_explicit(solution_np, problem_clauses)
                if num_unsatisfied == 0:
                    is_valid_solution = True
            solution_str = "".join(solution_np.astype(str)) if solved else "N/A"
            log_line = f"{problem_name}, {solved}, {steps}, {is_valid_solution}, {solution_str}\n"
            log_file.write(log_line)
            pbar_eval.set_postfix({"solve_rate": f"{solved_count / (i + 1):.2%}"})
    final_solve_rate = solved_count / len(problems_raw) if problems_raw else 0
    avg_steps = total_steps_for_solved / solved_count if solved_count > 0 else 0
    print("\n" + "=" * 60)
    print("      BC Model as a Solver - Evaluation Results")
    print("=" * 60)
    print(f"  Total Problems Tested: {len(problems_raw)}")
    print(f"  Problems Solved:       {solved_count}")
    print(f"  Final Solve Rate:      {final_solve_rate:.2%}")
    print(f"  Avg. Steps to Solve:   {avg_steps:.2f} (for solved problems)")
    print("=" * 60)
    print(f"✅ Detailed solutions saved to {SOLUTIONS_LOG_PATH}")


if __name__ == '__main__':
    def reset_for_bc(self, problem_clauses, assignments):
        literal_to_agent_idx = self.variable_to_agent_idx[jnp.abs(jnp.array(problem_clauses)) - 1]
        agent_clause_masks, agent_neighbor_masks = self._compute_observation_maps(jnp.array(problem_clauses))
        clauses_satisfied_status, num_unsatisfied = self._calculate_satisfaction_explicit(
            assignments, jnp.array(problem_clauses)
        )
        state = SATEnv.SATState(
            variable_assignments=assignments, clauses_satisfied_status=clauses_satisfied_status,
            num_unsatisfied=num_unsatisfied, step=0, done=jnp.array([False] * self.num_agents),
            clauses=jnp.array(problem_clauses), agent_clause_masks=agent_clause_masks,
            agent_neighbor_masks=agent_neighbor_masks, literal_to_agent_idx=literal_to_agent_idx,
            action_mask=self.action_mask
        )
        static_graph = create_static_graph(
            num_vars=self.num_vars, num_clauses=self.num_clauses, clauses=problem_clauses
        )
        return state, static_graph


    from src.envs.multi_agent_sat_env import SATState

    SATEnv.SATState = SATState
    SATEnv.reset_for_bc = reset_for_bc
    main()