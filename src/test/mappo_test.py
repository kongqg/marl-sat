# --- 1. 测试参数设置 ---
import jax
import jax.numpy as jnp
from src.envs.multi_agent_sat_env import SATEnv
from src.learners.mappo_gnn_sat_learner import SATDataWrapper, GNNWrapperState
from src.learners.mappo_gnn_sat_learner import GNNEncoder

NUM_VARS = 20
NUM_CLAUSES = 40
MAX_STEPS = 10
HIDDEN_DIM = 128  # 必须与 GNNEncoder 中的 hidden_dim 一致

# --- 2. 创建一个随机的 SAT 问题实例 (clauses) ---
key = jax.random.PRNGKey(0)
key, key_clauses = jax.random.split(key)

# 创建随机的文字 (literals), 范围 [-NUM_VARS, -1] U [1, NUM_VARS]
literals = jax.random.randint(key_clauses,
                              shape=(NUM_CLAUSES, 3),
                              minval=1,
                              maxval=NUM_VARS + 1)
signs = jax.random.choice(key_clauses, jnp.array([-1, 1]), shape=(NUM_CLAUSES, 3))
problem_clauses = literals * signs

# --- 3. 初始化环境和我们强大的 SATDataWrapper ---
print("--> 正在初始化环境和 Wrapper...")
env = SATEnv(num_vars=NUM_VARS, num_clauses=NUM_CLAUSES, max_steps=MAX_STEPS)
wrapped_env = SATDataWrapper(env)
print("✅ 初始化成功!")

# --- 4. 测试 reset 方法 ---
print("\n--> 正在测试 wrapped_env.reset()...")
key, key_reset = jax.random.split(key)

# 修正了所有 JAX 错误后的 reset 调用
obs, state = wrapped_env.reset(problem_clauses, key_reset)

print("\n--- Reset 方法验证 ---")
assert 'gnn_input' in obs, "失败: 'gnn_input' 未在 observation 中找到!"
print("✅ 成功: 'gnn_input' 已成功注入到 observation 中。")
assert isinstance(state, GNNWrapperState), "失败: reset 返回的状态不是 GNNWrapperState 类型!"
print("✅ 成功: reset 返回了正确的 GNNWrapperState 状态容器。")

gnn_input_data = obs['gnn_input']
print("\n--- GNNInput 内容检查 (Reset) ---")
print(f"  - A_pos shape: {gnn_input_data.A_pos.shape}")
print(f"  - 赋值 (assignment) shape: {gnn_input_data.assignment.shape}")
print(f"  - 子句特征 (clause_features) shape: {gnn_input_data.clause_features.shape}")

assert gnn_input_data.A_pos.shape == (NUM_VARS, NUM_CLAUSES)
assert gnn_input_data.assignment.shape == (NUM_VARS,)
assert gnn_input_data.clause_features.shape == (NUM_CLAUSES, 3)
print("✅ 成功: GNNInput 中的数据形状正确。")

# --- 5. 测试 step 方法 ---
print("\n--> 正在测试 wrapped_env.step()...")

# 创建一个随机的 agent 动作字典
key, key_actions = jax.random.split(key)
actions = {}
for agent_name, var_group in env.agent_groups.items():
    num_vars_for_agent = len(var_group)
    key, subkey = jax.random.split(key_actions)
    actions[agent_name] = jax.random.randint(subkey, shape=(num_vars_for_agent,), minval=0, maxval=2)

# 调用 step
key, key_step = jax.random.split(key)
next_obs, next_state, reward, done, info = wrapped_env.step(key_step, state, actions)

print("\n--- Step 方法验证 ---")
assert 'gnn_input' in next_obs, "失败: 'gnn_input' 未在下一步的 observation 中找到!"
print("✅ 成功: 'gnn_input' 已成功注入到下一步的 observation 中。")
assert isinstance(next_state, GNNWrapperState), "失败: step 返回的状态不是 GNNWrapperState 类型!"
print("✅ 成功: step 返回了正确的 GNNWrapperState 状态容器。")
assert next_state.env_state.step == 1, "失败: 环境步数未正确增加。"
print("✅ 成功: 环境步数已正确更新。")

# --- 6. 测试 GNNEncoder 集成 ---
print("\n--> 正在测试 GNNEncoder 集成...")

gnn_encoder = GNNEncoder(hidden_dim=HIDDEN_DIM, num_message_passing_step=2)  # 测试时减少步数以加快速度
key, key_gnn = jax.random.split(key)

# 初始化 GNN 参数
# 使用从 reset 中得到的 gnn_input 作为“形状模板”
params = gnn_encoder.init(key_gnn, gnn_input_data)
print("✅ 成功: GNNEncoder 参数初始化成功。")

# 执行一次前向传播
H_v_pos, H_v_neg, H_c = gnn_encoder.apply(params, gnn_input_data)
print("✅ 成功: GNNEncoder 前向传播执行成功。")

print("\n--- GNN 输出形状验证 ---")
print(f"  - H_v_pos shape: {H_v_pos.shape}")
print(f"  - H_v_neg shape: {H_v_neg.shape}")
print(f"  - H_c shape: {H_c.shape}")

assert H_v_pos.shape == (NUM_VARS, HIDDEN_DIM)
assert H_v_neg.shape == (NUM_VARS, HIDDEN_DIM)
assert H_c.shape == (NUM_CLAUSES, HIDDEN_DIM)
print("✅ 成功: GNN 输出的嵌入矩阵形状正确。")

print("\n\n🎉🎉🎉 恭喜！所有测试已通过，您的第一阶段代码已准备就绪！ 🎉🎉🎉")