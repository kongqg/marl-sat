import jax
import flax
from flax.training import checkpoints, train_state
from src.runners.mappo_runner import set_global_seeds
from src.envs.multi_agent_sat_env import SATEnv
from src.learners.mappo_gnn_sat_learner import GNN_ActorCritic, SATDataWrapper
import jax.numpy as jnp
import optax


def inspect_model(ckpt_path: str, model_prefix: str, step: int, action_mode_to_init: int):
    """加载一个 Flax 检查点并打印其参数结构。"""
    print(f"--- 正在检查模型: {ckpt_path} ---")
    print(f"--- 假设模型是用 action_mode = {action_mode_to_init} 初始化的 ---")

    # 1. 创建一个与保存时结构相匹配的“模板”网络和状态
    env = SATEnv(num_vars=30, num_clauses=128, action_mode=action_mode_to_init,max_steps=256)
    key = set_global_seeds(42)
    key, net_key = jax.random.split(key)

    network = GNN_ActorCritic(
        gnn_hidden_dim=128,
        gnn_num_message_passing_steps=12,
        num_agents=env.num_agents,
        max_vars_per_agent=env.max_vars_per_agent,
        action_mode=action_mode_to_init
    )

    # 创建一个虚拟输入来初始化模板
    dummy_clauses = jnp.ones((128, 3), dtype=jnp.int32)
    wrapper_env = SATDataWrapper(env)
    obs, _ = wrapper_env.reset(dummy_clauses, key)
    dummy_gnn_input = obs['gnn_input']

    # 初始化模板参数和状态
    params = network.init(net_key, dummy_gnn_input, env.agent_vars, env.action_mask)['params']
    tx = optax.adam(1e-4)
    template_state = train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # 2. 尝试从文件中恢复状态
    try:
        loaded_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_path,
            target=template_state,
            prefix=model_prefix,
            step=step
        )
        if loaded_state is None:
            print("\n[错误] 无法加载模型，restore_checkpoint 返回 None。")
            print("可能的原因：路径、前缀或步骤号不正确。")
            return

        # 3. 打印加载成功的模型参数的顶层键
        print("\n[成功] 模型加载成功！参数结构如下:")
        param_keys = loaded_state.params.keys()
        print(f"  - 顶层参数键: {list(param_keys)}")

        # 4. 检查关键的 Actor 头是否存在
        print("\n--- 关键层检查 ---")
        if 'var_flip_head' in param_keys:
            print("  ✅ 找到了 'var_flip_head' (对应 action_mode=0)")
        else:
            print("  ❌ 未找到 'var_flip_head'")

        if 'var_actor_head' in param_keys:
            print("  ✅ 找到了 'var_actor_head' (对应 action_mode=1)")
        else:
            print("  ❌ 未找到 'var_actor_head'")

    except Exception as e:
        print(f"\n[错误] 加载模型时发生异常: {e}")
        print("这通常意味着模板的网络结构与保存的模型结构不匹配。")


if __name__ == '__main__':
    BC_MODEL_PATH = "models/bc_pretrained"
    BC_MODEL_PREFIX = "bc_model_"
    # 这里填写你保存时用的 step/epoch 数，例如 1 或 100
    BC_MODEL_STEP = 120

    # --- 尝试用两种模式去解读模型文件 ---
    # 假设它是用 action_mode=0 保存的
    inspect_model(BC_MODEL_PATH, BC_MODEL_PREFIX, BC_MODEL_STEP, action_mode_to_init=0)

    print("\n" + "="*60 + "\n")

    # 假设它是用 action_mode=1 保存的
    inspect_model(BC_MODEL_PATH, BC_MODEL_PREFIX, BC_MODEL_STEP, action_mode_to_init=1)