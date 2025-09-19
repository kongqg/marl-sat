import random
import chex
import jax
import os
import numpy as np
from tqdm import tqdm

# 确保这些导入路径与您的项目结构一致
from src.envs.multi_agent_sat_env import SATEnv
from src.utils.data_parser import parse_cnf


def load_cnf_problems(cnf_data_dir: str):
    #
    cnf_fnames = sorted([f for f in os.listdir(cnf_data_dir) if f.endswith('.cnf')])
    problems = []
    print(f"在目录 '{cnf_data_dir}' 中找到了 {len(cnf_fnames)} 个 SAT 实例。")
    for fname in tqdm(cnf_fnames, desc="正在加载CNF问题"):
        cnf_path = os.path.join(cnf_data_dir, fname)
        num_vars, num_clauses, clauses = parse_cnf(cnf_path)
        problems.append({
            "num_vars": num_vars,
            "num_clauses": num_clauses,
            "clauses": clauses,
            "name": fname
        })
    return problems


def global_seed(seed: int) -> chex.PRNGKey:
    #
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    print(f"🌱 全局种子已为 random, numpy, 和 jax 设置为 {seed}。")
    return key


def single_problem(key: chex.PRNGKey, sat_problem: dict):
    """
    对单个SAT问题实例运行一个完整的“无策略”测试。
    """
    print("\n" + "=" * 50)
    print(f"▶️  开始测试问题: {sat_problem['name']}")
    print("=" * 50)

    # 1. 为当前问题初始化环境
    # reward type，  问题的大小
    num_vars = sat_problem["num_vars"]
    num_clauses = sat_problem["num_clauses"]
    env = SATEnv(num_vars,num_clauses, max_steps=500)
    print(f"✅ 环境已初始化，包含 {env.num_agents} 个智能体。")

    # 2. 重置环境
    key, reset_key = jax.random.split(key)

    # 返回一个示例
    obs, state = env.reset(key = reset_key,problem_clauses=sat_problem["clauses"])
    print(f"初始状态: {state.num_unsatisfied} 个子句未满足。")

    # 3. 运行随机动作循环
    num_steps_to_run = 500
    print(f"🤖 正在以随机动作运行 {num_steps_to_run} 步...")

    episodes = 0
    for step_num in tqdm(range(num_steps_to_run)):
        key, act_key, step_key = jax.random.split(key, 3)

        # 为每个智能体采样随机动作
        act_keys = jax.random.split(act_key, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(act_keys[i])
            for i, agent in enumerate(env.agents)
        }

        # 执行一步
        obs, next_state, rewards, dones, infos = env.step_env(key, state, actions)
        # 如果 episode 结束，打印信息并重置
        if dones["__all__"]:
            # 注意：打印 next_state，而不是 state
            print(f"step(after) = {int(next_state.step)}")
            print(f"unsat(before -> after) = {int(state.num_unsatisfied)} -> {int(next_state.num_unsatisfied)}")
            solved = bool(next_state.num_unsatisfied == 0)
            timed_out = bool(next_state.step >= env.max_steps)
            print(f"solved={solved}, timed_out={timed_out}")
            state = next_state  # （可选）保持一致性
            print(f"solution:{next_state.variable_assignments}")
            break
        else:
            state = next_state

    print(f"🏁 问题 {sat_problem['name']} 测试完成。")


# --- 主程序入口 ---
print("🚀 启动 SATEnv 无策略测试脚本...")
# --- 准备工作 ---
# 1. 稳健地处理路径
# script_dir = os.path.dirname(__file__)
cnf_dir_abs =  f"../../data/uf20-91"

# 2. 加载并打乱问题
problems = load_cnf_problems(cnf_dir_abs)
main_key = global_seed(42)
random.shuffle(problems)

# --- 循环测试多个问题 ---
# 我们可以选择测试前几个问题，例如前3个
num_problems_to_test = 3
for problem in problems[:num_problems_to_test]:
    # 3. (关键修正) 为每个问题的测试“分裂”出一个独立的key
    main_key, problem_key = jax.random.split(main_key)

    # 4. 调用封装好的测试函数
    single_problem(problem_key, problem)

print("\n✅ 所有测试已成功完成！")

