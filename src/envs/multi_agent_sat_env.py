import math
from functools import partial
from typing import Tuple, Dict, Optional, List

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments import State, spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Space


@chex.dataclass(frozen=True)
class SATState(State):
    variable_assignments: jnp.ndarray
    clauses_satisfied_status: jnp.ndarray
    num_unsatisfied: jnp.ndarray

    # 静态部分
    clauses: jnp.ndarray
    agent_clause_masks: jnp.ndarray  # shape: (num_agents, num_clauses)
    agent_neighbor_masks: jnp.ndarray  # shape: (num_agents, num_vars)
    literal_to_agent_idx: jnp.ndarray  # shape: (num_clauses, 3)
    action_mask: jnp.ndarray



class SATEnv(MultiAgentEnv):
    def __init__(self, num_vars, num_clauses,
                 max_steps: int, vars_per_agent: Optional[int] = None,
                 action_mode: int = 0,r_clause: float = 0.02,
                 r_sat: float = 1.0,gamma: float = 0.99):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.agent_groups=self._create_agent_groups(num_vars=self.num_vars,
                                                    vars_per_agent=vars_per_agent)
        self.agents = list(self.agent_groups.keys())
        self.num_agents = len(self.agents)
        self.agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}
        self.r_clause = r_clause
        self.r_sat = r_sat
        self.gamma = gamma

        # 决定agent flip掉的变量
        self.action_mode = action_mode
        super().__init__(num_agents=self.num_agents)

        # 一次性flip掉全部的变量
        ''' eg.
            action = {
                'agent_0': [0,0,1,1,1....,num_vars],
                'agent_1': [1,1,1,0,.....,num_vars],
                ....
            }
        '''
        # 计算每个agent 负责的最大变量数
        self.max_vars_per_agent = 0
        if self.agent_groups:
            self.max_vars_per_agent = max(len(v) for v in self.agent_groups.values())

        self.agent_vars = jnp.full((self.num_agents,self.max_vars_per_agent), -1, dtype=jnp.int32)
        self.action_mask = jnp.zeros((self.num_agents, self.max_vars_per_agent), dtype=jnp.bool_)

        for i, agent in enumerate(self.agents):
            vars_list = self.agent_groups[agent]
            self.agent_vars = self.agent_vars.at[i, :len(vars_list)].set(jnp.array(vars_list))
            self.action_mask = self.action_mask.at[i, : len(vars_list)].set(True)
        self.action_spaces = {}
        for agent in self.agents:
            # --- 使用数字进行判断 ---
            if self.action_mode == 0:  # single_flip
                num_actions = self.max_vars_per_agent + 1
                action_space_obj = self.action_spaces[agent] = spaces.Discrete(num_actions)
            else:  # multi_flip (action_mode == 1)
                action_categories = [2] * self.max_vars_per_agent
                action_space_obj = self.action_spaces[agent] = spaces.MultiDiscrete(action_categories)

            action_space_obj.dtype = jnp.int32
            self.action_spaces[agent] = action_space_obj

        obs_dim = self._calculate_obs_dim()

        # -1 represents padding
        self.observation_spaces = {agent: spaces.Box(-1, 1, (obs_dim,)) for agent in self.agents}

        self.max_steps = max_steps
        
        self.variable_to_agent_idx = self._compute_variable_to_agent_map()



    def _compute_variable_to_agent_map(self) -> jnp.ndarray:
        variable_to_agent_idx = jnp.full((self.num_vars,), -1, dtype=jnp.int32)
        for i, agent in enumerate(self.agents):
            agent_vars = jnp.array(self.agent_groups[agent], dtype=jnp.int32)
            variable_to_agent_idx = variable_to_agent_idx.at[agent_vars].set(i)
        return variable_to_agent_idx

    def _compute_observation_maps(self, clauses: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        variable_indices_in_clauses = jnp.abs(clauses) - 1

        # --- 1. 计算 agent_clause_masks ---
        def get_clause_mask_for_agent(agent_vars_padded_row):
            # (num_clauses, 3, max_vars_per_agent)
            # matches[i, j, k] 的值为 True，当且仅当“第 i 个子句的第 j 个文字所对应的变量”，与“当前智能体负责的第 k 个变量”是同一个
            matches = (variable_indices_in_clauses[..., None] == agent_vars_padded_row[None, None, :])
            related_mask = jnp.any(matches, axis=(1, 2))
            # (num_clauses,)
            return related_mask.astype(jnp.int32)

        agent_clause_masks = jax.vmap(get_clause_mask_for_agent)(self.agent_vars)
        agent_clause_masks = jnp.where(agent_clause_masks, 1, -1)

        # --- 2. 计算 agent_neighbor_masks ---
        def get_neighbor_mask_for_agent(agent_idx, agent_vars_padded_row):
            related_clauses_for_agent = agent_clause_masks[agent_idx] == 1
            relevant_vars = jnp.where(related_clauses_for_agent[:, None], variable_indices_in_clauses, -1)
            unique_related_vars = jnp.unique(relevant_vars, size=self.num_vars + 1, fill_value=-1)

            is_own_var_mask = jnp.any(jnp.arange(self.num_vars)[:, None] == agent_vars_padded_row[None, :], axis=1)
            is_related_var_mask = jnp.any(jnp.arange(self.num_vars)[:, None] == unique_related_vars[None, :], axis=1)

            is_neighbor_mask = is_related_var_mask & ~is_own_var_mask
            return jnp.where(is_neighbor_mask, 1, -1)

        agent_neighbor_masks = jax.vmap(get_neighbor_mask_for_agent)(jnp.arange(self.num_agents), self.agent_vars)

        return agent_clause_masks, agent_neighbor_masks

    def _calculate_satisfaction_explicit(
            self, variable_assignments: chex.Array,clauses: jnp.ndarray
    ) -> Tuple[chex.Array, chex.Array]:
        # valid_literals_mask = (self.clauses != 0) # ()
        # 1、获取clauses每个literal的下标
        variable_indices = jnp.abs(clauses) - 1 # shape: (num_clauses, 3)

        # advancing indexing
        # 2、计算每个子句中为true的元素
        assignments_for_each_literal = variable_assignments[variable_indices] #
        # 分别计算这两种情况， shape: (num_clauses, 3)
        positive_literals_are_true = (clauses > 0) & (assignments_for_each_literal == 1)
        negative_literals_are_true = (clauses < 0) & (assignments_for_each_literal == 0)

        literal_truth_values = positive_literals_are_true | negative_literals_are_true

        # filter mask value
        # literal_truth_values = literal_truth_values & valid_literals_mask

        # 4、计算最后的结果
        # 对行进行 or 有一个true就行, shape: (num_clauses,)
        clauses_satisfied_status = jnp.any(literal_truth_values, axis=1)

        # 5、取反得到unsat clauses
        num_unsatisfied = jnp.sum(~clauses_satisfied_status)
        # (num_clauses,),
        return clauses_satisfied_status, num_unsatisfied

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, problem_clauses: jnp.ndarray ,key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        literal_to_agent_idx = self.variable_to_agent_idx[jnp.abs(jnp.array(problem_clauses)) - 1]
        agent_clause_masks, agent_neighbor_masks = self._compute_observation_maps(jnp.array(problem_clauses))
        variable_assignments = jax.random.randint(key, shape=(self.num_vars,), minval=0, maxval=2)
        clauses_satisfied_status, num_unsatisfied = self._calculate_satisfaction_explicit(
            variable_assignments, jnp.array(problem_clauses)
        )
        state = SATState(
            variable_assignments=variable_assignments,
            clauses_satisfied_status=clauses_satisfied_status,
            num_unsatisfied=jnp.asarray(num_unsatisfied),
            step=jnp.asarray(0),
            done=jnp.array([False] * self.num_agents),
            clauses=jnp.array(problem_clauses),
            agent_clause_masks=agent_clause_masks,
            agent_neighbor_masks=agent_neighbor_masks,
            literal_to_agent_idx=literal_to_agent_idx,
            action_mask=self.action_mask
        )

        obs = self.get_obs(state)

        return obs, state

    def _calculate_rewards(self, state: SATState, next_state: SATState, solved: bool) -> Dict[str, float]:
        """
        极简稀疏奖励函数：
        - 只有在问题被解决的时刻，给予一个值为 1.0 的团队奖励。
        - 其他所有情况，奖励均为 0.0。
        """
        # 判断问题是否在“这一步”被解决
        # solved 是一个布尔值，直接代表 next_state 是否为解

        # 如果 solved 为 True，奖励为 1.0，否则为 0.0
        r_terminal = jnp.where(solved, 1.0, 0.0)

        # 所有智能体共享这一个奖励信号
        rewards = {agent: r_terminal for agent in self.agents}

        return rewards


    # def _calculate_rewards(self, state: SATState, next_state: SATState, solved: bool) -> Dict[str, float]:
    #     """
    #     Baseline（推荐）：共享团队奖励
    #     r_t = PBRS + (#新满足子句)*r_clause + [solved]*R_sat
    #     所有 agent 拿同一个奖励标量（不随人数缩小）。
    #     """
    #     # 1) PBRS（不再除以 num_agents）
    #     potential_old = -state.num_unsatisfied
    #     potential_new = -next_state.num_unsatisfied
    #     r_pbrs = self.gamma * potential_new - potential_old  # 标量
    #
    #     # 2) 新满足子句计数（只做计数，不按贡献分配）
    #     newly = (next_state.clauses_satisfied_status & ~state.clauses_satisfied_status).astype(jnp.float32)
    #     r_clause_total = jnp.sum(newly) * self.r_clause  # 标量
    #
    #     # 3) 终局大奖励（不再除以 num_agents）
    #     r_sat = jnp.where(solved, self.r_sat, 0.0)  # 标量
    #
    #     # 4) 组合共享奖励
    #     r_common = r_pbrs + r_clause_total + r_sat  # 标量
    #     rewards = {agent: r_common for agent in self.agents}
    #
    #     return rewards

    def step_env(
            self, key: chex.PRNGKey, state: SATState, actions_array: chex.Array
    ) -> Tuple[Dict[str, chex.Array], SATState, Dict[str, float], Dict[str, bool], Dict]:

        # --- 1. 动作应用与状态更新 (逻辑不变) ---
        new_assignments = state.variable_assignments
        # actions_array = jnp.stack([actions[agent] for agent in self.agents])

        if self.action_mode == 0:  # single_flip
            def get_var_to_flip(agent_idx, local_action_idx):
                num_agent_vars = jnp.sum(self.action_mask[agent_idx])
                is_no_op = (local_action_idx >= num_agent_vars)
                # 先將索引裁剪到安全範圍內，再進行索引
                safe_idx = jnp.minimum(local_action_idx, num_agent_vars - 1)
                global_var_idx = self.agent_vars[agent_idx, safe_idx]
                return jnp.where(is_no_op, -1, global_var_idx)

            vars_to_flip = jax.vmap(get_var_to_flip)(jnp.arange(self.num_agents), actions_array)
            flip_mask = jax.nn.one_hot(vars_to_flip, num_classes=self.num_vars).sum(axis=0)
            new_assignments = jnp.logical_xor(new_assignments, flip_mask).astype(jnp.int32)
        else:  # multi_flip
            valid_vars_to_update = self.agent_vars[self.action_mask]
            valid_actions = actions_array[self.action_mask]
            current_assignments_for_valid_vars = new_assignments[valid_vars_to_update]
            new_assignments_for_valid_vars = current_assignments_for_valid_vars ^ valid_actions
            new_assignments = new_assignments.at[valid_vars_to_update].set(new_assignments_for_valid_vars)

        new_clauses_status, new_num_unsatisfied = self._calculate_satisfaction_explicit(
            new_assignments, jnp.array(state.clauses)
        )

        # --- 2. 游戏结束判断 ---
        solved = (new_num_unsatisfied == 0)
        timed_out = (state.step + 1 >= self.max_steps)
        done = solved | timed_out
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done


        # --- 3. 创建新的 State 对象 ---
        next_state = state.replace(
            variable_assignments=new_assignments,
            clauses_satisfied_status=new_clauses_status,
            num_unsatisfied=new_num_unsatisfied,
            step=state.step + 1,
            done=jnp.array([done] * self.num_agents)
        )

        # --- 4. 调用独立的奖励函数 ---
        rewards = self._calculate_rewards(state, next_state, solved)

        # --- 5. 生成新的观观测并返回所有结果 ---
        obs = self.get_obs(next_state)
        infos = {
            "solved": solved,
            "num_unsatisfied": new_num_unsatisfied,
            "episode_step": state.step + 1,
        }

        return obs, next_state, rewards, dones, infos

    def _find_factors(self,n: int) -> List[int]:
        """找到一个整数的所有除数。"""
        factors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(list(factors))
    def _create_agent_groups(self, num_vars, vars_per_agent):

        if vars_per_agent is not None:
            print(f"User specified mode: aiming for {vars_per_agent} vars per agent.")

            num_agents = math.ceil(num_vars / vars_per_agent)

            base_size = num_vars // num_agents
            remainder = num_vars % num_agents

            groups = {}
            current_var_idx = 0
            for i in range(num_agents):
                agent_name = f"agent_{i}"
                group_size = base_size + 1 if i < remainder else base_size
                groups[agent_name] = list(range(current_var_idx, current_var_idx + group_size))
                current_var_idx += group_size

            return groups
        else:
            print("Auto-distribution mode: Environment is determining the optimal grouping.")
            ideal_min_size = 4
            ideal_max_size = 4
            factors = self._find_factors(num_vars)

            # 我们希望能够在一个范围内，让每个agent负责尽可能多的vars
            candidate_group_sizes = [f for f in factors if ideal_min_size <= f <= ideal_max_size]
            if candidate_group_sizes:
                best_group_size = max(candidate_group_sizes)
                num_agents = num_vars // best_group_size
                print(f"Found ideal grouping: {num_agents} agents, each with {best_group_size} vars.")
            else:
                num_agents = max(2, int(math.sqrt(num_vars)))

            base_size = num_vars // num_agents
            remainder = num_vars % num_agents

            groups = {}
            current_var_idx = 0
            for i in range(num_agents):
                agent_name = f"agent_{i}"
                group_size = base_size + 1 if i < remainder else base_size
                groups[agent_name] = list(range(current_var_idx, current_var_idx + group_size))
                current_var_idx += group_size
            return groups

    def _calculate_obs_dim(self) -> int:
        dim = self.num_vars + self.num_clauses + self.num_vars

        return dim

    def get_obs(self, state: SATState) -> Dict[str, chex.Array]:
        # 创建一个空字典，用于存储最终的观测结果
        observations = {}

        # 遍历环境中的每一个智能体
        for agent in self.agents:
            agent_idx = self.agent_to_idx[agent]
            # --- 第一部分：获取智能体自己负责变量的状态 ---
            # 1. 获取该智能体负责的变量索引，-1表示无关
            my_variable_indices = jnp.array(self.agent_groups[agent], dtype=jnp.int32)
            is_my_variable_mask = jnp.full((self.num_vars,), False).at[my_variable_indices].set(True)
            my_variables_status = jnp.where(
                is_my_variable_mask,
                state.variable_assignments,
                -1
            )
            # shape: (num_vars_for_this_agent,)

            # --- 第二部分：获取与该智能体相关的子句的状态 ---
            # 1. shape: (num_clauses,)  bool list
            related_clauses_mask = state.agent_clause_masks[agent_idx]

            # 2. 使用布尔掩码，从全局状态中筛选出相关的子句满足状态
            all_clauses_status = state.clauses_satisfied_status
            related_and_satisfied_status = jnp.where(
                related_clauses_mask == 1,
                jnp.where(all_clauses_status == 1, 1, 0),
                # 不相关
                -1
            ) # shape: (num_clauses)

            # --- 第三部分：获取邻居变量的状态 ---
            # 1. 从预计算的 map 中获取该智能体的邻居掩码
            neighbor_mask = state.agent_neighbor_masks[agent_idx]
            # shape: (num_vars,)
            # 示例: 如果总共有 50 个变量, shape -> (50,)
            # 2. 0 代表 false，1代表true， -1代表无关
            neighbor_variables_status = jnp.where(
                neighbor_mask != -1,
                neighbor_mask * state.variable_assignments,
                -1
            )
            # shape: (num_vars,)
            # 示例: shape -> (50,)

            observation_vector = jnp.concatenate([
                my_variables_status,  # shape: (self.num_vars,)
                related_and_satisfied_status.astype(jnp.int32),  # shape: (num_clauses,)
                neighbor_variables_status  # shape: (num_vars,)
            ])
            # shape: (num_vars_for_this_agent + num_clauses + num_vars,)
            observations[agent] = observation_vector

        return observations


    @property
    def name(self) -> str:
        "Environment name"
        return "SATEnv"

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> Space:
        """为一个指定的智能体返回其观测空间。"""
        return self.observation_spaces[agent]

