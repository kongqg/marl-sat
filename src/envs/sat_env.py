import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax.struct import dataclass
from gymnax.environments import environment, spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, MultiDiscrete, Discrete
from gymnax.environments.environment import EnvParams, TEnvParams
from typing import Tuple, Optional, Dict

from src.utils.graph_constructor import GraphData, create_graph_data, StaticGraphData, create_static_graph, GNNInput
from src.utils.data_parser import parse_cnf
from src.utils.check_sat import check_satisfiability

@dataclass
class State:
    graph : StaticGraphData
    assignment: chex.Array
    step: int # 回合步
    key: chex.PRNGKey
    prev_unsat_ratio: jnp.float32  # 新增：上一时刻的 u(s)

class SatEnv(MultiAgentEnv):
    def __init__(self, num_vars , num_clauses, max_clause_len, c_bonus=1.0, alpha=1.0, max_steps=128):
        self.num_agents = 1
        self.agents = ["agent_0"]

        self.c_bonus = c_bonus
        self.alpha = alpha
        self.max_steps = max_steps
        self.max_clause_len = max_clause_len
        self.num_vars = num_vars
        self.num_clauses = num_clauses

        self.observation_spaces = {agent: Box(-jnp.inf, jnp.inf, (1,)) for agent in self.agents}
        self.action_spaces = {
            agent: Discrete(num_vars) for agent in self.agents
        }
    @property
    def name(self) -> str:
        return "SATEnv"

    @property
    def agent_classes(self) -> dict:
        return {"agents": self.agents}

    def reset(self, key: chex.PRNGKey, cnf_problem: Dict) -> Tuple[Dict[str, chex.Array], State]:
        clauses = jnp.array(cnf_problem['clauses'])


        key, subkey = jax.random.split(key)

        # 初始化时，随机给变量赋值
        initial_assignment = jax.random.randint(subkey, (self.num_vars,), 0, 2)

        static_graph_obj  = create_static_graph(self.num_vars, self.num_clauses, clauses)
        init_unsat = self.unsat_ratio_from_assignment(static_graph_obj, initial_assignment)

        state = State(
            graph=static_graph_obj,
            assignment=initial_assignment,
            step=0,
            key=key,
            prev_unsat_ratio=init_unsat,  # 保存 u(s0)
        )

        return self.get_obs(state), state

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool]]:

        # agent 选择的动作
        variable_to_flip = actions[self.agents[0]]
        current_assignment = state.assignment

        # 改变这一个变量的值
        new_assignment = current_assignment.at[variable_to_flip].set(1- current_assignment[variable_to_flip])

        # 计算新状态的不满足比例 u(s_{t+1})
        new_unsat = self.unsat_ratio_from_assignment(state.graph, new_assignment)


        # delta 奖励：u(s_t) - u(s_{t+1})
        delta = state.prev_unsat_ratio -  new_unsat
        delta_reward = delta * 10.0


        # delta = -0.01
        # 是否已经 SAT
        is_sat = (new_unsat == 0.0)

        # 终局 bonus（解出时叠加）
        reward_val = delta_reward + jnp.where(is_sat, self.c_bonus, 0.0).astype(jnp.float32)

        step_penalty = -0.005
        reward_val = reward_val + step_penalty

        # reward_val = jnp.clip(reward_val, -1.0, 1.0)



        rewards = {agent: reward_val for agent in self.agents}

        done = jnp.logical_or(is_sat, state.step >= self.max_steps)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        new_state = state.replace(
            assignment=new_assignment,
            step=state.step + 1,
            key=key,
            prev_unsat_ratio=new_unsat  #
        )

        return self.get_obs(new_state) ,new_state, rewards, dones, {}

    def get_unsat_clause_mask(self, graph, assignment):
        sat_pos = (graph.A_pos.T @ assignment) > 0
        sat_neg = (graph.A_neg.T @ (1 - assignment)) > 0
        clause_sat = sat_pos | sat_neg
        # 返回一个布尔数组，True 代表不满足
        return ~clause_sat

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        static_info = state.graph
        current_assignment = state.assignment

        # --- 变量特征 ---
        # 1. 计算正负度数作为前两维
        pos_degrees = jnp.sum(static_info.A_pos, axis=1, keepdims=True)
        neg_degrees = jnp.sum(static_info.A_neg, axis=1, keepdims=True)
        normalized_pos_degrees = pos_degrees / static_info.num_clauses
        normalized_neg_degrees = neg_degrees / static_info.num_clauses

        zeros_for_vars = jnp.zeros_like(normalized_pos_degrees)

        static_var_features = jnp.concatenate(
            [normalized_pos_degrees, normalized_neg_degrees, zeros_for_vars], axis=-1
        )

        # 1. 获取动态的“不满足”遮罩
        unsat_clause_mask = self.get_unsat_clause_mask(static_info, current_assignment)

        # 2. 构造前两维：[is_satisfied, is_unsatisfied]
        unsat_feature = unsat_clause_mask.astype(jnp.float32)[:, None]  # Shape: (n_clauses, 1)
        sat_feature = 1.0 - unsat_feature  # Shape: (n_clauses, 1)

        # 3. 构造第三维：静态的类型标识
        type_feature = jnp.ones_like(sat_feature)  # Shape: (n_clauses, 1)

        # 4. 沿最后一个维度拼接成最终的特征矩阵
        dynamic_clause_features = jnp.concatenate([sat_feature, unsat_feature, type_feature], axis=-1)
        # 最终 shape: (n_clauses, 3)

        gnn_input_data = GNNInput(
            static_var_features=static_var_features,  # Shape: (n_vars, 3)
            assignment=current_assignment,  # Shape: (n_vars,)
            clause_features=dynamic_clause_features,  # Shape: (n_clauses, 3)
            A_pos=static_info.A_pos,
            A_neg=static_info.A_neg
        )

        obs = {agent: gnn_input_data  for agent in self.agents}
        return obs

    def unsat_ratio_from_assignment(self,graph, assignment):
        # graph.A_pos/A_neg: (n_vars, n_clauses) - jnp.ndarray
        sat_pos = (graph.A_pos.T @ assignment) > 0
        sat_neg = (graph.A_neg.T @ (1 - assignment)) > 0
        clause_sat = sat_pos | sat_neg
        m = clause_sat.shape[0]
        # 返回 float32，保证和 JAX 算子 dtype 一致
        return (m - clause_sat.sum()).astype(jnp.float32) / m




