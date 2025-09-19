from functools import partial

import chex
import distrax
import jax
import jax.numpy as jnp
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import State
from flax import struct

from src.envs.multi_agent_sat_env import SATState, SATEnv
from src.utils.graph_constructor import GNNInput, StaticGraphData, create_static_graph
from flax import linen as nn
from typing import Tuple, Optional
from flax.training.train_state import TrainState
import optax


class GNNEncoder(nn.Module):
    """
    【最终版】Literal-Level GNN Encoder，支持边掩码以实现局部消息传递。
    """
    hidden_dim: int = 128
    num_message_passing_step: int = 8

    @nn.compact
    def __call__(self, gnn_input: GNNInput,
                 edge_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        A_pos, A_neg = gnn_input.A_pos, gnn_input.A_neg

        # 如果提供了边掩码，则应用它来阻断非局部信息流
        if edge_mask is not None:
            A_pos_eff = A_pos * edge_mask
            A_neg_eff = A_neg * edge_mask
        else:  # 否则，使用完整的邻接矩阵（用于Critic）
            A_pos_eff = A_pos
            A_neg_eff = A_neg

        # --- 提取输入数据 ---
        static_var_features = gnn_input.static_var_features
        assignment = gnn_input.assignment
        dynamic_clause_features = gnn_input.clause_features

        # --- 定义所有神经网络层 (MLPs) ---
        literal_pos_embed = nn.Dense(features=self.hidden_dim, name="literal_pos_embed")
        literal_neg_embed = nn.Dense(features=self.hidden_dim, name="literal_neg_embed")
        clause_embed = nn.Dense(features=self.hidden_dim, name="clause_embed")
        phi_c_pos = nn.Dense(features=self.hidden_dim, name="phi_c_pos")
        phi_c_neg = nn.Dense(features=self.hidden_dim, name="phi_c_neg")
        phi_v_pos = nn.Dense(features=self.hidden_dim, name="phi_v_pos")
        phi_v_neg = nn.Dense(features=self.hidden_dim, name="phi_v_neg")
        update_c = nn.GRUCell(features=self.hidden_dim, name="update_c")
        update_v_pos = nn.GRUCell(features=self.hidden_dim, name="update_v_pos")
        update_v_neg = nn.GRUCell(features=self.hidden_dim, name="update_v_neg")

        # --- 初始化节点嵌入 ---
        H_v_pos = literal_pos_embed(static_var_features)
        H_v_neg = literal_neg_embed(static_var_features)
        H_c = clause_embed(dynamic_clause_features)

        # --- 迭代消息传递 ---
        for _ in range(self.num_message_passing_step):
            H_v_pos_prev, H_v_neg_prev, H_c_prev = H_v_pos, H_v_neg, H_c
            msg_from_pos_literals = phi_c_pos(H_v_pos_prev)
            msg_from_neg_literals = phi_c_neg(H_v_neg_prev)
            m_c_pos = A_pos_eff.T @ msg_from_pos_literals
            m_c_neg = A_neg_eff.T @ msg_from_neg_literals
            update_c_input = jnp.concatenate([m_c_pos, m_c_neg], axis=-1)
            _, H_c = update_c(H_c_prev, update_c_input)
            H_c = nn.LayerNorm()(H_c)
            msg_to_pos_literals = phi_v_pos(H_c)
            msg_to_neg_literals = phi_v_neg(H_c)
            n_v_pos = A_pos_eff @ msg_to_pos_literals
            n_v_neg = A_neg_eff @ msg_to_neg_literals
            update_v_pos_input = jnp.concatenate([n_v_pos, assignment[..., None], static_var_features], axis=-1)
            _, H_v_pos = update_v_pos(H_v_pos_prev, update_v_pos_input)
            H_v_pos = nn.LayerNorm()(H_v_pos)
            update_v_neg_input = jnp.concatenate([n_v_neg, assignment[..., None], static_var_features], axis=-1)
            _, H_v_neg = update_v_neg(H_v_neg_prev, update_v_neg_input)
            H_v_neg = nn.LayerNorm()(H_v_neg)

        return H_v_pos, H_v_neg, H_c


@chex.dataclass(frozen=True)
class GNNWrapperState:
    "提供 state 和 GNN的静态图数据"

    env_state: SATState
    static_graph: StaticGraphData


class SATDataWrapper(JaxMARLWrapper):
    """
        将SATState 转换成 GNNInput 并添加到observation中

    """

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, problem_clauses: jnp.ndarray, key: chex.PRNGKey):
        # 1. 获取基础环境的本地观测和状态
        local_obs, pure_env_state = self._env.reset(problem_clauses, key)

        # 2. 创建 GNN 静态图和 GNN 输入 (全局状态)
        static_graph_part = create_static_graph(
            num_vars=self._env.num_vars,
            num_clauses=self._env.num_clauses,
            clauses=problem_clauses
        )
        global_state = self._state_to_gnn_input(pure_env_state, static_graph_part)

        # 3. 创建包装器状态
        wrapper_state = GNNWrapperState(
            env_state=pure_env_state,
            static_graph=static_graph_part
        )
        # 4. ---  返回分离的 local_obs 和 global_state ---
        return (local_obs, global_state), wrapper_state

    @partial(jax.jit, static_argnums=0)
    def step(self,
             key: jax.random.PRNGKey,
             state: GNNWrapperState,
             actions: dict):
        pure_env_state = state.env_state
        static_graph_part = state.static_graph

        actions_array = jnp.stack([actions[agent] for agent in self._env.agents])

        # 1. 基础环境步进，获取本地观测
        local_obs, next_pure_env_state, reward, done, info = self._env.step_env(key,
                                                                                pure_env_state,
                                                                                actions_array)
        # 2. 创建 GNN 输入 (全局状态)
        global_state = self._state_to_gnn_input(next_pure_env_state, static_graph_part)

        # 3. 创建包装器状态
        next_wrapper_state = GNNWrapperState(
            env_state=next_pure_env_state,
            static_graph=static_graph_part
        )
        # 4. --- 返回分离的 local_obs 和 global_state ---
        return (local_obs, global_state), next_wrapper_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def _state_to_gnn_input(self, state: SATState, static_graph: StaticGraphData):
        A_pos = static_graph.A_pos
        A_neg = static_graph.A_neg
        num_clauses = static_graph.num_clauses
        # --- 变量特征 ---
        # 1. 计算正负度数作为前两维
        pos_degrees = jnp.sum(A_pos, axis=1, keepdims=True)
        neg_degrees = jnp.sum(A_neg, axis=1, keepdims=True)
        normalized_pos_degrees = pos_degrees / num_clauses
        normalized_neg_degrees = neg_degrees / num_clauses

        zeros_for_vars = jnp.zeros_like(normalized_pos_degrees)

        static_var_features = jnp.concatenate(
            [normalized_pos_degrees, normalized_neg_degrees, zeros_for_vars], axis=-1
        )
        assignment = state.variable_assignments
        dynamic_clause_features = self._calculate_dynamic_clause_features(state)

        return GNNInput(
            static_var_features=static_var_features,
            assignment=assignment,
            clause_features=dynamic_clause_features,
            A_pos=A_pos,
            A_neg=A_neg
        )

    @partial(jax.jit, static_argnums=0)
    def _calculate_dynamic_clause_features(self, state: SATState):
        # 统计满足的子句中有几个值是true
        variable_indices = jnp.abs(state.clauses) - 1
        assignments = state.variable_assignments[variable_indices]
        literal_truth_values = ((state.clauses > 0) & (assignments == 1)) | \
                               ((state.clauses < 0) & (assignments == 0))
        num_satisfying_literals = jnp.sum(literal_truth_values, axis=1)
        # norm
        norm_num_satisfying_literals = num_satisfying_literals / 3.0

        is_sat = state.clauses_satisfied_status.astype(jnp.int32)

        type_indentify = jnp.ones_like(is_sat)

        dynamic_features = jnp.stack(
            [is_sat, norm_num_satisfying_literals, type_indentify],
            axis=-1
        )
        return dynamic_features


class GNN_ActorCritic(nn.Module):
    gnn_hidden_dim: int
    gnn_num_message_passing_steps: int
    num_agents: int
    max_vars_per_agent: int
    action_mode: int
    agent_id_embed_dim: int = 16

    def setup(self):
        self.encoder = GNNEncoder(
            hidden_dim=self.gnn_hidden_dim,
            num_message_passing_step=self.gnn_num_message_passing_steps,
            name="encoder"
        )
        # --- Critic 部分保持不变 ---
        self.critic_dense_0 = nn.Dense(128, name="critic_dense_0")
        self.critic_dense_1 = nn.Dense(64, name="critic_dense_1")
        self.critic_output = nn.Dense(1, name="critic_output")

        self.agent_id_embedding = nn.Embed(
            num_embeddings=self.num_agents,
            features=self.agent_id_embed_dim,
            name="agent_id_embedding"
        )

        # <--- 【核心修正】为 Actor 头设计两个并行的逻辑分支 --->
        if self.action_mode == 0:
            # 分支 1: 处理每个变量的翻转决策 (Per-Variable Flip Branch)
            # 这个 MLP 会在每个变量的专属特征上独立运行
            var_input_dim = (self.gnn_hidden_dim * 2) + ((self.gnn_hidden_dim * 4) + self.agent_id_embed_dim)
            self.actor_flip_head_dense = nn.Dense(128, name="actor_flip_head_dense")
            self.actor_flip_head_output = nn.Dense(1, name="actor_flip_head_output")

            # 分支 2: 处理 agent 级别的 no-op 决策 (Agent-Level No-Op Branch)
            # 这个 MLP 只接收 agent 的整体上下文信息
            local_context_dim = (self.gnn_hidden_dim * 4) + self.agent_id_embed_dim
            self.actor_noop_head_dense = nn.Dense(64, name="actor_noop_head_dense")
            self.actor_noop_head_output = nn.Dense(1, name="actor_noop_head_output")

        else:  # multi_flip 模式 (保持不变)
            self.actor_dense_0 = nn.Dense(128, name="actor_dense_0")
            self.actor_dense_1 = nn.Dense(64, name="actor_dense_1")
            self.actor_output = nn.Dense(2, name="actor_output")

    # ... (_get_local_edge_masks 函數保持不變) ...
    def _get_local_edge_masks(self, gnn_input: GNNInput, agent_vars: jnp.ndarray):
        V, C = gnn_input.A_pos.shape
        valid_vars_mask = (agent_vars != -1)
        safe_agent_vars = jnp.maximum(agent_vars, 0)
        var_mask_one_hot = jax.nn.one_hot(safe_agent_vars, V, axis=-1)
        var_mask_one_hot_valid = var_mask_one_hot * valid_vars_mask[..., None]
        var_mask = var_mask_one_hot_valid.sum(axis=1)
        A = ((gnn_input.A_pos + gnn_input.A_neg) > 0).astype(jnp.float32)
        clause_mask = (var_mask @ A > 0).astype(jnp.float32)
        related_var_mask = (clause_mask @ A.T > 0).astype(jnp.float32)
        visible_var_mask = jnp.logical_or(var_mask > 0, related_var_mask > 0).astype(jnp.float32)
        edge_mask = visible_var_mask[:, :, None] * clause_mask[:, None, :]
        return edge_mask

    def apply_actor(self,
                    gnn_input: GNNInput,
                    agent_vars: jnp.ndarray,
                    action_mask: jnp.ndarray):
        A = self.num_agents

        # --- (特徵提取部分，包括 GNN 編碼和局部池化，保持不變) ---
        def masked_mean_pool(X, M):
            w = M[..., None].astype(X.dtype)
            num = (X * w).sum(axis=1)
            den = jnp.maximum(M.sum(axis=1, keepdims=True), 1.0)
            return num / den

        edge_mask = self._get_local_edge_masks(gnn_input, agent_vars)

        def _encode_with_mask(mask_one_agent):
            H_v_pos, H_v_neg, H_c = self.encoder(gnn_input, edge_mask=mask_one_agent)
            return H_v_pos, H_v_neg, H_c

        H_v_pos_loc, H_v_neg_loc, H_c_loc = jax.vmap(_encode_with_mask, in_axes=0)(edge_mask)
        H_v_loc = jnp.concatenate([H_v_pos_loc, H_v_neg_loc], axis=-1)
        safe_idx = jnp.maximum(agent_vars, 0)[..., None]
        my_var_embeddings = jnp.take_along_axis(H_v_loc, safe_idx, axis=1)

        def _pool_my_vars(emb, m):
            w = m[..., None].astype(emb.dtype)
            num = (emb * w).sum(axis=0)
            den = jnp.maximum(m.sum(), 1.0)
            return num / den

        var_mask = (agent_vars != -1)
        my_vars_summary = jax.vmap(_pool_my_vars)(my_var_embeddings, var_mask)
        visible_var_mask = (edge_mask.sum(axis=2) > 0).astype(jnp.float32)
        V = gnn_input.A_pos.shape[0]
        rows = jnp.repeat(jnp.arange(A)[:, None], self.max_vars_per_agent, axis=1)
        safe_idx_1d = jnp.maximum(agent_vars, 0)
        own_var_mask = jnp.zeros((A, V), dtype=jnp.float32).at[rows, safe_idx_1d].add(var_mask.astype(jnp.float32))
        neighbor_var_mask = jnp.clip(visible_var_mask - own_var_mask, 0.0, 1.0)
        clause_mask = (edge_mask.sum(axis=1) > 0).astype(jnp.float32)
        neighbor_vars_summary = masked_mean_pool(H_v_loc, neighbor_var_mask)
        clauses_summary = masked_mean_pool(H_c_loc, clause_mask)
        agent_ids = jnp.arange(A)
        agent_id_embeds = self.agent_id_embedding(agent_ids)
        local_context = jnp.concatenate(
            [my_vars_summary, neighbor_vars_summary, clauses_summary, agent_id_embeds], axis=-1)

        # <--- 【核心修正】並行分支計算 Logits --->
        if self.action_mode == 0:
            # --- 分支 1: 計算每個變數的翻轉 Logit ---
            # 1. 將 agent 的整體上下文廣播到每個變數上
            local_ctx_exp = jnp.repeat(local_context[:, None, :], self.max_vars_per_agent, axis=1)
            # 2. 拼接每個變數的「專屬嵌入」和「共享上下文」
            var_inputs = jnp.concatenate([my_var_embeddings, local_ctx_exp], axis=-1)
            # 3. 將拼接後的特徵送入翻轉決策頭，得到每個變數的 logit
            flip_h = nn.relu(self.actor_flip_head_dense(var_inputs))
            flip_logits = jnp.squeeze(self.actor_flip_head_output(flip_h), axis=-1)  # Shape: (A, max_vars)

            # --- 分支 2: 計算 agent 的 No-Op Logit ---
            # 1. 將 agent 的整體上下文送入 no-op 決策頭
            noop_h = nn.relu(self.actor_noop_head_dense(local_context))
            no_op_logits = self.actor_noop_head_output(noop_h)  # Shape: (A, 1)

            # --- 合併與輸出 ---
            # 1. 將翻轉 logits 和 no-op logit 拼接成完整的動作 logits
            logits = jnp.concatenate([flip_logits, no_op_logits], axis=-1)
            # 2. 應用動作掩碼
            full_mask = jnp.concatenate([action_mask, jnp.ones((A, 1), dtype=bool)], axis=-1)
            logits = jnp.where(full_mask, logits, -jnp.inf)
            pi = distrax.Categorical(logits=logits)

        else:  # multi_flip 模式 (保持不變)
            # ... (此處邏輯與原版完全相同) ...
            agent_id_exp = jnp.repeat(agent_id_embeds[:, None, :], self.max_vars_per_agent, axis=1)
            actor_input = jnp.concatenate([my_var_embeddings, agent_id_exp], axis=-1)
            h0 = nn.relu(self.actor_dense_0(actor_input))
            h1 = nn.relu(self.actor_dense_1(h0))
            var_logits = self.actor_output(h1)
            logits = jnp.where(action_mask[..., None], var_logits, -jnp.inf)
            pi = distrax.Categorical(logits=logits)

        return pi

    # ... (apply_critic 和 __call__ 函數保持不變) ...
    def apply_critic(self, gnn_input: GNNInput):
        # ... (此處邏輯與原版完全相同) ...
        H_v_pos, H_v_neg, H_c = self.encoder(gnn_input, edge_mask=None)
        H_v_combined = jnp.concatenate([H_v_pos, H_v_neg], axis=-1)
        var_mean, var_max = jnp.mean(H_v_combined, axis=-2), jnp.max(H_v_combined, axis=-2)
        clause_mean, clause_max = jnp.mean(H_c, axis=-2), jnp.max(H_c, axis=-2)
        global_graph_embedding = jnp.concatenate([var_mean, var_max, clause_mean, clause_max], axis=-1)
        critic_h = nn.relu(self.critic_dense_0(global_graph_embedding))
        critic_h = nn.relu(self.critic_dense_1(critic_h))
        value = jnp.squeeze(self.critic_output(critic_h), axis=-1)
        return value

    def __call__(self, gnn_input: GNNInput, agent_vars: jnp.ndarray, action_mask: jnp.ndarray):
        pi = self.apply_actor(gnn_input, agent_vars, action_mask)
        value = self.apply_critic(gnn_input)
        return pi, value


@chex.dataclass(frozen=True)
class Transition:
    global_done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    local_obs: dict  # Actor 使用的本地观测, shape: (num_envs, num_agents, obs_dim)
    global_state: GNNInput  # Critic 使用的全局状态, 也就是 gnn_input
    info: dict
    agent_clause_masks: chex.Array
    agent_neighbor_masks: chex.Array


@chex.dataclass(frozen=True)
class RunnerState:
    train_state: TrainState
    env_state: GNNWrapperState
    last_local_obs: dict
    last_global_state: GNNInput
    rng: chex.PRNGKey


def make_train_cycle(config, env: SATDataWrapper, network: GNN_ActorCritic, vmapped_reset_fn):
    def _train_cycle(runner_state: RunnerState, problems: dict, update_idx: chex.Array):
        def _env_step(carry, _):
            train_state, env_state, last_local_obs, last_global_state, rng = carry

            agent_clause_masks = env_state.env_state.agent_clause_masks
            agent_neighbor_masks = env_state.env_state.agent_neighbor_masks

            actor_fn = partial(network.apply, method=GNN_ActorCritic.apply_actor)
            critic_fn = partial(network.apply, method=GNN_ActorCritic.apply_critic)
            pi = jax.vmap(actor_fn, in_axes=(None, 0, None, None), out_axes=0)(
                {'params': train_state.params}, last_global_state, env.agent_vars, env.action_mask
            )
            value = jax.vmap(critic_fn, in_axes=(None, 0), out_axes=0)(
                {'params': train_state.params}, last_global_state
            )
            rng, act_key = jax.random.split(rng)

            # (B, N_AGENTS, MAX_VARS)
            action = pi.sample(seed=act_key)
            # jax.debug.print("action: {x}", x=action)
            # (B, N_AGENTS, MAX_VARS)
            log_prob = pi.log_prob(action)

            # --- B. 环境步进 ---
            # 挑选每个env中 自己负责的 哪部分的变量
            # e.g., {'agent_0': shape(B, MAX_VARS), ...}
            if config["action_mode"] == 0:  # single_flip
                # 在 single_flip 模式下, action 的 shape 是 (NUM_ENVS, num_agents)
                # 我们需要为每个 agent 切片出 shape 为 (NUM_ENVS,) 的向量
                env_act = {agent: action[:, i] for i, agent in enumerate(env.agents)}
            else:  # multi_flip
                # 在 multi_flip 模式下, action 的 shape 是 (NUM_ENVS, num_agents, max_vars_per_agent)
                # 我们需要为每个 agent 切片出 shape 为 (NUM_ENVS, max_vars_per_agent) 的矩阵
                env_act = {agent: action[:, i, :] for i, agent in enumerate(env.agents)}
            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, config["NUM_ENVS"])
            (next_local_obs, next_global_state), next_env_state, reward, done, info = jax.vmap(env.step)(
                key=step_keys, state=env_state, actions=env_act
            )

            # --- C. 【关键】自动重置逻辑 ---
            done_mask = done["__all__"]  # (NUM_ENVS,)

            # 1. 准备新问题：无论是否done，都提前准备好
            rng, prob_key, reset_key = jax.random.split(rng, 3)

            # 获取问题的总数
            num_total_problems = jax.tree_util.tree_leaves(problems)[0].shape[0]
            new_problem_indices = jax.random.randint(prob_key, (config["NUM_ENVS"],), 0, num_total_problems)
            new_problems = jax.tree_util.tree_map(lambda x: x[new_problem_indices], problems)

            # 2. 对新问题进行重置
            reset_keys = jax.random.split(reset_key, config["NUM_ENVS"])
            (local_obs_after_reset, global_state_after_reset), state_after_reset = vmapped_reset_fn(
                new_problems['clauses'], reset_keys)

            """
                # 3. 使用 `where` 根据 done_mask 来选择性替换
                # jnp.where(condition, x, y)
                # jnp.tree_util.tree_map(fn, tree1, tree2)
                每次挑选问题都从整个数据集中取一个
            """

            def _reset_if_done(old, new, done):
                # 根據 old 的維度數，動態地重塑 done_mask
                # 例如，如果 old.ndim=2, mask_shape -> (B, 1)
                # 如果 old.ndim=3, mask_shape -> (B, 1, 1)
                mask_shape = done.shape + (1,) * (old.ndim - 1)
                reshaped_done = jnp.reshape(done, mask_shape)
                return jnp.where(reshaped_done, new, old)

            final_next_env_state = jax.tree_util.tree_map(
                lambda old, new: _reset_if_done(old, new, done_mask),
                next_env_state, state_after_reset
            )
            final_next_local_obs = jax.tree_util.tree_map(
                lambda old, new: _reset_if_done(old, new, done_mask),
                next_local_obs, local_obs_after_reset
            )
            final_next_global_state = jax.tree_util.tree_map(
                lambda old, new: _reset_if_done(old, new, done_mask),
                next_global_state, global_state_after_reset
            )

            # --- D. 存储 Transition ---
            transition = Transition(
                global_done=done["__all__"],
                action=action,
                value=value,
                reward=jnp.stack([reward[agent] for agent in env.agents], axis=-1),
                log_prob=log_prob,
                local_obs=last_local_obs,
                global_state=last_global_state,
                info=info,
                agent_clause_masks=agent_clause_masks,
                agent_neighbor_masks=agent_neighbor_masks
            )
            next_carry = (train_state, final_next_env_state, final_next_local_obs, final_next_global_state, rng)
            return next_carry, transition

        # 使用 scan 执行 rollout
        initial_carry = (
            runner_state.train_state,
            runner_state.env_state,
            runner_state.last_local_obs,
            runner_state.last_global_state,
            runner_state.rng
        )

        # jax.lax.scan(loop_function, initial_carry, xs, length(if xs is none))
        (final_train_state, final_env_state, final_local_obs, final_global_state, final_rng), traj_batch = jax.lax.scan(
            _env_step, initial_carry, None, config["NUM_STEPS"]
        )

        # --- 2. GAE 计算 ---
        critic_fn = partial(network.apply, method=GNN_ActorCritic.apply_critic)

        # Vmap over the new, simpler function
        last_val = jax.vmap(critic_fn, in_axes=(None, 0), out_axes=0)(
            {'params': final_train_state.params}, final_global_state
        )

        def _calculate_gae(traj_batch, last_val):
            #
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                #
                team_reward = reward[..., 0]
                delta = team_reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)
        # 一次性归一化
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        def calculate_current_ent_coef():
            anneal_ent = config.get("ANNEAL_ENT", False)
            if not anneal_ent:
                return jnp.array(config["ENT_COEF"])

            num_updates = config["NUM_UPDATES"]
            ent_coef_start = config["ENT_COEF"]
            ent_coef_end = config.get("ENT_COEF_END", 0.0)
            anneal_frac = config.get("ANNEAL_ENT_FRAC", 0.333)

            anneal_start_update = num_updates * (1.0 - anneal_frac)

            # 計算線性插值比例
            frac = (update_idx - anneal_start_update) / (num_updates - anneal_start_update)
            frac = jnp.maximum(0.0, jnp.minimum(1.0, frac))  # 裁剪到 [0, 1] 範圍

            # 計算當前係數
            current_coef = ent_coef_start - (ent_coef_start - ent_coef_end) * frac

            # 使用 jax.lax.cond 確保 JIT 兼容性
            return jax.lax.cond(
                update_idx >= anneal_start_update,
                lambda: current_coef,
                lambda: jnp.array(ent_coef_start)
            )

        current_ent_coef = calculate_current_ent_coef()

        # --- 3. PPO 更新 ---
        def _update_epoch(update_state, _):

            train_state, traj_batch, advantages, targets, rng, current_ent_coef = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            permutation = jax.random.permutation(_rng, batch_size)

            # jax.tree_util.tree_map(function, tree)
            # jnp.take(A, indices, axis)  打乱A
            # shuffled_batch.action 的 shape: (batch_size, num_agents, max_vars_per_agent)
            # shuffled_batch.value 的 shape: (batch_size, num_agents)
            # shuffled_batch.reward 的 shape: (batch_size, num_agents)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x.reshape((batch_size,) + x.shape[2:]), permutation, axis=0), traj_batch)

            shuffled_advantages = jnp.take(advantages.reshape((batch_size,)), permutation, axis=0)  # -> (B,)

            shuffled_targets = jnp.take(targets.reshape((batch_size,)), permutation, axis=0)  # -> (B,)

            num_minibatches = batch_size // config["MINIBATCH_SIZE"]

            # (MINIBATCH_SIZE, )
            #  minibatched_traj.action 的 shape: (num_minibatches, MINIBATCH_SIZE, num_agents, max_vars_per_agent)
            #  minibatched_traj.value 的 shape: (num_minibatches, MINIBATCH_SIZE, num_agents)
            #  minibatched_traj.reward 的 shape: (num_minibatches, MINIBATCH_SIZE, num_agents)
            minibatched_traj = jax.tree_util.tree_map(
                lambda x: x.reshape((num_minibatches, config["MINIBATCH_SIZE"]) + x.shape[1:]), shuffled_batch)

            minibatched_advantages = shuffled_advantages.reshape(num_minibatches, config["MINIBATCH_SIZE"])
            minibatched_targets = shuffled_targets.reshape(num_minibatches, config["MINIBATCH_SIZE"])

            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets, current_ent_coef):

                    actor_fn = partial(network.apply, method=GNN_ActorCritic.apply_actor)
                    critic_fn = partial(network.apply, method=GNN_ActorCritic.apply_critic)

                    pi = jax.vmap(actor_fn, in_axes=(None, 0, None, None), out_axes=0)(
                        {'params': params}, traj_batch.global_state,
                        env.agent_vars, env.action_mask
                    )
                    value = jax.vmap(critic_fn, in_axes=(None, 0), out_axes=0)(
                        {'params': params}, traj_batch.global_state
                    )
                    log_prob = pi.log_prob(traj_batch.action)
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    # gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    gae = gae[:, None]
                    if config["action_mode"] == 0:  # single_flip
                        # gae shape: (minibatch_size, num_agents)
                        # ratio shape: (minibatch_size, num_agents)
                        # 形状匹配，直接相乘
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    else:  # multi_flip
                        # 1. 计算当前策略下联合动作的对数概率 (在变量维度上求和)
                        # log_prob shape: (minibatch_size, num_agents, max_vars_per_agent)
                        log_prob_joint_new = log_prob.sum(axis=-1)  # shape: (minibatch_size, num_agents)

                        # 2. 计算旧策略下联合动作的对数概率
                        log_prob_joint_old = traj_batch.log_prob.sum(axis=-1)  # shape: (minibatch_size, num_agents)

                        # 3. 基于联合动作的对数概率计算 ratio
                        logratio_joint = log_prob_joint_new - log_prob_joint_old
                        ratio_joint = jnp.exp(logratio_joint)  # shape: (minibatch_size, num_agents)

                        # 4. gae 的 shape (minibatch_size, num_agents)
                        loss_actor1 = ratio_joint * gae
                        loss_actor2 = jnp.clip(ratio_joint, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae

                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    actor_loss = loss_actor - current_ent_coef * entropy
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["VF_CLIP"],
                                                                                            config["VF_CLIP"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    total_loss = actor_loss + config["VF_COEF"] * value_loss
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, aux_data), grads = grad_fn(train_state.params, traj_batch, advantages, targets,
                                                        current_ent_coef)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, aux_data

            final_train_state, loss_info = jax.lax.scan(_update_minibatch, train_state,
                                                        (minibatched_traj, minibatched_advantages, minibatched_targets))
            return (final_train_state, traj_batch, advantages, targets, rng, current_ent_coef), loss_info

        update_state = (final_train_state, traj_batch, advantages, targets, final_rng, current_ent_coef)
        (updated_train_state, _, _, _, updated_rng, _), metrics = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        # 计算episode return
        # traj_batch.reward 的 shape: (NUM_STEPS, NUM_ENVS, NUM_AGENTS)
        # 得到每个agent 的 每一步的全局奖励
        team_rewards = traj_batch.reward[:, :, 0]  # Shape: (NUM_STEPS, NUM_ENVS)
        # 在step的维度上求和
        total_rollout_return = jnp.sum(team_rewards, axis=0)  # Shape: (NUM_ENVS,)
        # 取所有环境的平均值
        mean_episodic_return = jnp.mean(total_rollout_return)

        finished_episodes_mask = traj_batch.global_done
        num_episodes_finished = jnp.sum(finished_episodes_mask)
        solved_at_finish_mask = traj_batch.info["solved"] & finished_episodes_mask
        num_episodes_solved = jnp.sum(solved_at_finish_mask)
        solve_rate = num_episodes_solved / jnp.maximum(num_episodes_finished, 1.0)

        # 计算未满足子句指标
        total_unsatisfied_at_finish = jnp.sum(
            traj_batch.info["num_unsatisfied"] * finished_episodes_mask
        )
        avg_unsatisfied_clauses = total_unsatisfied_at_finish / jnp.maximum(num_episodes_finished, 1.0)

        # 计算平均解决步数
        total_steps_for_solved = jnp.sum(
            traj_batch.info["episode_step"] * solved_at_finish_mask
        )
        avg_steps_to_solve = total_steps_for_solved / jnp.maximum(num_episodes_solved, 1.0)

        # Explained Variance (EV，解释变异度)
        var_targets = jnp.var(targets)
        # 1. 将 (NUM_STEPS, NUM_ENVS) 两个维度合并成一个大的批处理维度
        batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        flat_global_state = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]),
            traj_batch.global_state
        )

        # 2. 在这个合并后的批处理维度上使用 vmap
        value_pred_flat = jax.vmap(critic_fn, in_axes=(None, 0), out_axes=0)(
            {'params': updated_train_state.params}, flat_global_state
        )
        value_pred = value_pred_flat.reshape(config["NUM_STEPS"], config["NUM_ENVS"])
        var_pred_minus_targets = jnp.var(targets - value_pred)
        explained_variance = 1 - var_pred_minus_targets / jnp.maximum(var_targets, 1e-8)

        # final_losses = jax.tree_util.tree_map(lambda x: x[-1], metrics)
        value_loss_epochs, actor_loss_epochs, entropy_epochs = metrics[0], metrics[1], metrics[2]

        all_metrics = {
            "mean_episodic_return": mean_episodic_return,
            "solve_rate": solve_rate,
            "avg_unsatisfied_clauses": avg_unsatisfied_clauses,
            "avg_steps_to_solve": avg_steps_to_solve,
            "explained_variance": explained_variance,

            "epoch_value_losses": value_loss_epochs,
            "epoch_actor_losses": actor_loss_epochs,
            "epoch_entropies": entropy_epochs,
            "current_ent_coef": current_ent_coef,
        }

        # --- 4. 返回更新后的状态和指标 ---
        final_runner_state = RunnerState(
            train_state=updated_train_state,
            env_state=final_env_state,
            last_local_obs=final_local_obs,
            last_global_state=final_global_state,
            rng=updated_rng
        )
        return final_runner_state, all_metrics

    # 返回 JIT 编译后的单轮训练函数
    return jax.jit(_train_cycle)