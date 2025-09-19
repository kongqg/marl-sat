import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
import optax
import distrax
from typing import NamedTuple, Dict, Any
from functools import partial

from omegaconf import DictConfig

from src.models.ac_gnn import ACGNN
from src.utils.graph_constructor import GraphData


class Transition(NamedTuple):
    done: jnp.ndarray  # <--- 修正 1: ndarray 拼写正确
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: GraphData  # 我们的观测是图结构
    info: Dict


def create_train_state(model: ACGNN,
                       key: jax.random.PRNGKey,
                       config: DictConfig,
                       dummy_input: GraphData) -> TrainState:
    """为 RL 任务创建训练状态 (TrainState)"""
    params = model.init(key, dummy_input)['params']

    learning_rate = config.TRAIN_PARAMS.LR

    if config.TRAIN_PARAMS.ANNEAL_LR:
        def linear_schedule(count):
            num_cycles = config.TRAIN_PARAMS.NUM_CYCLES  # 或者 config.NUM_CYCLES
            update_epochs = config.PPO_PARAMS.UPDATE_EPOCHS  #
            num_minibatches = config.PPO_PARAMS.NUM_MINIBATCHES  #

            # 总优化步数 = 总循环数 * 每次循环的更新轮数 * 每轮的小批量数
            total_optimization_steps = num_cycles * update_epochs * num_minibatches

            frac = 1.0 - (count / total_optimization_steps)

            # 确保学习率不会低于 0，增加鲁棒性
            frac = jnp.maximum(0.0, frac)

            return learning_rate * frac

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=linear_schedule, eps=1e-5)
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate, eps=1e-5)
        )

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_update(config: Dict[str, Any]):
    """
    创建一个 JIT 编译的 PPO 更新函数。
    这个函数是整个学习算法的核心。
    """

    # 我们将 network 设为静态参数，因为它在 JIT 编译时是固定的
    @partial(jax.jit, static_argnames=("network",))
    def ppo_update(
            network: ACGNN,
            train_state: TrainState,
            traj_batch: Transition,
            last_val: jnp.ndarray,  # 最后一个状态的价值
            key: jax.random.PRNGKey,
    ):
        # ---- 1. 计算优势函数 GAE ----
        #
        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                '''
                 这里用n_step td error的方式 计算优势， 比A2C 单步的td error 更加的平滑
                 第一个gamma 代表 对未来奖励的重要性
                 第二个gamma 代表 对未来td_error的重要性
                 lambda 表示要融入多少 td_error， 如果lambda = 1 那么表示 mc的return td error ，如果 lambda = 0表示 单步td error
                '''
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward

                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,  #
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        batch_size = traj_batch.reward.shape[0] * traj_batch.reward.shape[1]  # 時間步 * 環境數

        def flatten(x):
            return x.reshape((batch_size,) + x.shape[2:])

        traj_batch = jax.tree_util.tree_map(flatten, traj_batch)
        advantages = flatten(advantages)
        targets = flatten(targets)

        # ---- 2. PPO 损失函数定义 ----
        def _ppo_loss(params, traj_batch, advantages, targets):
            logits, value = network.apply({'params': params}, traj_batch.obs)
            value_loss = jnp.mean((value - targets) ** 2)

            # value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            #     -config["CLIP_EPS"], config["CLIP_EPS"]
            # )
            # # 2. 计算裁剪前和裁剪后的损失
            # value_losses = (value - targets) ** 2
            # value_losses_clipped = (value_pred_clipped - targets) ** 2
            # # 3. 取两者中的最大值
            # value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # logit -> softmax -> sample -> log_prob,   pi is an object
            pi = distrax.Categorical(logits=logits)

            # new policy to extract the new prob  shape (num_vars, 1)
            log_prob = pi.log_prob(traj_batch.action)

            # ratio = pai(theta) / pai_old(theta)
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            entropy = pi.entropy().mean()
            # advantage normalization

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            loss_actor1 = ratio * advantages


            # 计算L(theta)
            loss_actor2 = jnp.clip(
                ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
            ) * advantages
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
            entropy = pi.entropy().mean()
            total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
            )
            return total_loss, (value_loss, loss_actor, entropy)

        # ---- 3. 多轮次、小批量更新 ----
        def _update_epoch(update_state, _):
            train_state, traj_batch, advantages, targets, key = update_state
            key, subkey = jax.random.split(key)
            batch_size = traj_batch.reward.shape[0]
            permutation = jax.random.permutation(subkey, batch_size)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: x[permutation], (traj_batch, advantages, targets)
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            def _update_minibatch(train_state, batch_info):
                grad_fn = jax.value_and_grad(_ppo_loss, has_aux=True)
                traj_batch, advantages, targets = batch_info
                (loss, (vloss, aloss, ent)), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )

                # if not isinstance(grads, FrozenDict):
                #     grads = freeze(grads)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (loss, vloss, aloss, ent)

            train_state, metrics = jax.lax.scan(
                # fn, init_state, nums
                _update_minibatch, train_state, minibatches
            )
            return (train_state, traj_batch, advantages, targets, key), metrics

        update_state = (train_state, traj_batch, advantages, targets, key)
        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        return update_state[0], metrics

    return ppo_update