import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from typing import Tuple, Dict, Any

from src.models.base_gnn import SATGNN
from src.utils.graph_constructor import GraphData


class TrainState(train_state.TrainState):
    pass

def create_train_state(model: SATGNN, key: jax.random.PRNGKey,
                       learning_rate: float, dummy_input: GraphData) -> TrainState:
    params = model.init(key, dummy_input)['params']

    tx = optax.adam(learning_rate)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state: TrainState, batch_of_graphs: GraphData, batch_of_labels: jnp.ndarray):

    vmapped_apply_fn = jax.vmap(state.apply_fn, in_axes=({'params':None}, 0))
    def loss_fn(params):
        logits = vmapped_apply_fn({'params': params}, batch_of_graphs) # shape: (num, 2)

        vmapped_loss = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)
        loss = jnp.mean(vmapped_loss(logits, batch_of_labels))

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch_of_labels)
    metrics = {"loss":loss,
               "accuracy": accuracy}
    return new_state, metrics, logits


@jax.jit
def eval_step(state: TrainState, batch_of_graphs: GraphData,
              batch_of_labels: jnp.ndarray):  # 建议3: 添加返回类型

    vmapped_apply_fn = jax.vmap(state.apply_fn, in_axes=({'params': None}, 0))
    logits = vmapped_apply_fn({'params': state.params}, batch_of_graphs)

    vmapped_loss = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)
    loss = jnp.mean(vmapped_loss(logits, batch_of_labels))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch_of_labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy
    }
    return metrics, logits