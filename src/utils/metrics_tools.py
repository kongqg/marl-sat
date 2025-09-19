import jax.numpy as jnp

def flatten_metrics(metrics):
    """
    metrics: shape (E, M, 4)  约定顺序: total_loss, value_loss, policy_loss, entropy
    返回标量 tuple
    """
    m = jnp.asarray(metrics)  # [E, M, 4]
    m = m.mean(axis=(0, 1))   # [4]
    total, vloss, aloss, ent = [float(x) for x in m]
    return total, vloss, aloss, ent

def mean_std(x: jnp.ndarray):
    """返回 (mean, std)，都是 float"""
    x = jnp.asarray(x)
    return float(x.mean()), float(x.std() + 1e-8)