import jax
import jax.numpy as jnp
from jax import random

# 打印 JAX 和设备的版本信息
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# 1. 创建一个随机的 JAX key
# JAX 中的随机数生成是可重复的，需要一个 key
key = random.PRNGKey(0)

# 2. 创建一个示例矩阵
# 我们来创建一个 3x2 的随机矩阵 A
A = random.normal(key, (3, 2))
print("原始矩阵 A:")
print(A)

# 3. 使用 jnp.linalg.svd 进行奇异值分解
# full_matrices=False 会让 U 和 V^T 的维度更紧凑，这在 m > n 或 n > m 时很有用
# U 的维度会是 (m, k), V^T 的维度会是 (k, n)，其中 k = min(m, n)
U, s, Vt = jnp.linalg.svd(A, full_matrices=False)

# 4. 打印分解结果
print("\n--- SVD 分解结果 ---")
print("左奇异向量 U:")
print(U)
print("\n奇异值 s (一个一维数组):")
print(s)
print("\n右奇异向量的转置 V^T:")
print(Vt)

# 5. 验证分解的正确性
# 将奇异值 s 转换为对角矩阵 Sigma
Sigma = jnp.diag(s)

# 通过 U @ Sigma @ V^T 重构矩阵
A_reconstructed = U @ Sigma @ Vt

print("\n--- 验证 ---")
print("重构后的矩阵 A_reconstructed:")
print(A_reconstructed)

# 检查重构矩阵与原始矩阵是否近似相等
is_close = jnp.allclose(A, A_reconstructed)
print(f"\n重构矩阵与原始矩阵是否近似相等? {is_close}")