import os, jax, jaxlib
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())
print("jaxlib:", jaxlib.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))