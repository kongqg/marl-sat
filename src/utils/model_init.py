# -*- coding: utf-8 -*-
import os
import re
from typing import Optional

import flax
from flax.training import checkpoints as flax_checkpoints


# ---------- 小工具 ----------
def _shape_of(p):
    try:
        return tuple(p.shape)
    except Exception:
        return None

def _copy_leaf(dst_tree, src_tree, dst_path, src_path):
    """把 src_tree[src_path] 这个叶子拷到 dst_tree[dst_path]。
       若形状不匹配会报错；若 src_path 不存在返回 False。"""
    # 找 src 叶子
    cur = src_tree
    for k in src_path:
        if k not in cur:
            return False
        cur = cur[k]
    src_leaf = cur

    # 确保 dst 的父路径存在
    cur = dst_tree
    for k in dst_path[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    dst_key = dst_path[-1]

    # 做个形状 sanity check
    src_shape = _shape_of(src_leaf)
    dst_shape = _shape_of(cur.get(dst_key, src_leaf))
    if (src_shape is not None) and (dst_shape is not None) and (src_shape != dst_shape):
        raise ValueError(
            f"形状不匹配: dst {dst_path} 期待 {dst_shape}, 但 src {src_path} 是 {src_shape}"
        )

    cur[dst_key] = src_leaf
    return True

def _find_latest_step(bc_ckpt_path: str, prefix: str) -> Optional[int]:
    """从目录里找出形如 f'{prefix}{step}' 的**最大 step**。"""
    latest = None
    if not os.path.isdir(bc_ckpt_path):
        return None
    for name in os.listdir(bc_ckpt_path):
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        # 只接受纯数字的 step
        if re.fullmatch(r"\d+", tail):
            s = int(tail)
            if (latest is None) or (s > latest):
                latest = s
    return latest


# ---------- 主函数 ----------
def load_bc_and_inject(train_state,
                       bc_ckpt_path: str,
                       action_mode: int,
                       step: Optional[int] = None,
                       prefix: str = "bc_model_"):
    """
    从 BC 目录恢复参数，并把 encoder + actor 注入到当前 train_state 中。
    - 目录风格假设为: <bc_ckpt_path>/<prefix><step>/（例如 bc_model_120/）
    - 统一使用 Flax 的 restore_checkpoint 恢复（与此目录结构最匹配）
    """
    if not os.path.isdir(bc_ckpt_path):
        raise FileNotFoundError(f"找不到 BC 目录: {bc_ckpt_path}")

    # 自动找最新 step（如果未指定）
    if step is None:
        step = _find_latest_step(bc_ckpt_path, prefix)
        if step is None:
            raise FileNotFoundError(
                f"在 '{bc_ckpt_path}' 下找不到任何以 '{prefix}<step>' 命名的检查点目录。"
            )

    print(f"🔄 正在从 '{bc_ckpt_path}' 加载 BC 检查点（step={step}，prefix='{prefix}'）...")
    ckpt_root = os.path.abspath("models/bc_pretrained")
    os.makedirs(ckpt_root, exist_ok=True)
    # ---- 恢复（Flax）----
    bc_state = flax_checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_root,
        target=train_state,    # 结构模板
        step=step,
        prefix=prefix
    )
    if bc_state is None:
        raise ValueError("flax_checkpoints.restore_checkpoint 返回 None，未找到可用检查点。")
    print("  ✅ Flax 恢复成功")

    # ---- 注入 ----
    loaded_params = flax.core.unfreeze(bc_state.params)
    new_params    = flax.core.unfreeze(train_state.params)

    # 1) encoder 整棵子树
    if 'encoder' in loaded_params:
        new_params['encoder'] = loaded_params['encoder']
        print("  ✅ encoder 已注入")
    else:
        raise KeyError("BC 检查点中没有 'encoder' 子树。")

    # 2) actor 头（优先使用嵌套命名；否则用扁平键映射）
    if action_mode == 0:
        print("  ▶ Single-Flip：注入 actor_flip_* / actor_noop_*")
        # 直接从源拷贝到目标，路径完全一致
        new_params['actor_flip_dense_0'] = loaded_params['actor_flip_dense_0']
        new_params['actor_flip_output'] = loaded_params['actor_flip_output']
        new_params['actor_noop_dense_0'] = loaded_params['actor_noop_dense_0']
        new_params['actor_noop_output'] = loaded_params['actor_noop_output']
        print("    ✅ 扁平 Actor (mode 0) 参数已注入")
    else:
        print("  ▶ Multi-Flip：注入 actor_dense_* / actor_output")
        # 直接从源拷贝到目标，路径完全一致
        new_params['actor_dense_0'] = loaded_params['actor_dense_0']
        new_params['actor_dense_1'] = loaded_params['actor_dense_1']
        new_params['actor_output'] = loaded_params['actor_output']
        print("    ✅ 扁平 Actor (mode 1) 参数已注入")

    print("  ✅ 注入完成（仅 encoder+actor，critic/优化器为新初始化）")
    return train_state.replace(params=flax.core.freeze(new_params))
