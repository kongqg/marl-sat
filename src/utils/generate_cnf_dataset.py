import random
import os
from typing import List

def generate_sat_cnf(num_vars: int, num_clauses: int, clause_size: int = 3, seed: int = None) -> str:
    """
    生成“保证可满足”的 k-SAT（默认 3-SAT）：
    1) 先采样一个隐藏解 sigma[v] ∈ {+1,-1}
    2) 每个子句抽取 k 个不同变量，并保证至少一个文字与 sigma 一致，从而子句为真
    3) 禁止子句内出现重复变量（因此也不会出现 v 和 -v 同时出现的永真子句）
    """
    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random

    assert 1 <= clause_size <= num_vars
    # 隐藏解：sigma[v] = True/False
    sigma = {v: rnd.choice([True, False]) for v in range(1, num_vars + 1)}

    lines: List[str] = [f"p cnf {num_vars} {num_clauses}"]

    for _ in range(num_clauses):
        # 选 k 个不同的变量
        vars_k = rnd.sample(range(1, num_vars + 1), clause_size)

        # 随机挑一个位置，强制与 sigma 一致（保证该子句可满足）
        sat_pos = rnd.randrange(clause_size)
        lits = []
        for i, v in enumerate(vars_k):
            if i == sat_pos:
                # 使该文字与隐藏解一致：若 sigma[v]==True，就放正 v；否则放负 v
                lit = v if sigma[v] else -v
            else:
                # 其他位置随机给正负；不影响“至少一个为真”的保证
                lit = v if rnd.choice([True, False]) else -v
            lits.append(lit)

        #（vars_k 不含重复，故不会出现同变量正负同现）
        lines.append(" ".join(map(str, lits)) + " 0")

    return "\n".join(lines)


def generate_cnf_dataset_sat(num_files: int, num_vars: int, num_clauses: int, save_dir: str, seed: int = None):
    """
    批量生成“保证 SAT”的 CNF 文件。命名与目录结构保持你的习惯。
    """
    os.makedirs(save_dir, exist_ok=True)
    rnd = random.Random(seed) if seed is not None else random
    for i in range(1, num_files + 1):
        # 为了可复现，也可以给每个文件一个子种子
        cnf_str = generate_sat_cnf(num_vars, num_clauses, clause_size=3, seed=rnd.randrange(1 << 30))
        filename = os.path.join(save_dir, f"uf{num_vars}-{i:03d}.cnf")
        with open(filename, "w") as f:
            f.write(cnf_str)
    print(f"✅ 已生成 {num_files} 个“保证 SAT”的 CNF 文件，目录：{save_dir}")


# 示例：生成 1000 个“保证 SAT”的 5变量/15子句 数据
# 保存路径你按自己的结构改
num_var = 40
num_clauses = 170
generate_cnf_dataset_sat(1000, num_var, num_clauses, f"../../data/uf{num_var}-{num_clauses}", seed=42)
