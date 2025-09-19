from pathlib import Path
from pysat.formula import CNF
from pysat.solvers import Glucose3, Minisat22

def solve_one(path: Path, output_dir: Path):
    print("─── 调试信息 ───")
    print("读取公式文件：", path.resolve())
    cnf = CNF(from_file=str(path))
    print(f"变量数 nv = {cnf.nv}, 子句数 = {len(cnf.clauses)}")

    # 用 Glucose3 求解，并在上下文内获取模型
    with Glucose3(bootstrap_with=cnf.clauses) as solver:
        sat = solver.solve()
        print("Glucose3 可满足？", sat)
        assert sat, f"{path} 应该是 SAT，但 Glucose3 给出 UNSAT"
        model = solver.get_model()      # <<< 一定要在这里调用
        assign = [int(lit > 0) for lit in model]

    # 可选：再验证一次其他解算器结果，确保一致
    with Minisat22(bootstrap_with=cnf.clauses) as solver2:
        print("Minisat22 可满足？", solver2.solve())

    # 写出结果
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / path.with_suffix('.sol').name
    out.write_text(' '.join(map(str, assign)))
    print("写出结果到：", out.resolve())
    return assign

if __name__ == "__main__":
    data_dir   = Path(r"../../data/uf50-213")
    output_dir = Path(r"../../data/uf50-213-answer")
    for cnf_file in data_dir.glob('*.cnf'):
        solve_one(cnf_file, output_dir)
