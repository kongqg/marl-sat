import os
import re


def parse_cnf_file(file_path: str) -> list[list[int]]:
    """
    解析一个 DIMACS CNF 格式的文件，提取所有子句。

    Args:
        file_path (str): .cnf 文件的路径。

    Returns:
        list[list[int]]: 一个包含所有子句的列表。
                         每个子句也是一个列表，包含代表文字的整数。
    """
    clauses = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 忽略注释行和 'p' 定义行
                if line.startswith('c') or line.startswith('p') or not line:
                    continue

                # 读取子句，去除末尾的 0
                literals = [int(x) for x in line.split()[:-1]]
                if literals:
                    clauses.append(literals)
    except FileNotFoundError:
        print(f"  [错误] CNF 文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"  [错误] 解析 CNF 文件时出错 {file_path}: {e}")
        return None
    return clauses


def verify_solution(clauses: list[list[int]], solution_str: str) -> tuple[bool, list[int] | None]:
    """
    验证一个解是否满足给定的所有子句。

    Args:
        clauses (list[list[int]]): 所有的子句。
        solution_str (str): '0' 和 '1' 组成的解字符串。

    Returns:
        tuple[bool, list[int] | None]: 一个元组，第一个元素是布尔值表示是否验证成功，
                                     第二个元素在失败时返回导致失败的子句。
    """
    # 将解 '0101...' 转换为布尔值列表 (False, True, False, True, ...)
    # 注意：变量索引从 1 开始，所以我们需要在查找时进行调整
    assignments = [val == '1' for val in solution_str]

    for clause in clauses:
        is_clause_satisfied = False
        for literal in clause:
            var_index = abs(literal) - 1  # 转换为 0-based 索引

            # 检查索引是否在解的范围内
            if var_index >= len(assignments):
                print(f"  [错误] 变量索引 {abs(literal)} 超出解的范围 (长度 {len(assignments)})")
                return False, clause

            assigned_value = assignments[var_index]

            # 如果是正文字 (e.g., 9)，且其赋值为 True，则子句满足
            if literal > 0 and assigned_value:
                is_clause_satisfied = True
                break  # 该子句已满足，检查下一个子句

            # 如果是负文字 (e.g., -11)，且其赋值为 False，则子句满足
            if literal < 0 and not assigned_value:
                is_clause_satisfied = True
                break  # 该子句已满足，检查下一个子句

        # 如果遍历完一个子句的所有文字后，子句仍未被满足，则该解是错误的
        if not is_clause_satisfied:
            return False, clause

    # 如果所有子句都被满足了
    return True, None


def main():
    """主执行函数"""
    # --- 请根据您的文件结构修改以下路径 ---
    # 包含解的日志文件路径
    solutions_file_path = '../../experiments/mappo_runs/2025-09-08_23-55-38/test_solutions.txt'
    # 存放 .cnf 文件的目录路径
    cnf_directory_path = '../../data/uf20-50'
    # -----------------------------------------

    if not os.path.exists(solutions_file_path):
        print(f"错误：找不到解文件 '{solutions_file_path}'。请检查路径。")
        return

    print(f"🔍 开始验证, 读取解文件: '{solutions_file_path}'")
    print(f"📂 CNF 文件目录: '{cnf_directory_path}'")
    print("-" * 50)

    verified_count = 0
    failed_count = 0
    skipped_count = 0

    with open(solutions_file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            # 使用正则表达式解析行，更稳健
            match = re.search(r"Problem: ([\w.-]+), Solved: True, .* Solution: ([01]+)", line)

            if not match:
                skipped_count += 1
                # 如果行格式不匹配或 "Solved" 不为 "True"，则跳过
                continue

            cnf_filename, solution_str = match.groups()
            print(f"[{i:03d}] 正在验证: {cnf_filename}")

            # 1. 构造 CNF 文件的完整路径
            full_cnf_path = os.path.join(cnf_directory_path, cnf_filename)

            # 2. 解析 CNF 文件
            clauses = parse_cnf_file(full_cnf_path)
            if clauses is None:
                failed_count += 1
                print(f"  ❌ 验证失败: 无法读取或解析 CNF 文件。")
                print("-" * 50)
                continue

            # 3. 验证解
            is_correct, failing_clause = verify_solution(clauses, solution_str)

            # 4. 输出结果
            if is_correct:
                verified_count += 1
                print(f"  ✅ 验证通过！")
            else:
                failed_count += 1
                print(f"  ❌ 验证失败！")
                print(f"     解: {solution_str}")
                print(f"     未满足的子句: {failing_clause}")

            print("-" * 50)

    print("\n🎉 验证完成！")
    print("=" * 20)
    print(f"  - ✅ 验证通过: {verified_count} 个")
    print(f"  - ❌ 验证失败: {failed_count} 个")
    print(f"  - ⏭️  跳过 (未解决或格式错误): {skipped_count} 个")
    print("=" * 20)


if __name__ == '__main__':
    main()