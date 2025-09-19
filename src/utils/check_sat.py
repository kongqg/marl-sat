import numpy as np


def check_satisfiability(clauses: list, assignment: np.ndarray) -> bool:
    """
    Checks if a given assignment satisfies a set of clauses.

    Args:
        clauses: A list of clauses, e.g., [[1, -2], [-1, 3, 4]].
        assignment: A numpy array of 0s and 1s, where the index corresponds to the
                    variable (0-indexed) and the value is its assignment.

    Returns:
        True if the assignment satisfies all clauses, False otherwise.
    """
    if not clauses:
        return True  # An empty set of clauses is trivially satisfied.

    # 遍歷每一個子句
    for clause in clauses:
        clause_is_satisfied = False
        # 檢查子句中的每一個文字
        for literal in clause:
            var_idx = abs(literal) - 1
            var_assignment = assignment[var_idx]

            # 檢查文字是否為真
            # 正文字 (e.g., 5) 且變數賦值為 1 (True)
            if literal > 0 and var_assignment == 1:
                clause_is_satisfied = True
                break  # 這個子句已滿足，跳到下一個子句
            # 負文字 (e.g., -5) 且變數賦值為 0 (False)
            elif literal < 0 and var_assignment == 0:
                clause_is_satisfied = True
                break  # 這個子句已滿足，跳到下一個子句

        # 如果遍歷完一個子句中的所有文字後，該子句仍未被滿足，
        # 則整個公式都不可能被滿足，直接返回 False。
        if not clause_is_satisfied:
            return False

    # 如果所有子句都被滿足了，返回 True。
    return True