import os
from typing import Tuple, List
# import numpy as np
import jax.numpy as jnp
from tqdm import tqdm


def parse_cnf(file_path: str) -> Tuple[int, int, List[List[int]]]:
    """
    Parses a DIMACS CNF file.

    Returns:
            A tuple containing:
            - num_vars (int): The number of variables.
            - num_clauses (int): The number of clauses.
            - clauses (List[List[int]]): A list of clauses, where each clause
              is a list of integers representing literals.
    """

    clauses = []
    num_vars = 0
    num_clauses = 0
    with open(file_path, "r") as f:

        for line in f:
            # remove the empty context
            line = line.strip()

            if line.startswith('c'):
                continue
            elif line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            else:
                parts = line.split()
                literals = [int(x) for x in parts]
                # except for the last element 0
                literals = literals[: -1]
                clauses.append(literals)

    return num_vars, num_clauses, clauses

def parse_sol(file_path: str) -> jnp.ndarray:
    """
    expect data
    Returns:
        A numpy array of integers (0 or 1) representing the variable assignments
    """

    with open(file_path,"r") as f:
        lines = f.readline()

    lines = lines.strip()
    elements = lines.split()
    answer = [int(x) for x in elements]
    return jnp.array(answer, dtype=jnp.int32)

def load_cnf_problems(cnf_data_dir: str):
    cnf_fnames = sorted([f for f in os.listdir(cnf_data_dir) if f.endswith('.cnf')])
    problems = []
    print(f"Found {len(cnf_fnames)} SAT instances.")
    for fname in tqdm(cnf_fnames, desc="LOADING CFN PROBLEMS"):
        cnf_path = os.path.join(cnf_data_dir, fname)
        num_vars, num_clauses, clauses = parse_cnf(cnf_path)
        problems.append({
            "num_vars": num_vars,
            "num_clauses": num_clauses,
            "clauses": clauses,
            "name": fname
        })
    return problems

# def test_function(file_path: str, expect_file_path: str):
#     cnf = parse_cnf(file_path)
#     expect_data = parse_sol(expect_file_path)
#
#     print(cnf)
#     print(expect_data)
#
# if __name__ == "__main__":
#     path_file = "../../data/uf20-91/uf20-01.cnf"
#     expect_path_file = "../../data/uf20-91-answer/uf20-01.sol"
#     test_function(path_file, expect_path_file)