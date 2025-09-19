# import numpy as np
import io

import jax.numpy as jnp
import numpy as np
from scipy.sparse import lil_matrix  # create sparse matrix
from typing import List, Dict, Any, Tuple, Union
from flax.struct import dataclass

from src.utils.data_parser import parse_cnf, parse_sol



@dataclass
class GraphData:
    """A container for the graph representation of a SAT problem."""
    var_features: jnp.ndarray      # H_v: Variable node features (n, d_v)
    clause_features: jnp.ndarray   # H_c: Clause node features (m, d_c)
    A_pos: Union[lil_matrix, jnp.ndarray]           # A^+: Positive adjacency matrix (n, m)
    A_neg: Union[lil_matrix, jnp.ndarray]            # A^-: Negative adjacency matrix (n, m)


# 再每个step时，我们只用更新 assignment，对于整个问题的图表示，我们是不变的
@dataclass
class StaticGraphData:
    """只包含圖中靜態部分的容器。"""
    A_pos: jnp.ndarray
    A_neg: jnp.ndarray
    clause_features: jnp.ndarray
    num_vars: int
    num_clauses: int


@dataclass
class GNNInput:
    """為 GNN Encoder 量身定做的輸入容器。不包含任何多餘信息。"""
    static_var_features: jnp.ndarray   # Shape: (n_vars, 3)
    assignment: jnp.ndarray          # Shape: (n_vars,)
    clause_features: jnp.ndarray# Shape: (n_clauses, 3)
    A_pos: jnp.ndarray
    A_neg: jnp.ndarray

def create_graph_data(num_vars: int, num_clauses: int,
                      clauses: List[List[int]], assignment : jnp.ndarray) -> Tuple[GraphData, jnp.ndarray]:
    """
    Constructs the graph data structure for the GNN.

    Args:
        num_vars: The number of variables (n).
        num_clauses: The number of clauses (m).
        clauses: A list of clauses from the parser.

    Returns:
        A GraphData object containing the matrix representations.

    When H^0_v is first created, all the variables are set to 1.0, which represents True.
    """
    feature_dim = 0
    var_features = jnp.zeros((num_vars, feature_dim), dtype=jnp.float32)
    # at return new array
    var_features = var_features.at[:, 0].set(1.0)

    #
    var_features = var_features.at[:, 3].set(assignment.astype(jnp.float32))

    clause_features = jnp.zeros((num_clauses, feature_dim), dtype=jnp.float32)
    clause_features = clause_features.at[:, 2].set(1.0)

    # num_vars represents the number of rows, and num_clauses represents the number of columns.
    A_pos = lil_matrix((num_vars, num_clauses), dtype=jnp.float32)
    A_neg = lil_matrix((num_vars, num_clauses), dtype=jnp.float32)

    for clause_idx, clause  in enumerate(clauses):
        for literal in clause:
            var_idx = abs(literal) - 1
            if literal > 0:
                A_pos[var_idx, clause_idx] = 1.0
            else:
                A_neg[var_idx, clause_idx] = 1.0
    A_pos_dense = jnp.array(A_pos.todense())
    A_neg_dense = jnp.array(A_neg.todense())

    graph_input = GraphData(
        var_features=var_features,
        clause_features=clause_features,
        A_pos=A_pos_dense,
        A_neg=A_neg_dense
    )

    return graph_input


def create_static_graph(num_vars: int, num_clauses: int, clauses: jnp.ndarray) -> StaticGraphData:
    clauses_arr = jnp.array(clauses)
    var_indices = jnp.abs(clauses_arr) - 1

    base_clause_indices = jnp.arange(num_clauses)[:, None]
    clause_indices = jnp.broadcast_to(base_clause_indices, clauses_arr.shape)
    pos_mask = clauses_arr > 0
    neg_mask = clauses_arr < 0
    update_values_pos = jnp.where(pos_mask, 1.0, 0.0)
    update_values_neg = jnp.where(neg_mask, 1.0, 0.0)
    A_pos = jnp.zeros((num_vars, num_clauses), dtype=jnp.float32).at[var_indices, clause_indices].add(update_values_pos)
    A_neg = jnp.zeros((num_vars, num_clauses), dtype=jnp.float32).at[var_indices, clause_indices].add(update_values_neg)
    clause_feature_dim = 3
    clause_features = jnp.zeros((num_clauses, clause_feature_dim), dtype=jnp.float32)
    clause_features = clause_features.at[:, 2].set(1.0)
    return StaticGraphData(
        A_pos=A_pos,
        A_neg=A_neg,
        clause_features=clause_features,
        num_vars=num_vars,
        num_clauses=num_clauses
    )
