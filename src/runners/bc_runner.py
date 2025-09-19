import os
import random
import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm
from datetime import datetime

from flax.training import checkpoints  # ⭐ Flax 的 checkpoint 工具

from src.utils.check_sat import check_satisfiability
from src.utils.data_parser import parse_cnf, parse_sol
from src.utils.graph_constructor import create_graph_data, GraphData
from src.models.base_gnn import SATGNN
from src.learners.bc_learner import create_train_state, train_step, eval_step



def load_dataset(cnf_data_dir: str, expect_data_dir: str):
    cnf_fnames = sorted(f.split('.')[0] for f in os.listdir(cnf_data_dir) if f.endswith('.cnf'))
    expect_fnames = sorted(f.split('.')[0] for f in os.listdir(expect_data_dir) if f.endswith('.sol'))

    dataset = []
    print(f"Found {len(cnf_fnames)} SAT instances.")

    for fname in tqdm(cnf_fnames, desc="Loading and preparing data"):
        cnf_path = os.path.join(cnf_data_dir, f"{fname}.cnf")
        sol_path = os.path.join(expect_data_dir, f"{fname}.sol")

        # 1. parse files
        num_vars, num_clauses, clauses_list = parse_cnf(cnf_path)
        solution = parse_sol(sol_path)

        # 2. create graph
        graph_input, labels = create_graph_data(num_vars, num_clauses, clauses_list, solution)

        dataset.append({
            "graph": graph_input,
            "labels": labels,
            "clauses": clauses_list
        })

    return dataset


def collate_fn(batch: list) -> tuple:
    graphs = [item['graph'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 1. stack each feature
    var_features_batch = jnp.stack([g.var_features for g in graphs])
    clause_features_batch = jnp.stack([g.clause_features for g in graphs])

    # 2.
    A_pos_batch = jnp.stack([jnp.array(g.A_pos.todense()) for g in graphs])
    A_neg_batch = jnp.stack([jnp.array(g.A_neg.todense()) for g in graphs])

    # 3.
    labels_batch = jnp.stack(labels)

    # 4.
    batch_graph = GraphData(
        var_features=var_features_batch,
        clause_features=clause_features_batch,
        A_pos=A_pos_batch,
        A_neg=A_neg_batch
    )
    return batch_graph, labels_batch


def main():
    seed = 0
    learning_rate = 1e-4
    epoches = 100
    key = jax.random.PRNGKey(0)
    batch_size = 32

    cnf_data_dir = '../../data/uf5-15'
    expect_data_dir = '../../data/uf5-15-answer'
    dataset = load_dataset(cnf_data_dir, expect_data_dir)
    random.shuffle(dataset)
    train_ratio = 0.8
    split_idx = int(len(dataset) * train_ratio)
    train_dataset, val_dataset = dataset[:split_idx], dataset[split_idx:]
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # init
    dummy_graph_input = dataset[0]['graph']

    model = SATGNN()

    key, init_key = jax.random.split(key)
    state = create_train_state(model, init_key, learning_rate, dummy_graph_input)

    print("TrainState created.")

    # ⭐ 保存目录
    time_str = datetime.now().strftime("%m-%d_%H-%M")
    ckpt_dir = os.path.abspath(f"./experiments/bc/{time_str}/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_sol_acc = -1.0  # 记录最优解率

    for epoch in range(1, epoches+1):
        train_solved_count = 0
        random.shuffle(train_dataset)
        train_metrics_list = []
        train_loader = [train_dataset[i: i+batch_size]
                        for i in range(0, len(train_dataset), batch_size)]

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epoches} [Train]"):
            batched_graph, batched_labels = collate_fn(batch)
            state, metrics, logits = train_step(state, batched_graph, batched_labels)
            train_metrics_list.append(metrics)
            predicted_assignments = np.array(jnp.argmax(logits, axis=-1))  # (batch_size, num_vars)
            for i in range(len(batch)):
                assignment = predicted_assignments[i]
                clauses = batch[i]['clauses']
                if check_satisfiability(clauses, assignment):
                    train_solved_count += 1

        val_metrics_list = []
        val_solved_count = 0
        val_loader = [val_dataset[i:i + batch_size] for i in range(0, len(val_dataset), batch_size)]
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epoches} [Val]"):
            batched_graph, batched_labels = collate_fn(batch)
            metric, logits = eval_step(state, batched_graph, batched_labels)
            val_metrics_list.append(metric)
            predicted_assignments = np.array(jnp.argmax(logits, axis=-1))
            for i in range(len(batch)):
                assignment = predicted_assignments[i]
                clauses = batch[i]['clauses']
                if check_satisfiability(clauses, assignment):
                    val_solved_count += 1

        avg_train_loss = np.mean([m['loss'] for m in train_metrics_list])
        avg_train_acc = np.mean([m['accuracy'] for m in train_metrics_list])
        avg_val_loss = np.mean([m['loss'] for m in val_metrics_list])
        avg_val_acc = np.mean([m['accuracy'] for m in val_metrics_list])

        train_sol_acc = train_solved_count / len(train_dataset)
        val_sol_acc = val_solved_count / len(val_dataset)

        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}, Train Var-Acc: {avg_train_acc:.4f}, Train Sol-Acc: {train_sol_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Var-Acc: {avg_val_acc:.4f}, Val Sol-Acc: {val_sol_acc:.4f}"
        )

        # ⭐ 保存最优模型
        if val_sol_acc > best_val_sol_acc:
            best_val_sol_acc = val_sol_acc
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=state.params,
                step=epoch,
                prefix="best_",
                overwrite=True
            )
            print(f"✅ New best model saved at epoch {epoch} with Val Sol-Acc={val_sol_acc:.4f}")

    print("Training finished!")


if __name__ == '__main__':
    main()
