import os
import argparse
import pickle
import pandas as pd
import torch
import numpy as np
import yaml
import mlflow
from rich.console import Console
from rich.table import Table as RichTable
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision

from pattern_recognition import Model, Dataset, load_pitch
from utils import embed_ssl

console = Console()
CACHE_DIR = ".cache"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="Pattern Recognition"):
        sequences = load_pitch(os.path.join(CACHE_DIR, "segments.pkl"))
        with open(os.path.join(CACHE_DIR, "ids.pkl"), "rb") as f:
            ids = pickle.load(f)

        model = Model(task="pattern", embed_dim=args["embed_dim"], depth=args["depth"], num_classes=len(set(ids))).to(device)
        model.encoder.load_state_dict(torch.load(os.path.join("checkpoints", "encoder.pth"), map_location=device))

        mlflow.log_params(args)

        sim_matrix = _similarity_matrix(model, sequences, args["window_size"])
        map_s, mrr, p1, p5 = _retrieval_metrics(sim_matrix, ids, device)

        mlflow.log_metrics({"map": map_s, "mrr": mrr, "p1": p1, "p5": p5})

        os.makedirs("results", exist_ok=True)
        pd.DataFrame({"map": [map_s], "mrr": [mrr], "p@1": [p1], "p@5": [p5]}).to_csv(
            os.path.join("results", "pattern_retrieval.tsv"), sep="\t", index=False
        )

        table = RichTable(title="Pattern Recognition")
        for col, val in [("MAP", map_s), ("MRR", mrr), ("P@1", p1), ("P@5", p5)]:
            table.add_column(col, style="green")
        table.add_row(f"{map_s:.4f}", f"{mrr:.4f}", f"{p1:.4f}", f"{p5:.4f}")
        console.print(table)


def _similarity_matrix(model, sequences, window_size):
    n = len(sequences)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sims = []
            x_splits, y_splits, num = _split(sequences[i], sequences[j], window_size)
            for k in range(num):
                x_loader = torch.utils.data.DataLoader(Dataset(np.array([x_splits[k]])), batch_size=1)
                y_loader = torch.utils.data.DataLoader(Dataset(np.array([y_splits[k]])), batch_size=1)
                x_emb = embed_ssl(model, x_loader)
                y_emb = embed_ssl(model, y_loader)
                sims.append(cosine_similarity(x_emb, y_emb)[0][0])
            sim[i][j] = np.mean(sims)
    return torch.tensor(sim, dtype=torch.float32)


def _split(x, y, window_size=200):
    num = max(max(len(x), len(y)) // window_size, 1)
    sx, sy = len(x) // num, len(y) // num
    return [x[i*sx:(i+1)*sx] for i in range(num)], [y[i*sy:(i+1)*sy] for i in range(num)], num


def _retrieval_metrics(sim_matrix, ids, device):
    preds = sim_matrix.clone()
    preds[torch.eye(len(ids), dtype=bool)] = -torch.inf
    preds = preds.flatten()
    targets = (ids[:, None] == ids[None, :])
    targets[torch.eye(len(ids), dtype=bool)] = False
    targets = targets.flatten()
    indexes = torch.arange(len(ids)).repeat_interleave(len(ids))

    return (
        RetrievalMAP()(preds, targets, indexes).item(),
        RetrievalMRR()(preds, targets, indexes).item(),
        RetrievalPrecision(k=1)(preds, targets, indexes).item(),
        RetrievalPrecision(k=5)(preds, targets, indexes).item(),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    defaults = {"depth": 5, "embed_dim": 48, "window_size": 200}
    for k, v in defaults.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    try:
        with open("configs.yaml") as f:
            cfg = yaml.safe_load(f).get("pattern_recognition", {})
        for k in defaults:
            if k in cfg:
                setattr(args, k, cfg[k])
    except FileNotFoundError:
        pass
    return vars(args)


if __name__ == "__main__":
    main()
