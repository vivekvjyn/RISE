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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score

from clustering import Model, Dataset, load_pitch
from utils import train_classifier, embed_triplet

console = Console()
CACHE = ".cache"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="Clustering"):
        prec, prec_processed = load_pitch(os.path.join(CACHE, "forms", "prec.pkl"))
        curr, curr_processed = load_pitch(os.path.join(CACHE, "forms", "curr.pkl"))
        succ, succ_processed = load_pitch(os.path.join(CACHE, "forms", "succ.pkl"))
        with open(os.path.join(CACHE, "forms", "svaras.pkl"), "rb") as f:
            svaras = pickle.load(f)
        with open(os.path.join(CACHE, "forms", "clusters.pkl"), "rb") as f:
            forms = pickle.load(f)

        svara_forms = list(zip(svaras, forms))
        unique_svara_forms = sorted(set(svara_forms))
        targets = np.array([unique_svara_forms.index(sf) for sf in svara_forms])

        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        train_idx, test_idx = next(gss.split(curr_processed, groups=targets))

        train_prec, val_prec, train_curr, val_curr, train_succ, val_succ, train_labels, val_labels = (
            train_test_split(prec_processed[train_idx], curr_processed[train_idx], succ_processed[train_idx], targets[train_idx], test_size=0.3, random_state=42, stratify=targets[train_idx])
        )

        train_dataset = Dataset((train_prec, train_curr, train_succ, train_labels))
        val_dataset = Dataset((val_prec, val_curr, val_succ, val_labels))
        test_dataset = Dataset((prec_processed[test_idx], curr_processed[test_idx], succ_processed[test_idx], targets[test_idx]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0)

        mlflow.log_params(args)

        model = Model(task="clustering", embed_dim=args["embed_dim"], depth=args["depth"], num_classes=len(unique_svara_forms)).to(device)
        train_classifier(model, train_loader, val_loader, args["epochs"], args["lr"], args["weight_decay"], args["early_stopping"], 0, "forms.pth")
        model.load_state_dict(torch.load(os.path.join("checkpoints", "forms.pth"), map_location=device))
        embeddings = embed_triplet(model, test_loader)
        nmi = normalized_mutual_info_score(targets[test_idx], HDBSCAN().fit_predict(embeddings))

        model = Model(task="clustering", embed_dim=args["embed_dim"], depth=args["depth"], num_classes=len(unique_svara_forms)).to(device)
        enc = os.path.join("checkpoints", "encoder.pth")
        model.prec_encoder.load_state_dict(torch.load(enc, map_location=device))
        model.curr_encoder.load_state_dict(torch.load(enc, map_location=device))
        model.succ_encoder.load_state_dict(torch.load(enc, map_location=device))
        model.apply_lora(r=4, alpha=16, dropout=0.0)
        train_classifier(model, train_loader, val_loader, args["epochs"], args["lr"], args["weight_decay"], args["early_stopping"], args["catchup"], "forms_lora.pth")
        model.load_state_dict(torch.load(os.path.join("checkpoints", "forms_lora.pth"), map_location=device))
        embeddings = embed_triplet(model, test_loader)
        pretrained_nmi = normalized_mutual_info_score(targets[test_idx], HDBSCAN().fit_predict(embeddings))

        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame({"nmi": [nmi], "nmi (pretrained)": [pretrained_nmi], "difference": [pretrained_nmi - nmi]})
        df.to_csv(os.path.join("results", "clustering_nmi.tsv"), sep="\t", index=False)

        table = RichTable(title="Clustering")
        table.add_column("NMI", style="green")
        table.add_column("NMI (Pretrained)", style="green")
        table.add_column("Diff", style="yellow")
        table.add_row(f"{nmi:.4f}", f"{pretrained_nmi:.4f}", f"{pretrained_nmi - nmi:.4f}")
        console.print(table)


def parse_args():
    parser = argparse.ArgumentParser()
    defaults = {"batch_size": 64, "depth": 5, "embed_dim": 48, "epochs": 200, "lr": 1e-3, "weight_decay": 1e-3, "catchup": 10, "early_stopping": 30}
    for k, v in defaults.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    try:
        cfg = yaml.safe_load(open("config.yaml")).get("clustering", {})
        for k in defaults:
            if k in cfg:
                setattr(args, k, cfg[k])
    except FileNotFoundError:
        pass
    return vars(args)


if __name__ == "__main__":
    main()
