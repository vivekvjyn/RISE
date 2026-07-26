import argparse
import os
import pickle
import pandas as pd
import torch
import yaml
import mlflow
from rich.console import Console
from rich.table import Table as RichTable
from sklearn.model_selection import train_test_split

from classification import Model, Dataset, load_pitch
from utils import train_classifier, evaluate_classifier

console = Console()
CACHE_DIR = ".cache"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name=f"Classification ({args['dataset']})"):
        prec = load_pitch(os.path.join(CACHE_DIR, args["dataset"], "prec.pkl"))
        curr = load_pitch(os.path.join(CACHE_DIR, args["dataset"], "curr.pkl"))
        succ = load_pitch(os.path.join(CACHE_DIR, args["dataset"], "succ.pkl"))
        with open(os.path.join(CACHE_DIR, args["dataset"], "svaras.pkl"), "rb") as f:
            svaras = pickle.load(f)
        with open(os.path.join(CACHE_DIR, args["dataset"], "labels.pkl"), "rb") as f:
            labels = pickle.load(f)

        train_prec, test_prec, train_curr, test_curr, train_succ, test_succ, train_labels, test_labels = (
            train_test_split(prec, curr, succ, svaras, test_size=0.4, random_state=42, stratify=svaras)
        )
        train_prec, val_prec, train_curr, val_curr, train_succ, val_succ, train_labels, val_labels = (
            train_test_split(train_prec, train_curr, train_succ, train_labels, test_size=0.3, random_state=42, stratify=train_labels)
        )

        train_dataset = Dataset((train_prec, train_curr, train_succ, train_labels), args["dataset"], labels)
        val_dataset = Dataset((val_prec, val_curr, val_succ, val_labels), args["dataset"], labels)
        test_dataset = Dataset((test_prec, test_curr, test_succ, test_labels), args["dataset"], labels)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0)

        mlflow.log_params(args)

        model = Model(task="classification", embed_dim=args["embed_dim"], depth=args["depth"], num_classes=train_dataset.num_class).to(device)
        train_classifier(model, train_loader, val_loader, args["epochs"], args["lr"], args["weight_decay"], args["early_stopping"], 0, f'{args["dataset"]}.pth')
        model.load_state_dict(torch.load(os.path.join("checkpoints", f'{args["dataset"]}.pth'), map_location=device))
        f1 = evaluate_classifier(model, test_loader)

        model = Model(task="classification", embed_dim=args["embed_dim"], depth=args["depth"], num_classes=train_dataset.num_class).to(device)
        enc_path = os.path.join("checkpoints", "encoder.pth")
        model.prec_encoder.load_state_dict(torch.load(enc_path, map_location=device))
        model.curr_encoder.load_state_dict(torch.load(enc_path, map_location=device))
        model.succ_encoder.load_state_dict(torch.load(enc_path, map_location=device))
        model.apply_lora(r=8, alpha=16, dropout=0.05)
        train_classifier(model, train_loader, val_loader, args["epochs"], args["lr"], args["weight_decay"], args["early_stopping"], args["catchup"], f'{args["dataset"]}_lora.pth')
        model.load_state_dict(torch.load(os.path.join("checkpoints", f'{args["dataset"]}_lora.pth'), map_location=device))
        pretrained_f1 = evaluate_classifier(model, test_loader)

        os.makedirs("results", exist_ok=True)
        results_path = os.path.join("results", f"{args['dataset']}_f1.tsv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path, sep="\t")
        else:
            df = pd.DataFrame(columns=["raga", "f1", "f1 (pretrained)", "difference"])
        mask = df["raga"] == args["dataset"]
        if mask.any():
            df.loc[mask, ["f1", "f1 (pretrained)", "difference"]] = [f1, pretrained_f1, pretrained_f1 - f1]
        else:
            df.loc[len(df)] = {"raga": args["dataset"], "f1": f1, "f1 (pretrained)": pretrained_f1, "difference": pretrained_f1 - f1}
        df.to_csv(results_path, sep="\t", index=False)

        table = RichTable(title=f"Classification ({args['dataset']})")
        table.add_column("Raga", style="cyan")
        table.add_column("F1", style="green")
        table.add_column("F1 (Pretrained)", style="green")
        table.add_column("Diff", style="yellow")
        table.add_row(args["dataset"], f"{f1:.4f}", f"{pretrained_f1:.4f}", f"{pretrained_f1 - f1:.4f}")
        console.print(table)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="abhogi", choices=["abhogi", "begada", "kalyani", "mohanam", "sahana", "saveri", "sri"])
    defaults = {"batch_size": 64, "depth": 5, "embed_dim": 48, "epochs": 200, "lr": 1e-3, "weight_decay": 1e-3, "catchup": 10, "early_stopping": 30}
    for k, v in defaults.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    try:
        with open("configs.yaml") as f:
            cfg = yaml.safe_load(f).get("classification", {})
        for k in defaults:
            if k in cfg:
                setattr(args, k, cfg[k])
    except FileNotFoundError:
        pass
    return vars(args)


if __name__ == "__main__":
    main()
