import argparse
import os
import pickle
import torch
import yaml
import mlflow
from rich.console import Console

from pretrain import Model, Dataset, normalize, zero_pad
from utils import augment, train_ssl

console = Console()
CACHE = ".cache"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="Pretrain"):
        with open(os.path.join(CACHE, "cmr.pkl"), "rb") as f:
            dataset = pickle.load(f)

        padded = zero_pad(normalize(dataset))
        loader = torch.utils.data.DataLoader(Dataset(padded), batch_size=args["batch_size"], shuffle=True)

        model = Model(task="ssl", embed_dim=args["embed_dim"], depth=args["depth"], out_dim=args["out_dim"]).to(device)
        for param in model.encoder.parameters():
            param.requires_grad = False

        mlflow.log_params(args)
        train_ssl(model, augment, loader, args["epochs"], args["lr"], args["patience"])


def parse_args():
    parser = argparse.ArgumentParser()
    defaults = {"batch_size": 256, "depth": 5, "embed_dim": 48, "epochs": 1000, "lr": 1e-5, "out_dim": 16, "patience": 20}
    for k, v in defaults.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    try:
        cfg = yaml.safe_load(open("config.yaml")).get("pretrain", {})
        for k in defaults:
            if k in cfg:
                setattr(args, k, cfg[k])
    except FileNotFoundError:
        pass
    return vars(args)


if __name__ == "__main__":
    main()
