import argparse
import os
import pickle
import numpy as np
import torch
import yaml
import mlflow
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from synthesis import Model, InceptionDecoder, Dataset, normalize, denormalize, zero_pad, dtw_normalized, dft_oscillation_count, dft_pitch_position
from utils import train_decoder, embed_ssl, generate, plot_reconstruction, plot_boxplot

console = Console()
CACHE_DIR = ".cache"
VARNAM_RAGAS = ["abhogi", "begada", "kalyani", "mohanam", "sahana", "saveri", "sri"]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="Synthesis"):
        encoder = Model(task="synth", embed_dim=args["embed_dim"], depth=args["depth"], out_dim=args["out_dim"]).to(device)
        encoder.load_state_dict(torch.load(os.path.join("checkpoints", "encoder.pth"), map_location=device))

        decoder = InceptionDecoder(embed_dim=args["embed_dim"], depth=args["depth"]).to(device)
        decoder.load_state_dict(torch.load(os.path.join("checkpoints", "decoder.pth"), map_location=device))

        with open(os.path.join(CACHE_DIR, "cmr.pkl"), "rb") as f:
            all_data = pickle.load(f)
        train_data = all_data[:25000]
        test_data = _load_varnam()
        mlflow.log_params(args)

        padded = zero_pad(normalize(train_data))
        extract_loader = torch.utils.data.DataLoader(Dataset(padded), batch_size=args["extract_batch_size"], shuffle=False)
        embeddings = embed_ssl(encoder, extract_loader)
        targets = torch.tensor(padded, dtype=torch.float32).unsqueeze(1)
        silence_masks = (~torch.isnan(targets)).float()
        targets = torch.cat([torch.nan_to_num(targets, nan=0.0), silence_masks], dim=1)
        embed_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(embeddings, targets), batch_size=args["batch_size"])

        padded_test = zero_pad(normalize(test_data))
        extract_loader = torch.utils.data.DataLoader(Dataset(padded_test), batch_size=args["extract_batch_size"], shuffle=False)
        embeddings = embed_ssl(encoder, extract_loader)
        targets = torch.tensor(padded_test, dtype=torch.float32).unsqueeze(1)
        silence_masks = (~torch.isnan(targets)).float()
        targets = torch.cat([torch.nan_to_num(targets, nan=0.0), silence_masks], dim=1)
        embed_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(embeddings, targets), batch_size=args["batch_size"])

        T = padded_test.shape[1]
        outputs = generate(decoder, embed_loader, T)

        recon = denormalize(np.clip(outputs[:, 0], 0, 1))
        pred_mask = (outputs[:, 1] > 0.5).astype(float)

        dtw_dists, osc_orig, osc_recon, pitch_orig, pitch_recon = [], [], [], [], []

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task("Evaluating", total=len(test_data))
            for i in range(len(test_data)):
                orig = test_data[i]
                rec = recon[i][:len(orig)].copy()
                rec[pred_mask[i][:len(orig)] == 0] = np.nan
                orig_d = denormalize(padded_test[i][:len(orig)])
                orig_d[np.isnan(orig)] = np.nan

                valid = ~np.isnan(orig_d) & ~np.isnan(rec)
                if valid.sum() > 0:
                    dtw_dists.append(dtw_normalized(orig_d[valid], rec[valid]))

                osc_orig.append(dft_oscillation_count(orig_d))
                osc_recon.append(dft_oscillation_count(rec))
                pitch_orig.append(dft_pitch_position(orig_d))
                pitch_recon.append(dft_pitch_position(rec))

                if i < args["num_plots"]:
                    gt_mask = (~np.isnan(padded_test[i][:len(orig)])).astype(float)
                    plot_reconstruction(orig_d, rec, gt_mask, pred_mask[i][:len(orig)], "results/synthesis", i)

                progress.update(task, advance=1)

        mlflow.log_metrics({
            "avg_dtw": np.mean(dtw_dists),
            "avg_osc_diff": np.mean(np.abs(np.array(osc_orig) - np.array(osc_recon))),
            "avg_pitch_diff": np.mean(np.abs(np.array(pitch_orig) - np.array(pitch_recon))),
        })

        for name, data, title, color in [
            ("dtw_distance", dtw_dists, "DTW Distance", "#3498db"),
            ("pitch_diff", np.abs(np.array(pitch_orig) - np.array(pitch_recon)), "Pitch Diff", "#e74c3c"),
            ("osc_diff", np.abs(np.array(osc_orig) - np.array(osc_recon)), "Oscillation Diff", "#f39c12"),
        ]:
            plot_boxplot(data, title, "Distance", os.path.join("results", "synthesis", f"{name}.png"), color=color)


def _load_varnam():
    data = []
    for raga in VARNAM_RAGAS:
        with open(os.path.join(CACHE_DIR, raga, "curr.pkl"), "rb") as f:
            data.extend(pickle.load(f))
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    defaults = {"batch_size": 64, "extract_batch_size": 256, "depth": 5, "embed_dim": 48, "epochs": 200, "lr": 1e-6, "out_dim": 16, "patience": 10, "accumulation_steps": 8, "num_plots": 20}
    for k, v in defaults.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    try:
        with open("configs.yaml") as f:
            cfg = yaml.safe_load(f).get("synthesis", {})
        for k in defaults:
            if k in cfg:
                setattr(args, k, cfg[k])
    except FileNotFoundError:
        pass
    return vars(args)


if __name__ == "__main__":
    main()
