import argparse
import os
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

from synth import Model, Logger, Dataset, Embedder, Generator, zero_pad, normalize, denormalize, Decoder, Trainer
from synth import dtw_normalized, harmonic_distance, dft_oscillation_count, dft_pitch_position, slope_difference, linear_regression_difference

VARNAM_RAGAS = ["abhogi", "begada", "kalyani", "mohanam", "sahana", "saveri", "sri"]

def load_varnam():
    varnam_data = []
    for raga in VARNAM_RAGAS:
        with open(os.path.join("dataset", raga, "curr.pkl"), "rb") as f:
            varnam_data.extend(pickle.load(f))
    return varnam_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    args = parse_args()

    model = Model(embed_dim=args.embed_dim, out_dim=args.out_dim, depth=args.depth).to(device)
    model.encoder.load("encoder.pth", device)
    embedder = Embedder(model, logger)

    decoder = Decoder(embed_dim=args.embed_dim, depth=args.depth).to(device)
    decoder.load("decoder.pth", device)

    with open("dataset/cmr.pkl", "rb") as f:
        all_data = pickle.load(f)
    train_dataset = all_data[:25000]
    test_dataset = load_varnam()
    random.shuffle(test_dataset)
    logger(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")

    trainer = Trainer(decoder, logger)
    normalized_data = normalize(train_dataset)
    padded_data = zero_pad(normalized_data)
    extract_loader = torch.utils.data.DataLoader(Dataset(padded_data, device), batch_size=args.extract_batch_size, shuffle=False)
    embeddings = embedder(extract_loader)
    targets = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(1)
    silence_masks = (~torch.isnan(targets)).float()
    targets = torch.nan_to_num(targets, nan=0.0)
    targets = torch.cat([targets, silence_masks], dim=1)
    embed_dataset = torch.utils.data.TensorDataset(embeddings, targets)
    embed_loader = torch.utils.data.DataLoader(embed_dataset, batch_size=args.batch_size)
    #trainer(embed_loader, args.epochs, args.lr, args.patience, args.accumulation_steps)

    normalized_data = normalize(test_dataset)
    padded_data = zero_pad(normalized_data)

    extract_loader = torch.utils.data.DataLoader(Dataset(padded_data, device), batch_size=args.extract_batch_size, shuffle=False)
    embeddings = embedder(extract_loader)

    targets = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(1)
    silence_masks = (~torch.isnan(targets)).float()
    targets = torch.nan_to_num(targets, nan=0.0)
    targets = torch.cat([targets, silence_masks], dim=1)

    embed_dataset = torch.utils.data.TensorDataset(embeddings, targets)
    embed_loader = torch.utils.data.DataLoader(embed_dataset, batch_size=args.batch_size)

    T = padded_data.shape[1]
    generator = Generator(model=decoder, logger=logger, T=T)
    outputs = generator(embed_loader)

    recon = denormalize(np.clip(outputs[:, 0], 0, 1))
    pred_mask = (outputs[:, 1] > 0.5).astype(float)

    out_dir = os.path.join(logger.dir, "reconstructions")
    os.makedirs(out_dir, exist_ok=True)

    dtw_distances = []
    harmonic_dists = []
    osc_counts_orig = []
    osc_counts_recon = []
    pitch_pos_orig = []
    pitch_pos_recon = []
    lr_slope_diffs = []

    num_plots = min(args.num_plots, len(test_dataset))
    for i in range(len(test_dataset)):
        logger.pbar(i + 1, len(test_dataset))
        orig = test_dataset[i]
        rec = recon[i][:len(orig)].copy()
        mask = pred_mask[i][:len(orig)]

        rec[mask == 0] = np.nan

        orig_denorm = denormalize(padded_data[i][:len(orig)])
        orig_denorm[np.isnan(orig)] = np.nan

        valid = ~np.isnan(orig_denorm) & ~np.isnan(rec)
        if valid.sum() > 0:
            dtw_distances.append(dtw_normalized(orig_denorm[valid], rec[valid]))
            harmonic_dists.append(harmonic_distance(orig_denorm[valid], rec[valid]))

        osc_orig = dft_oscillation_count(orig_denorm)
        osc_recon = dft_oscillation_count(rec)
        osc_counts_orig.append(osc_orig)
        osc_counts_recon.append(osc_recon)

        pitch_orig = dft_pitch_position(orig_denorm)
        pitch_recon = dft_pitch_position(rec)
        pitch_pos_orig.append(pitch_orig)
        pitch_pos_recon.append(pitch_recon)

        slope_diff, _ = linear_regression_difference(orig_denorm, rec)
        lr_slope_diffs.append(slope_diff)

        if i < num_plots:
            fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1, 1]})
            axes[0].plot(orig_denorm, label="original")
            axes[0].plot(rec, label="reconstructed", alpha=0.7)
            axes[0].set_ylabel("cents")
            axes[0].set_title(f"Sample {i}")
            axes[0].legend()
            gt_mask = (~np.isnan(padded_data[i][:len(orig)])).astype(float)
            axes[1].plot(gt_mask, label="ground truth mask", color="green")
            axes[1].set_ylabel("mask")
            axes[1].set_ylim(-0.1, 1.1)
            axes[1].legend()
            axes[2].plot(mask, label="predicted mask", color="orange")
            axes[2].set_ylabel("mask")
            axes[2].set_xlabel("time")
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"recon_{i}.png"))
            plt.close()

    logger(f"\nSaved {num_plots} plots to {out_dir}")
    logger(f"  Avg DTW distance:           {np.mean(dtw_distances):.2f}")
    logger(f"  Avg oscillation diff:       {np.mean(np.abs(np.array(osc_counts_orig) - np.array(osc_counts_recon))):.2f}")
    logger(f"  Avg pitch position diff:    {np.mean(np.abs(np.array(pitch_pos_orig) - np.array(pitch_pos_recon))):.2f} cents")

    results_dir = os.path.join(logger.dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    pitch_diffs = np.abs(np.array(pitch_pos_orig) - np.array(pitch_pos_recon))
    osc_diffs = np.abs(np.array(osc_counts_orig) - np.array(osc_counts_recon))

    metrics = [
        ("dtw_distance", dtw_distances, "DTW Distance (cents)", "#2196F3"),
        ("pitch_position_diff", pitch_diffs, "Pitch Position Diff (cents)", "#E91E63"),
        ("oscillation_diff", osc_diffs, "Oscillation Diff", "#FF9800"),
    ]

    for name, data, title, color in metrics:
        fig, ax = plt.subplots(figsize=(5, 5))
        bplot = ax.boxplot(data, patch_artist=True, showmeans=True,
                           meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
        bplot["boxes"][0].set_facecolor(color)
        bplot["boxes"][0].set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel(title.split("(")[-1].replace(")", "") if "(" in title else "Value")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{name}.png"), dpi=150)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 512)')
    parser.add_argument('--extract-batch-size', type=int, default=256, help='batch size for embedding extraction (default: 256)')
    parser.add_argument('--depth', type=int, default=5, help='number of inception modules (default: 5)')
    parser.add_argument('--embed-dim', type=int, default=48, help='dimension of embedding space (default: 48)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--out-dim', type=int, default=16, help='dimension of projection space (default: 16)')
    parser.add_argument('--patience', type=int, default=20, help='patience for learning rate scheduler (default: 20)')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='gradient accumulation steps (default: 4)')
    parser.add_argument('--num-plots', type=int, default=20, help='number of reconstruction plots to save (default: 20)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
