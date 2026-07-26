import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_reconstruction(orig, rec, gt_mask, pred_mask, save_dir, sample_idx):
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    axes[0].plot(orig, linewidth=1.2, label="Original", color="#2c3e50")
    axes[0].plot(rec, linewidth=1.2, alpha=0.8, label="Reconstructed", color="#e74c3c")
    axes[0].set_ylabel("Pitch (cents)")
    axes[0].legend(loc="upper right", framealpha=0.9)

    axes[1].plot(gt_mask, linewidth=1.2, label="Ground truth mask", color="#27ae60")
    axes[1].set_ylabel("Mask")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc="upper right", framealpha=0.9)

    axes[2].plot(pred_mask, linewidth=1.2, label="Predicted mask", color="#e67e22")
    axes[2].set_ylabel("Mask")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"recon_{sample_idx}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"recon_{sample_idx}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(os.path.join(save_dir, f"recon_{sample_idx}.png"))


def plot_boxplot(data, title, ylabel, save_path, color="#3498db"):
    fig, ax = plt.subplots(figsize=(4, 4))
    bplot = ax.boxplot(
        data, patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#e74c3c", markeredgecolor="#e74c3c", markersize=5),
        medianprops=dict(color="#2c3e50", linewidth=1.5),
        whiskerprops=dict(color="#2c3e50"),
        capprops=dict(color="#2c3e50"),
        flierprops=dict(marker="o", markerfacecolor="#95a5a6", markersize=3, alpha=0.5),
    )
    bplot["boxes"][0].set_facecolor(color)
    bplot["boxes"][0].set_alpha(0.6)
    bplot["boxes"][0].set_edgecolor("#2c3e50")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    sns.despine(left=True, bottom=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(save_path)


def plot_confusion_matrix(cm, labels, raga, f1, pretrained, save_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=labels, yticklabels=labels,
        cbar=False, ax=ax, annot_kws={"size": 10},
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(f"{raga} -- F1: {f1:.4f}{' (Pretrained)' if pretrained else ''}", fontweight="bold", pad=10)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{raga}_{'pretrained_' if pretrained else ''}confusion_matrix"
    plt.savefig(os.path.join(save_dir, f"{fname}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(os.path.join(save_dir, f"{fname}.png"))
