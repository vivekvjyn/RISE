import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

class Evaluator:
    def __init__(self, logger, timestep=0.0100006335037896):
        self.timestep = timestep
        self.logger = logger

    def __call__(self, prec, curr, succ, embeddings, truths, model):
        scanner = HDBSCAN()
        clusters = scanner.fit_predict(embeddings)
        self.logger(f'\tNumber of clusters: {len(set(clusters))}')

        nmi = normalized_mutual_info_score(truths, clusters)
        self.logger(f'\tNMI: {nmi:.8f}\n')

        self._plot_umap(embeddings, clusters, truths, model)

        for i in range(len(clusters)):
            self._plot_clusters(prec[i], curr[i], succ[i], clusters[i], model)

        return nmi

    def _plot_clusters(self, prec, curr, succ, cluster, model):
        plt.figure(figsize=(5, 2))
        plt.axvspan(0, len(prec) * self.timestep, facecolor='lightgray', alpha=0.5)
        plt.axvspan((len(prec) + len(curr)) * self.timestep, (len(prec) + len(curr) + len(succ)) * self.timestep, facecolor='lightgray', alpha=0.5)
        plt.plot(np.arange(len(prec)) * self.timestep, prec, c='gray')
        plt.plot(np.arange(len(prec), len(prec) + len(curr)) * self.timestep, curr, c='green')
        plt.plot(np.arange(len(prec) + len(curr), len(prec) + len(curr) + len(succ)) * self.timestep, succ, c='gray')
        plt.xlabel('Time (s)')
        plt.xlim(0, (len(prec) + len(curr) + len(succ)) * self.timestep)
        svaras = ["S", "R1", "R2", "G1", "G2", "M1", "M2", "P", "D1", "D2", "N1", "N2"]
        plt.yticks(np.arange(-2400, 2400, 100), labels=[svaras[i % 12] for i in range(-2400 // 100, 2400 // 100)])
        plt.ylim(np.min(curr) - 100, np.max(curr) + 100)
        plt.grid()
        plt.tight_layout()
        save_dir = os.path.join("results", "clustering", model, str(cluster))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{len(os.listdir(save_dir))}.png"))
        plt.close()

    def _plot_umap(self, embeddings, clusters, truths, model):
        clusters = np.array(clusters) - np.min(clusters)
        truths = np.array(truths) - np.min(truths)
        reducer = UMAP(n_components=2, random_state=42, n_jobs=1, n_neighbors=5)
        umap = reducer.fit_transform(embeddings)
        scaler = MinMaxScaler()
        umap = scaler.fit_transform(umap)

        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title(f'Predicted Clusters', fontsize=28)
        plt.scatter(umap[:, 0], umap[:, 1], c=clusters, cmap=cmap, s=5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.title(f'Ground Truths', fontsize=28)
        plt.scatter(umap[:, 0], umap[:, 1], c=truths, cmap=cmap, s=8)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid()
        plt.tight_layout()
        save_dir = os.path.join("results", "clustering")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"umap_{model}.png"))
        plt.close()
