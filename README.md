# RISE — Raga Independent Svara Encoder

Self-supervised learning for raga independent svara representation, primarily aimed at Carnatic music transcription and related tasks such as performance analysis and melodic pattern recognition. This codebase accompanies both the DLfM 2026 submission and the Master Thesis, Sound and Music Computing (2026 batch), Universitat Pompeu Fabra:

> *A Raga Independent Encoder for Svara Representation in Carnatic Music* — Vivek Vijayan, Thomas Nuttall, Xavier Serra

---

## Setup

```bash
git clone https://github.com/MTG/RISE.git
cd RISE
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
./run.sh <experiment>
```

Experiments: `preprocess`, `pretrain`, `classification`, `clustering`, `pattern_recognition`, `synthesis`

## Experiments

| Experiment | Description | Metrics |
|---|---|---|
| `preprocess` | Extract pitch contours from CMR and Varnam datasets, sample svara candidates | — |
| `pretrain` | Self-supervised pretraining of InceptionTime encoder using InfoNCE loss on unannotated pitch contours with augmentations (time warping, pitch drifting) | — |
| `classification` | Fine-tune pretrained model on annotated Varnam data for svara classification using LoRA | `F1 Score` |
| `clustering` | Cluster svara embeddings using HDBSCAN against expert svara-form annotations | `Normalized Mutual Information` |
| `pattern_recognition` | Retrieve melodic patterns using cosine similarity on encoder embeddings | `Mean Average Precision`, `Mean Reciprocal Rank`, `Precision@k` |
| `synthesis` | Reconstruct pitch contours from encoder embeddings using a transpose-inception decoder | `Dynamic Time Warping Distance`, `Periodicity Error`, `Pitch Position Error` |

## Citation

```bibtex
@inproceedings{10.1145/3815723.3815730,
  author    = {Vijayan, Vivek and Nuttall, Thomas and Serra, Xavier},
  title     = {A Rāga Independent Encoder for Svara Representation in Carnatic Music},
  year      = {2026},
  isbn      = {9798400723698},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3815723.3815730},
  doi       = {10.1145/3815723.3815730},
  booktitle = {Proceedings of the 13th International Conference on Digital Libraries for Musicology},
  pages     = {56--62},
  numpages  = {7},
  keywords  = {Carnatic music, Svara representation, Representation learning, Self-supervised learning, Pitch contour, Contrastive learning},
  series    = {DLfM '26}
}
```
