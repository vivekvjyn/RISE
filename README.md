# RISE — Rāga Independent Svara Encoder

Self-supervised representation learning for *svaras* in Carnatic music, aimed at
transcription and at the analysis tasks that follow from it: *svara* classification,
*svara*-form clustering, melodic pattern retrieval and *svara* synthesis.

A *svara* is not a fixed pitch. It is an ornamented pitch contour (*gamaka*) whose
shape depends on the *svaras* around it, and expert annotations of it are scarce.
RISE addresses both facts: it pretrains an encoder contrastively on unannotated
recordings, and it gives the downstream models the melodic context each *svara* was
performed in.

This codebase accompanies the DLfM 2026 paper and the Master's thesis in Sound and
Music Computing at Universitat Pompeu Fabra:

> *A Rāga Independent Encoder for Svara Representation in Carnatic Music* —
> Vivek Vijayan, Thomas Nuttall, Xavier Serra

---

## Setup

```bash
git clone https://github.com/MTG/RISE.git
cd RISE
pip install -e .              # add [viz] for the UMAP projection figure
```

## Usage

```bash
./run.sh <experiment> [options]
```

`run.sh` is a thin wrapper around `python -m rise`. Defaults come from
`configs.yaml`; anything passed on the command line overrides them.

```bash
./run.sh preprocess                              # build every dataset (run first)
./run.sh pretrain                                # train the encoder
./run.sh classification --ragas kalyani sahana   # override any config value
./run.sh figures --list                          # see the available figures
./run.sh classification --help                   # per-experiment options
```

## Experiments

Run in this order: `preprocess` builds every dataset, and `pretrain` produces the
encoder that the four downstream experiments load.

| Experiment | What it does | Metrics |
|---|---|---|
| `preprocess` | Cleans the pitch tracks of all three corpora and cuts them into the datasets below, as fixed train/validation/test splits | — |
| `pretrain` | Contrastive pretraining (InfoNCE) of the InceptionTime encoder on plausible *svaras* sampled from the beat grid of the unannotated CMR corpus | InfoNCE loss |
| `classification` | Fine-tunes the pretrained encoder with LoRA for *svara* classification, once per *rāga*, against the same model trained from scratch | Macro F1 |
| `clustering` | Clusters *svara*-form embeddings with HDBSCAN against expert annotations, on forms held out from training entirely | NMI |
| `pattern_recognition` | Retrieves melodic phrases by the mean cosine similarity of their windowed embeddings | MAP, MRR, P@k |
| `synthesis` | Reconstructs *svara* contours from the frozen encoder's embeddings with a transposed-Inception decoder | DTW distance, periodicity error, pitch position error |
| `figures` | Draws the illustrative figures of the thesis from the corpora and the reported numbers | — |

## Corpora

| Corpus | What it contributes |
|---|---|
| **CMR** — Carnatic Music Rhythm | No melodic annotation, but beat and downbeat markers; the unannotated corpus the encoder is pretrained on |
| **Varnam** — Carnatic Varnam | *Svara* and *svara*-form annotations across seven *rāgas*, for fine-tuning and evaluation |
| **IAMMS** — Indian Art Music Melodic Similarity | Labelled melodic phrases, for testing whether the representation transfers beyond the individual *svara* |

Pitch tracks are extracted with the FTA-Net Carnatic model and tonics with multipitch
tonic identification, both from [compIAM](https://mtg.github.io/compIAM/); the
extracted tracks are shipped under `data/`.

## Layout

```
src/rise/
├── cli.py               one sub-command per experiment, defaults from configs.yaml
├── config.py            configs.yaml loading
├── console.py           the shared Rich console and its output primitives
├── paths.py             every path in the project, derived from the repository root
├── reproducibility.py   seeding and device selection
├── training.py          the training loop shared by classification and clustering
├── data/                corpora, preprocessing, torch datasets, splits
├── dsp/                 pitch conversion, cleaning, augmentation, periodicity
├── nn/                  InceptionTime backbone, co-attention, LoRA, task models
├── evaluation/          retrieval, reconstruction and clustering metrics
├── experiments/         the seven sub-commands
└── figures/             the figure design system and the plots it produces
```

Outputs go to `results/` (metric tables, TSV) and `figures/` (PNG at 300 dpi).
`checkpoints/` holds only the two pretrained models, `encoder.pth` and
`decoder.pth`; the weights a downstream experiment trains are intermediates and go
to `.cache/runs/<run>/`, alongside the state that resumes an interrupted run.
Deleting `.cache/` costs compute and nothing else. MLflow logs parameters, metrics
and figures for every run.

## Figures

Figures come from two places. Each experiment draws the figure that reports its own
result — a confusion matrix per *rāga* for `classification`, the UMAP projection for
`clustering` (with `--projection`), the phrase-wise average precision for
`pattern_recognition` — and the `figures` sub-command draws the ones that illustrate
the data:

```bash
./run.sh figures                                        # all of them
./run.sh figures pitch-track --raga sahana --svara N    # or one, re-pointed
./run.sh figures beat-grid --audio path/to/recording.wav
```

Every figure is drawn through one design system (`src/rise/figures/style.py`): one
type scale, one palette assigned in a fixed order and checked for colour-vision
separation, a recessive grid and axes, and no dark variant, since the destination is
paper.

**No figure carries a title.** Each is placed in the thesis with a caption that
already names and explains it, and the captions are what the code is written
against — where a caption says the *svaras* are named by syllable, or the beats are
red and the downbeats black, or the phrase identifier is on the *x*-axis, the figure
does that and nothing more. Each is also drawn at the width its `\includegraphics`
gives it, so LaTeX places it unscaled and 8 pt inside the figure is 8 pt on the page.

Where a figure has to demonstrate something the caption claims — "three distinct
realisations of the *svara* *dha*", "four variants" — the generator searches the
corpus for an excerpt that actually contains it rather than hard-coding an index.

## Model

The encoder `F(x)` is an InceptionTime stack of five blocks. Each block convolves its
input with kernels of width 9, 19 and 39 in parallel and adds a bottleneck residual,
giving one block a view of a *svara* at three time scales at once. Its output,
averaged over time, is a 48-dimensional embedding.

Pretraining follows SimCLR: for each contour an augmented view forms the positive
pair and the rest of the batch the negatives, optimised with InfoNCE through a
projection head that is discarded afterwards. The augmentations — temporal resizing,
localised time warping and small pitch drift — imitate the way a singer varies one
*svara* between renditions.

The downstream classifier replicates the encoder three times, over the preceding
*svara*, the current one and the succeeding one, reads each with a GRU, and joins
them by co-attention referenced to the final hidden state of the current stream. Only
LoRA adapters and the classification head are trained, so a few hundred annotations
per *rāga* are enough to adapt without discarding what pretraining learned.

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
