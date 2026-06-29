import os
import argparse
import pickle
import torch
from logger import Logger

from pattern import Model, Evaluator, Embedder, Table, load_pitch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(log_dir="logs")
    args = parse_args()

    sequences = load_pitch(os.path.join("dataset", "segments.pkl"))
    with open(os.path.join("dataset", "ids.pkl"), "rb") as f:
        labels = torch.tensor(pickle.load(f)).to(device)

    model = Model(embed_dim=args.embed_dim, num_classes=len(set(labels)), depth=args.depth).to(device)
    model.encoder.load(os.path.join("checkpoints", "encoder.pth"), device)
    embedder = Embedder(model, logger)
    evaluator = Evaluator(embedder, logger, device, args.window_size)
    map, mrr, p1, p5 = evaluator(sequences, labels)
    table = Table()
    table.insert(map, mrr, p1, p5)

def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--embed-dim', type=int, default=48, help='dimension of embedding space (default: 48)')
    parser.add_argument('--depth', type=int, default=5, help='number of inception modules (default: 5)')
    parser.add_argument('--window-size', type=int, default=100, help='window size for similarity computation (default: 200)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
