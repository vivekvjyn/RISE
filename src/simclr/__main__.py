import argparse
import pickle
import torch

from simclr import Model, Trainer, Logger, Augmenter, Dataset, Tracker, normalize, zero_pad

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    augmenter = Augmenter()
    tracker = Tracker()
    args = parse_args()

    with open("dataset/cmr.pkl", "rb") as f:
        dataset = pickle.load(f)
    normalized_data = normalize(dataset)
    padded_data = zero_pad(normalized_data)

    data_loader = torch.utils.data.DataLoader(Dataset(padded_data, device), batch_size=args.batch_size, shuffle=True)
    model = Model(embed_dim=args.embed_dim, out_dim=args.out_dim, depth=args.depth).to(device)
    simclr = Trainer(model, augmenter, tracker, logger)

    simclr(data_loader, epochs=args.epochs, lr=args.lr, patience=args.patience)

def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 512)')
    parser.add_argument('--depth', type=int, default=5, help='number of inception modules (default: 5)')
    parser.add_argument('--embed-dim', type=int, default=48, help='dimension of embedding space (default: 48)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--out-dim', type=int, default=16, help='dimension of projection space (default: 16)')
    parser.add_argument('--patience', type=int, default=20, help='patience for learning rate scheduler (default: 20)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
