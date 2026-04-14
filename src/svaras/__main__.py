import argparse
import os
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from svaras import Model, Trainer, Evaluator, Logger, Dataset, Table, load_pitch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    args = parse_args()

    # Load dataset
    prec = load_pitch(os.path.join("dataset", args.dataset, "prec.pkl"))
    curr = load_pitch(os.path.join("dataset", args.dataset, "curr.pkl"))
    succ = load_pitch(os.path.join("dataset", args.dataset, "succ.pkl"))
    with open(os.path.join("dataset", args.dataset, "svaras.pkl"), "rb") as f:
        svaras = pickle.load(f)
    with open(os.path.join("dataset", args.dataset, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)

    # Prepare datasets
    train_prec, test_prec, train_curr, test_curr, train_succ, test_succ, train_labels, test_labels = train_test_split(prec, curr, succ, svaras, test_size=0.4, random_state=42, stratify=svaras)
    train_prec, val_prec, train_curr, val_curr, train_succ, val_succ, train_labels, val_labels = train_test_split(train_prec, train_curr, train_succ, train_labels, test_size=0.3, random_state=42, stratify=train_labels)
    train_dataset = Dataset((train_prec, train_curr, train_succ, train_labels), args.dataset, labels, device)
    val_dataset = Dataset((val_prec, val_curr, val_succ, val_labels), args.dataset, labels, device)
    test_dataset = Dataset((test_prec, test_curr, test_succ, test_labels), args.dataset, labels, device)
    print('Train set:', train_dataset)
    print('Validation set:', val_dataset)
    print('Test set:', test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate
    model = Model(args.embed_dim, train_dataset.num_class, args.depth, args.dataset).to(device)
    trainer = Trainer(model, logger)
    trainer(train_loader, val_loader, args.epochs, args.lr, args.weight_decay, args.early_stopping, catchup=0, filename=args.dataset + ".pth")
    model.load(args.dataset + ".pth", device)
    evaluator = Evaluator(model, logger)
    f1 = evaluator(test_loader, labels, args.dataset, False)

    model = Model(args.embed_dim, train_dataset.num_class, args.depth, args.dataset).to(device)
    encoder_path = os.path.join("model_weights", "encoder.pth")
    model.prec_encoder.load(encoder_path, device)
    model.curr_encoder.load(encoder_path, device)
    model.succ_encoder.load(encoder_path, device)
    model.apply_lora(r=4, alpha=16, dropout=0.0)
    trainer = Trainer(model, logger)
    trainer(train_loader, val_loader, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.catchup, filename=args.dataset + "_lora.pth")
    model.load(args.dataset + "_lora.pth", device)
    evaluator = Evaluator(model, logger)
    simclr_f1 = evaluator(test_loader, labels, args.dataset, True)

    table = Table(args.dataset)
    table.insert(f1, simclr_f1)

def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--dataset', type=str, default='abhogi', choices=['abhogi', 'begada', 'kalyani', 'mohanam', 'sahana', 'saveri', 'sri'], help='dataset to use for finetuning (default: abhogi)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--catchup', type=int, default=10, help='number of epochs to freeze encoders (default: 10)')
    parser.add_argument('--early-stopping', type=int, default=10, help='early stopping patience (default: 10)')
    parser.add_argument('--embed-dim', type=int, default=48, help='dimension of embedding space (default: 30)')
    parser.add_argument('--depth', type=int, default=5, help='number of inception modules (default: 5)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay (default: 1e-3)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
