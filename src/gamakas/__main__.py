import os
import argparse
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from gamakas import Model, Evaluator, Embedder, Trainer, Logger, Dataset, Table, load_pitch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    args = parse_args()

    # Load dataset
    prec, prec_processed = load_pitch(os.path.join("dataset", "forms", "prec.pkl"))
    curr, curr_processed = load_pitch(os.path.join("dataset", "forms", "curr.pkl"))
    succ, succ_processed = load_pitch(os.path.join("dataset", "forms", "succ.pkl"))
    with open(os.path.join("dataset", "forms", "svaras.pkl"), "rb") as f:
        svaras = pickle.load(f)
    with open(os.path.join("dataset", "forms", "clusters.pkl"), "rb") as f:
        forms = pickle.load(f)
    svara_forms = list(zip(svaras, forms))
    unique_svara_forms = sorted(set(svara_forms))
    targets = np.array([unique_svara_forms.index(svara_form) for svara_form in svara_forms])

    # Prepare datasets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx, test_idx = next(gss.split(curr_processed, groups=targets))
    train_prec = prec_processed[train_idx]
    test_prec = prec_processed[test_idx]
    train_curr = curr_processed[train_idx]
    test_curr = curr_processed[test_idx]
    train_succ = succ_processed[train_idx]
    test_succ = succ_processed[test_idx]
    train_labels = targets[train_idx]
    test_labels = targets[test_idx]
    prec = [prec[i] for i in test_idx]
    curr = [curr[i] for i in test_idx]
    succ = [succ[i] for i in test_idx]
    #train_prec, test_prec, train_curr, test_curr, train_succ, test_succ, train_labels, test_labels = train_test_split(prec_processed, curr_processed, succ_processed, targets, test_size=0.5, random_state=42, stratify=targets)
    train_prec, val_prec, train_curr, val_curr, train_succ, val_succ, train_labels, val_labels = train_test_split(train_prec, train_curr, train_succ, train_labels, test_size=0.3, random_state=42, stratify=train_labels)
    train_dataset = Dataset((train_prec, train_curr, train_succ, train_labels), device)
    val_dataset = Dataset((val_prec, val_curr, val_succ, val_labels), device)
    test_dataset = Dataset((test_prec, test_curr, test_succ, test_labels), device)
    print('Train set:', train_dataset)
    print('Validation set:', val_dataset)
    print('Test set:', test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Train model
    model = Model(embed_dim=args.embed_dim, num_classes=len(unique_svara_forms), depth=args.depth).to(device)
    trainer = Trainer(model, logger)
    trainer(train_loader, val_loader, args.epochs, args.lr, args.weight_decay, args.early_stopping, catchup=0, filename="forms.pth")

    # Get embeddings
    model.load("forms.pth", device)
    embedder = Embedder(model, logger)
    evaluator = Evaluator(logger)
    embeddings = embedder(test_loader)

    # Evaluate clustering
    nmi = evaluator(prec, curr, succ, embeddings, test_labels, 'scratch')

    # Train model
    model = Model(embed_dim=args.embed_dim, num_classes=len(unique_svara_forms), depth=args.depth).to(device)
    encoder_path = os.path.join("checkpoints", "encoder.pth")
    model.prec_encoder.load(encoder_path, device)
    model.curr_encoder.load(encoder_path, device)
    model.succ_encoder.load(encoder_path, device)
    model.apply_lora(r=4, alpha=16, dropout=0.0)
    trainer = Trainer(model, logger)
    trainer(train_loader, val_loader, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.catchup, filename="forms_lora.pth")

    # Get embeddings
    model.load("forms_lora.pth", device)
    embedder = Embedder(model, logger)
    evaluator = Evaluator(logger)
    embeddings = embedder(test_loader)

    # Evaluate clustering
    simclr_nmi = evaluator(prec, curr, succ, embeddings, test_labels, 'pretrained')

    table = Table()
    table.insert(nmi, simclr_nmi)

def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--catchup', type=int, default=10, help='number of epochs to freeze encoders (default: 10)')
    parser.add_argument('--early-stopping', type=int, default=10, help='early stopping patience (default: 30)')
    parser.add_argument('--embed-dim', type=int, default=48, help='dimension of embedding space (default: 48)')
    parser.add_argument('--depth', type=int, default=5, help='number of inception modules (default: 5)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay (default: 1e-3)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
