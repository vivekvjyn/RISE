import os
import numpy as np
import torch
import torch.nn.functional as F
from info_nce import InfoNCE
from sklearn.metrics import f1_score
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


def _device(model):
    return next(model.parameters()).device


def _mask(x):
    mask = torch.isnan(x).float()
    return torch.cat([torch.nan_to_num(x, nan=0.0), mask], dim=1)


def _accumulate(model, data_loader):
    device = _device(model)
    total_loss = 0.0
    all_pred = np.array([], dtype=np.int64)
    all_true = np.array([], dtype=np.int64)
    loss_fn = torch.nn.CrossEntropyLoss()

    for prec, curr, succ, targets in data_loader:
        prec, curr, succ, targets = prec.to(device), curr.to(device), succ.to(device), targets.to(device)
        logits = model(_mask(prec), _mask(curr), _mask(succ))
        total_loss += loss_fn(logits, targets).item()
        all_pred = np.concatenate((all_pred, logits.argmax(dim=1).cpu().numpy()))
        all_true = np.concatenate((all_true, targets.cpu().numpy()))

    return total_loss / len(data_loader), f1_score(all_true, all_pred, average="macro")


def train_ssl(model, augmenter, data_loader, epochs, lr, patience):
    device = _device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, threshold=1e-6, min_lr=1e-6)
    best_loss = np.inf
    loss_fn = InfoNCE()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("SSL training", total=epochs)
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for batch in data_loader:
                batch = batch.to(device)
                loss = loss_fn(model(_mask(batch)), _mask(augmenter(batch).to(device)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            avg = total / len(data_loader)
            scheduler.step(avg)
            if avg < best_loss:
                best_loss = avg
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.encoder.state_dict(), os.path.join("checkpoints", "encoder.pth"))
                torch.save(model.state_dict(), os.path.join("checkpoints", "model.pth"))
            progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Loss: {avg:.8f}")
    return best_loss


def train_classifier(model, train_loader, val_loader, epochs, lr, weight_decay, early_stopping, catchup, filename):
    device = _device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1 = -np.inf
    patience_counter = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("Classifier training", total=epochs)
        for epoch in range(epochs):
            model.set_encoders_trainable(epoch >= catchup)
            _, train_f1 = _accumulate(model, train_loader)
            _, val_f1 = _accumulate(model, val_loader)

            model.train()
            for prec, curr, succ, targets in train_loader:
                prec, curr, succ, targets = prec.to(device), curr.to(device), succ.to(device), targets.to(device)
                logits = model(_mask(prec), _mask(curr), _mask(succ))
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), os.path.join("checkpoints", filename))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    break

            progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")


def train_decoder(model, data_loader, epochs, lr, patience, accumulation_steps):
    device = _device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-6)
    best_loss = np.inf
    T = data_loader.dataset.tensors[1].shape[-1]

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("Decoder training", total=epochs)
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for i, batch in enumerate(data_loader):
                embedding, target = batch[0].to(device), batch[1].to(device)
                recon = model(embedding, T)
                if recon.shape[-1] > target.shape[-1]:
                    recon = recon[..., :target.shape[-1]]
                elif target.shape[-1] > recon.shape[-1]:
                    target = target[..., :recon.shape[-1]]

                loss = F.mse_loss(recon, target) / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * accumulation_steps

            avg_loss = total_loss / len(data_loader)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), os.path.join("checkpoints", "decoder.pth"))

            progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | MSE: {avg_loss:.8f}")
    return best_loss


def evaluate_classifier(model, data_loader):
    _, f1 = _accumulate(model, data_loader)
    return f1


def embed_triplet(model, data_loader):
    device = _device(model)
    model.eval()
    embeddings = np.array([])
    for prec, curr, succ, *_ in data_loader:
        emb = model.encode(_mask(prec.to(device)), _mask(curr.to(device)), _mask(succ.to(device))).detach().cpu().numpy()
        embeddings = np.concatenate((embeddings, emb), axis=0) if embeddings.size else emb
    return embeddings


def embed_ssl(model, data_loader):
    device = _device(model)
    model.eval()
    embeddings = np.array([])
    for batch in data_loader:
        inp = _mask(batch.squeeze(1).to(device) if batch.dim() == 3 else batch.to(device))
        emb = model(inp).detach().cpu().numpy()
        embeddings = np.concatenate((embeddings, emb), axis=0) if embeddings.size else emb
    return embeddings


def generate(model, data_loader, T):
    device = _device(model)
    model.eval()
    outputs = np.array([])
    for batch in data_loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        out = model(x, T).detach().cpu().numpy()
        outputs = np.concatenate((outputs, out), axis=0) if outputs.size else out
    return outputs
