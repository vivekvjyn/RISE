import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Trainer:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def __call__(self, data_loader, epochs, lr, patience, accumulation_steps=1):
        device = next(self.model.parameters()).device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-6
        )
        best_loss = np.inf

        T = data_loader.dataset.tensors[1].shape[-1]

        for epoch in range(epochs):

            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            for i, batch in enumerate(data_loader):
                self.logger.pbar(i + 1, len(data_loader))
                embedding = batch[0].to(device)
                target = batch[1].to(device)

                recon = self.model(embedding, T)

                if recon.shape[-1] > target.shape[-1]:
                    recon = recon[..., :target.shape[-1]]
                elif target.shape[-1] > recon.shape[-1]:
                    target = target[..., :recon.shape[-1]]

                loss = F.mse_loss(recon, target)
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps

            avg_loss = total_loss / len(data_loader)
            scheduler.step(avg_loss)
            self.logger(f"Epoch {epoch + 1}/{epochs}: MSE = {avg_loss:.8f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model.save("decoder.pth")
                self.logger(f"  Decoder saved")

        self.logger(f"Training complete. Best MSE: {best_loss:.8f}")
