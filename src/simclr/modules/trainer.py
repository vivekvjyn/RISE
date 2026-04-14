import os
import numpy as np
import torch
from info_nce import InfoNCE

class Trainer:
    def __init__(self, model, augmenter, tracker, logger):
        self.model = model
        self.augmenter = augmenter
        self.logger = logger
        self.tracker = tracker

    def __call__(self, data_loader, epochs, lr, patience):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=1e-6, min_lr=1e-6)
        min_loss = np.inf

        for epoch in range(epochs):
            self.logger(f"Epoch {epoch + 1}/{epochs}:")

            loss = self._propagate(data_loader, optimizer)
            scheduler.step(loss)
            self.logger(f"\tLoss: {loss:.8f}")
            self.tracker.update(loss)

            if loss < min_loss:
                min_loss = loss
                self.model.encoder.save('encoder.pth')
                self.logger(f"Model saved to {os.path.join(self.model.encoder.dir, 'encoder.pth')}")

        self.tracker.plot()

    def _propagate(self, data_loader, optimizer):
        loss_fn = InfoNCE()
        total_loss = 0.0

        for i, (batch) in enumerate(data_loader):
            self.logger.pbar(i + 1, len(data_loader))

            query_pitch = batch.clone()
            query = self._project(query_pitch)
            positive_pitch = self.augmenter(batch)
            positive_key = self._project(positive_pitch)

            loss = loss_fn(query, positive_key)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def _project(self, pitch):
        silence_mask = (torch.isnan(pitch)).float()
        pitch = torch.nan_to_num(pitch, nan=0)
        input = torch.cat([pitch, silence_mask], dim=1)

        return self.model(input)
