import os
import numpy as np
import torch
from gamakas.modules.meter import Meter

class Trainer:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def __call__(self, train_loader, val_loader, epochs, lr, weight_decay, early_stopping, catchup, filename):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        max_f1 = -np.inf
        patience = 0

        for epoch in range(epochs):
            self.model.freeze_encoders() if epoch < catchup else self.model.unfreeze_encoders()

            self.logger(f"Epoch {epoch + 1}/{epochs}:")

            train_loss, train_f1 = self._propagate(train_loader, optimizer, back_prop=True)
            self.logger(f"\tTrain Loss: {train_loss:.8f} | Train F1: {train_f1:.4f}")

            val_loss, val_f1 = self._propagate(val_loader, optimizer, back_prop=False)
            self.logger(f"\tValidation Loss: {val_loss:.8f} | Validation F1: {val_f1:.4f}")

            if val_f1 > max_f1:
                max_f1 = val_f1
                patience = 0
                self.model.save(filename=filename)
                self.logger(f"Model saved to {os.path.join(self.model.dir, filename)}")
            else:
                patience += 1
                if patience >= early_stopping:
                    self.logger("Early stopping triggered.")
                    break

    def _propagate(self, data_loader, optimizer, back_prop):
        self.model.train() if back_prop else self.model.eval()

        loss_fn = torch.nn.CrossEntropyLoss()
        meter = Meter()

        for i, (prec, curr, succ, targets) in enumerate(data_loader):
            self.logger.pbar(i + 1, len(data_loader))

            logits = self._predict(prec, curr, succ)
            loss = loss_fn(logits, targets)

            if back_prop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            meter(logits.detach().cpu(), targets.detach().cpu())

        loss = meter.loss
        f1 = meter.f1_score

        return loss, f1

    def _predict(self, prec, curr, succ):
        prec_mask = (torch.isnan(prec)).float()
        prec = torch.nan_to_num(prec, nan=0.0)
        prec_input = torch.cat([prec, prec_mask], dim=1)

        curr_mask = (torch.isnan(curr)).float()
        curr = torch.nan_to_num(curr, nan=0.0)
        curr_input = torch.cat([curr, curr_mask], dim=1)

        succ_mask = (torch.isnan(succ)).float()
        succ = torch.nan_to_num(succ, nan=0.0)
        succ_input = torch.cat([succ, succ_mask], dim=1)

        return self.model(prec_input, curr_input, succ_input)
