import os
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from svaras.modules.meter import Meter

class Evaluator:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger
        self.meter = Meter()

    def __call__(self, data_loader, labels, raga, pretrained):
        f1 = self._propagate(data_loader)
        self.logger(f"\n\tTest F1: {f1:.4f}\n")
        self._confusion_matrix(labels, raga, pretrained)

        return f1

    def _propagate(self, data_loader):
        self.model.eval()

        loss_fn = torch.nn.CrossEntropyLoss()

        for i, (prec, curr, succ, targets) in enumerate(data_loader):
            self.logger.pbar(i + 1, len(data_loader))

            logits = self._predict(prec, curr, succ)
            loss = loss_fn(logits, targets)

            self.meter(logits.detach().cpu(), targets.detach().cpu())

        f1 = self.meter.f1_score

        return f1

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

    def _confusion_matrix(self, labels, raga, pretrained):
        cm = confusion_matrix(self.meter._true, self.meter._pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels, cbar=False)
        plt.xlabel('Predicted svara')
        plt.ylabel('True svara')
        plt.title(f"{raga} | F1: {self.meter.f1_score:.4f}{' | Pretrained weights' if pretrained else ''}")
        plt.tight_layout()
        os.makedirs(os.path.join('results', 'classification'), exist_ok=True)
        plt.savefig(os.path.join('results', 'classification', f"{raga}_{'pretrained_' if pretrained else ''}confusion_matrix.png"))
        plt.close()
