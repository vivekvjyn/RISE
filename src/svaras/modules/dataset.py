import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, raga, labels, device):
        self.prec = torch.tensor(data[0], dtype=torch.float32)
        self.curr = torch.tensor(data[1], dtype=torch.float32)
        self.succ = torch.tensor(data[2], dtype=torch.float32)
        self.targets = torch.tensor(data[3], dtype=torch.long)
        self.raga = raga
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        prec = self.prec[idx].unsqueeze(0)
        curr = self.curr[idx].unsqueeze(0)
        succ = self.succ[idx].unsqueeze(0)
        targets = self.targets[idx]
        return prec, curr, succ, targets

    def __str__(self):
        num_samples = len(self)
        class_counts = torch.bincount(self.targets).numpy()
        class_distribution = {self.labels[i]: count for i, count in enumerate(class_counts)}
        return f"raga={self.raga}, num_samples={num_samples}, num_class={self.num_class} distribution={class_distribution}"

    @property
    def num_class(self):
        return len(torch.unique(self.targets))
