import torch


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, raga, labels):
        self.prec = torch.tensor(data[0], dtype=torch.float32)
        self.curr = torch.tensor(data[1], dtype=torch.float32)
        self.succ = torch.tensor(data[2], dtype=torch.float32)
        self.targets = torch.tensor(data[3], dtype=torch.long)
        self.raga = raga
        self.labels = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.prec[idx].unsqueeze(0),
            self.curr[idx].unsqueeze(0),
            self.succ[idx].unsqueeze(0),
            self.targets[idx],
        )

    @property
    def num_class(self):
        return len(torch.unique(self.targets))


class ClusteringDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.prec = torch.tensor(data[0], dtype=torch.float32)
        self.curr = torch.tensor(data[1], dtype=torch.float32)
        self.succ = torch.tensor(data[2], dtype=torch.float32)
        self.targets = torch.tensor(data[3], dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.prec[idx].unsqueeze(0),
            self.curr[idx].unsqueeze(0),
            self.succ[idx].unsqueeze(0),
            self.targets[idx],
        )

    @property
    def num_class(self):
        return len(torch.unique(self.targets))


class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.sequences = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx].unsqueeze(0)


class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0)
