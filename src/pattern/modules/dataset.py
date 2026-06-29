import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.sequences = torch.tensor(data, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].unsqueeze(0)
        return sequence

    def __str__(self):
        num_samples = len(self)
        return f"num_samples={num_samples}"
