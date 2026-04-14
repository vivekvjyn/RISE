import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x.unsqueeze(0)
        return x
