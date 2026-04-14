import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, input, hidden):
        e = torch.bmm(self.linear(input), hidden.permute(1, 2, 0))
        context = torch.bmm(input.permute(0, 2, 1), e.softmax(dim=1)).permute(0, 2, 1)

        return context.squeeze(1)
