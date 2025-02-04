import torch
import torch.nn as nn

class QueryTower(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x):
        x = self.tower[0](x)
        lstm_out, _ = self.tower[1](x)
        x = self.tower[2](lstm_out[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=1) 