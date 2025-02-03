import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.document_tower = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.query_tower = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def _encode(self, x, tower):
        x = tower[0](x)
        lstm_out, _ = tower[1](x)
        x = tower[2](lstm_out[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=1)
        
    def forward(self, documents, queries):
        doc_embeddings = self._encode(documents, self.document_tower)
        query_embeddings = self._encode(queries, self.query_tower)
        return torch.matmul(query_embeddings, doc_embeddings.t())
    
    def get_embeddings(self, x, is_query=True):
        return self._encode(x, self.query_tower if is_query else self.document_tower) 