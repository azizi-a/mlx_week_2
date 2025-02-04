import torch
import torch.nn as nn
from .document_tower import DocumentTower
from .query_tower import QueryTower

class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.document_tower = DocumentTower(vocab_size, embed_dim, hidden_dim)
        self.query_tower = QueryTower(vocab_size, embed_dim, hidden_dim)
        
    def forward(self, documents, queries):
        doc_embeddings = self.document_tower(documents)
        query_embeddings = self.query_tower(queries)
        return torch.matmul(query_embeddings, doc_embeddings.t())
    
    def get_embeddings(self, x, is_query=True):
        return self.query_tower(x) if is_query else self.document_tower(x) 