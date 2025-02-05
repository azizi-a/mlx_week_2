import torch.nn as nn

class DocumentTower(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, 
                 pretrained_embeddings=None):
        super().__init__()
        
        # Initialize embedding layer with pretrained weights if provided
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=True, num_layers=2, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return nn.functional.normalize(x, p=2, dim=1) 