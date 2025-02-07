import torch.nn as nn

class QueryTower(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64,
                 pretrained_embeddings=None):
        super().__init__()
        
        # Initialize embedding layer with pretrained weights if provided
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            
        # Single conv layer and single FC layer
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu()

        x = self.embedding(x)
        x = self.dropout(x)
        
        # Single conv layer
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)
        
        # Pack sequence and process with LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_x)
        x = hidden[-1]
        
        # Single FC layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return nn.functional.normalize(x, p=2, dim=1) 