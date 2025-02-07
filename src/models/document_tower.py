import torch
import torch.nn as nn

class DocumentTower(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, 
                 pretrained_embeddings=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=False, num_layers=1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def freeze_parameters(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu()
        
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths,
                                        batch_first=True, 
                                        enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_x)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        x = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1]
        x = self.batch_norm(x)
        x = self.fc(x)
        
        return nn.functional.normalize(x, p=2, dim=1) 