import torch
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
        lengths = (x != 0).sum(dim=1).cpu()

        x = self.embedding(x)
        x = self.dropout(x)
        
        # Pack the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), 
                                        batch_first=True, 
                                        enforce_sorted=False)
        # Process with LSTM
        packed_lstm_out, _ = self.lstm(packed_x)
        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        # Get the last non-padded output for each sequence
        x = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1]
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return nn.functional.normalize(x, p=2, dim=1) 