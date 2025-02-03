import torch
from torchtext import data, vocab

class TextProcessor:
    def __init__(self, max_length=200):
        self.tokenizer = data.utils.get_tokenizer("basic_english")
        self.vocab = None
        self.max_length = max_length
        
    def fit(self, texts):
        self.vocab = vocab.build_vocab_from_iterator(
            (self.tokenizer(text) for text in texts),
            specials=["<unk>", "<pad>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        
    def transform(self, texts):
        sequences = [
            torch.tensor([self.vocab[token] for token in self.tokenizer(text)][:self.max_length])
            for text in texts
        ]
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.vocab["<pad>"])
    
    def vocab_size(self):
        return len(self.vocab) 