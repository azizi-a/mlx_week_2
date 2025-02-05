import torch
from torchtext import data, vocab

class TextProcessor:
    def __init__(self):
        self.tokenizer = data.utils.get_tokenizer("basic_english")
        self.vocab = None
        
    def build_vocab(self, texts):
        self.vocab = vocab.build_vocab_from_iterator(
            (self.tokenizer(text) for text in texts),
            specials=["<unk>", "<pad>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        
    def encode_text(self, texts):
        sequences = [
            torch.tensor([self.vocab[token] for token in self.tokenizer(text)])
            for text in texts
        ]
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.vocab["<pad>"])
    
    def vocab_size(self):
        return len(self.vocab) 