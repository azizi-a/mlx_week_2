import torch
from tqdm import tqdm
from gensim.utils import simple_preprocess
from embeddings.train_word2vec import train_word2vec

class TextProcessor:
    def __init__(self, vector_size=128, word2vec_params=None):
        self.vector_size = vector_size
        self.word2vec_model = None
        self.vocab = None
        self.word_to_idx = None
        self.word2vec_params = word2vec_params
        
    def build_vocab(self, texts, use_text8=True):
        # Train or load cached Word2Vec model
        self.word2vec_model = train_word2vec(
            texts,
            vector_size=self.vector_size,
            word2vec_params=self.word2vec_params,
            use_text8=use_text8
        )
        
        print("Building vocabulary mapping...")
        self.vocab = list(self.word2vec_model.wv.index_to_key)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        print(f"Vocabulary size: {len(self.vocab)} words")
        
    def get_embedding_weights(self):
        print("Extracting embedding weights...")
        weights = torch.zeros((len(self.vocab), self.vector_size))
        for i, word in tqdm(enumerate(self.vocab), desc="Building embedding matrix", total=len(self.vocab)):
            weights[i] = torch.tensor(self.word2vec_model.wv[word])
        return weights
    
    def encode_text(self, texts):
        if self.word2vec_model is None:
            raise ValueError("Must call build_vocab before encoding texts")
            
        # First pass: encode and find max length
        encoded = []
        max_len = 0
        for text in tqdm(texts, desc="Encoding texts"):
            tokens = simple_preprocess(text)
            indices = [self.word_to_idx[token] for token in tokens 
                      if token in self.word_to_idx]
            max_len = max(max_len, len(indices))
            encoded.append(indices)
        
        # Second pass: pad sequences
        padded = []
        for seq in encoded:
            if len(seq) < max_len:
                padding = [0] * (max_len - len(seq))  # Use 0 as padding index
                padded.append(seq + padding)
            else:
                padded.append(seq)
        
        return torch.stack([torch.tensor(seq) for seq in padded])
    
    def vocab_size(self):
        return len(self.vocab) 