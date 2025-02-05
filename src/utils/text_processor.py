import torch
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec

class Word2VecCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.progress_bar = None

    def on_train_begin(self, model):
        print("\nStarting Word2Vec training...")
        self.progress_bar = tqdm(total=model.epochs, desc="Training Word2Vec")

    def on_epoch_end(self, model):
        self.progress_bar.update(1)
        self.epoch += 1

    def on_train_end(self, model):
        self.progress_bar.close()
        print("Word2Vec training completed!")

class TextProcessor:
    def __init__(self, vector_size=128, word2vec_params=None):
        self.vector_size = vector_size
        self.word2vec_model = None
        self.vocab = None
        self.word_to_idx = None
        # Default Word2Vec parameters if none provided
        self.word2vec_params = word2vec_params or {
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'epochs': 5
        }
        
    def build_vocab(self, texts):
        print("Preprocessing texts...")
        processed_texts = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            processed_texts.append(simple_preprocess(text))
        
        # Initialize callback for training progress
        callback = Word2VecCallback()
        
        # Train Word2Vec model with progress tracking
        self.word2vec_model = Word2Vec(
            sentences=processed_texts,
            vector_size=self.vector_size,
            **self.word2vec_params,
            callbacks=[callback]
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