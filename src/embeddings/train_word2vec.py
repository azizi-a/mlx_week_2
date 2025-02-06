import torch
from pathlib import Path
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from data.get_text8 import download_text8

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

def train_word2vec(texts, vector_size=128, word2vec_params=None, use_text8=True, cache_dir="saved_models"):
    """Train Word2Vec model and cache the results."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"word2vec_{vector_size}.pt"
    
    # Check if cached model exists
    if cache_path.exists():
        print("Loading cached Word2Vec model...")
        return torch.load(cache_path, weights_only=False)
    
    print("Training new Word2Vec model...")
    processed_texts = []
    
    # Process input texts
    for text in tqdm(texts, desc="Tokenizing input texts"):
        processed_texts.append(simple_preprocess(text))
    
    # Add text8 data if requested
    if use_text8:
        print("Loading text8 dataset...")
        text8_data = download_text8()
        
        # Process text8 in chunks
        chunk_size = 1000000  # Process 1M characters at a time
        for i in tqdm(range(0, len(text8_data), chunk_size), desc="Processing text8 chunks"):
            chunk = text8_data[i:i + chunk_size]
            words = simple_preprocess(chunk)
            if words:
                processed_texts.append(words)
    
    # Default Word2Vec parameters
    default_params = {
        'window': 5,
        'min_count': 1,
        'workers': 4,
        'epochs': 5
    }
    
    # Update with provided parameters
    if word2vec_params:
        default_params.update(word2vec_params)
    
    # Initialize callback
    callback = Word2VecCallback()
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=processed_texts,
        vector_size=vector_size,
        **default_params,
        callbacks=[callback]
    )
    
    # Save model
    torch.save(model, cache_path)
    print(f"Model saved to {cache_path}")
    
    return model 