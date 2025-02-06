import torch
from datasets import load_dataset

class TextMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, documents, queries, labels):
        self.documents = documents
        self.queries = queries
        self.labels = labels
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return {
            'document': self.documents[idx],
            'query': self.queries[idx],
            'label': self.labels[idx]
        } 

def load_sample_data(sample_size, cache_dir="src/data/ms_marco_cache"):
    """Load sample data from MS MARCO dataset with caching.
    
    Args:
        sample_size: Number of samples to load from each split
        cache_dir: Directory to store cached dataset
    """
    # Load MS MARCO dataset with caching enabled
    ds = load_dataset("microsoft/ms_marco", "v1.1", cache_dir=cache_dir)
    
    train_data = ds['train']
    validation_data = ds['validation']
    test_data = ds['test']
    
    # Return data sets
    return train_data[:sample_size], validation_data[:sample_size], test_data[:sample_size] 

def flatten_queries_and_documents(dataset):
    """
    Flattens queries and documents from a MS MARCO dataset into simple lists.
    
    Args:
        dataset: MS MARCO dataset split containing queries and passages
        
    Returns:
        queries: List of query strings
        documents: List of document strings
        labels: List of binary labels indicating if document is the best match for query
    """
    flat_queries = []
    flat_documents = []
    flat_labels = []
    
    for i, query in enumerate(dataset['query']):
        passages = dataset['passages'][i]['passage_text']
        is_selected = dataset['passages'][i]['is_selected']
        
        # Add each passage as a separate document
        for doc, label in zip(passages, is_selected):
            flat_queries.append(query)
            flat_documents.append(doc)
            flat_labels.append(label)
            
    return {
        'queries': flat_queries,
        'documents': flat_documents,
        'labels': flat_labels
    }
