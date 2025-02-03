import torch
import json
from pathlib import Path
# from datasets import load_dataset

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

def load_sample_data(data_path):
    """Load sample data from a JSON file or use the MS MARCO dataset."""
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            data = json.load(f)
            return data['documents'], data['queries'], data['labels']
    else:
        # Load MS MARCO dataset
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        
        # Get a small sample of documents and queries
        documents = ds['train']['passages'][:5]  # First 5 passages
        queries = ds['train']['query'][:5]  # First 5 queries
        
        # Create simple binary labels (1 for positive match, 0 otherwise)
        labels = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        
        return documents, queries, labels

def flatten_data(documents, queries, labels):
    """Flatten the documents, queries, and labels for training."""
    flat_documents = []
    flat_queries = []
    flat_labels = []
    
    for i, query in enumerate(queries):
        for j, doc in enumerate(documents):
            flat_queries.append(query)
            flat_documents.append(doc)
            flat_labels.append(labels[i][j])
            
    return flat_documents, flat_queries, flat_labels 