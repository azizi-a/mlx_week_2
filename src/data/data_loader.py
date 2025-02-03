import torch

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