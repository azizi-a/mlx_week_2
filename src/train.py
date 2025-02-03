import torch
from torch.utils.data import DataLoader
from models.two_tower_model import TwoTowerModel
from utils.text_processor import TextProcessor
from data.data_loader import TextMatchingDataset

def train_model(documents, queries, labels, config, device='cuda'):
    # Initialize processor and process data
    processor = TextProcessor(max_length=config['max_length'])
    processor.fit(documents + queries)
    doc_sequences = processor.transform(documents)
    query_sequences = processor.transform(queries)
    
    # Create dataloader
    dataset = TextMatchingDataset(doc_sequences, query_sequences, torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model and training components
    model = TwoTowerModel(
        vocab_size=processor.vocab_size(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            similarity = model(
                batch['document'].to(device),
                batch['query'].to(device)
            )
            # Reshape the label tensor to match similarity matrix dimensions
            batch_labels = batch['label'].float().view(-1, 1).to(device)
            # Only take the diagonal elements of the similarity matrix
            similarity = torch.diagonal(similarity).view(-1, 1)
            loss = criterion(similarity, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model, processor

def search(model, processor, query, documents, top_k=5, device='cuda'):
    model.eval()
    with torch.no_grad():
        query_emb = model.get_embeddings(
            processor.transform([query]).to(device),
            is_query=True
        )
        doc_emb = model.get_embeddings(
            processor.transform(documents).to(device),
            is_query=False
        )
        similarities = torch.matmul(query_emb, doc_emb.t()).cpu().numpy().flatten()
        top_idx = similarities.argsort()[-top_k:][::-1]
        return [(documents[i], similarities[i]) for i in top_idx] 