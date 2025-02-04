import torch
from torch.utils.data import DataLoader
from models.two_tower_model import TwoTowerModel
from utils.text_processor import TextProcessor
from data.data_loader import TextMatchingDataset

def triplet_loss_function(query_embeddings, positive_doc_embeddings, negative_doc_embeddings, margin=1.0):
    """
    Compute triplet loss between query, positive and negative document embeddings.
    
    Args:
        query_embeddings: Tensor of query embeddings
        positive_doc_embeddings: Tensor of positive document embeddings 
        negative_doc_embeddings: Tensor of negative document embeddings
        margin: Minimum margin between positive and negative distances
        
    Returns:
        loss: Triplet loss value
    """
    # Calculate cosine similarities
    positive_similarity = torch.cosine_similarity(query_embeddings, positive_doc_embeddings)
    negative_similarity = torch.cosine_similarity(query_embeddings, negative_doc_embeddings)
    
    # Convert similarities to distances (1 - similarity)
    positive_distance = 1 - positive_similarity 
    negative_distance = 1 - negative_similarity
    
    # Compute triplet loss
    loss = torch.clamp(positive_distance - negative_distance + margin, min=0)
    
    return loss.mean()

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
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            # Get embeddings for queries and positive documents
            query_embeddings = model.get_embeddings(
                batch['query'].to(device),
                model_type='query'
            )
            positive_doc_embeddings = model.get_embeddings(
                batch['document'].to(device), 
                model_type='document'
            )
            
            # Create negative examples by shuffling the documents
            negative_docs = batch['document'].roll(shifts=1, dims=0)
            negative_doc_embeddings = model.get_embeddings(
                negative_docs.to(device),
                model_type='document'
            )

            # Calculate triplet loss
            loss = triplet_loss_function(
                query_embeddings,
                positive_doc_embeddings, 
                negative_doc_embeddings
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model, processor
