import torch
from torch.utils.data import DataLoader
from models.two_tower_model import TwoTowerModel
from utils.text_processor import TextProcessor
from data.data_loader import TextMatchingDataset

def triplet_loss_function(query_embeddings, positive_doc_embeddings, negative_doc_embeddings, margin):
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
    positive_case_distance = 1 - positive_similarity 
    negative_case_distance = 1 - negative_similarity
    
    # Compute triplet loss
    loss = torch.clamp(positive_case_distance - negative_case_distance + margin, min=0)
    
    return loss.mean()

def train_model(train_data,
                val_data,
                config, device='cuda'):
    # Initialize processor and process data
    processor = TextProcessor(max_length=config['max_length'])
    processor.fit(train_data['documents'] + train_data['queries'])
    
    # Process training and validation data
    train_doc_sequences = processor.transform(train_data['documents'])
    train_query_sequences = processor.transform(train_data['queries'])
    val_doc_sequences = processor.transform(val_data['documents'])
    val_query_sequences = processor.transform(val_data['queries'])
    
    # Create dataloaders
    train_dataset = TextMatchingDataset(train_doc_sequences, train_query_sequences, torch.tensor(train_data['labels']))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    val_dataset = TextMatchingDataset(val_doc_sequences, val_query_sequences, torch.tensor(val_data['labels']))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = TwoTowerModel(
        vocab_size=processor.vocab_size(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_loader:
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
            shift_amount = torch.randint(10, 10000, (1,)).item()
            negative_docs = batch['document'].roll(shifts=shift_amount, dims=0)
            negative_doc_embeddings = model.get_embeddings(
                negative_docs.to(device),
                model_type='document'
            )

            # Calculate triplet loss
            loss = triplet_loss_function(
                query_embeddings,
                positive_doc_embeddings, 
                negative_doc_embeddings,
                config['margin']
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                query_embeddings = model.get_embeddings(
                    batch['query'].to(device),
                    model_type='query'
                )
                positive_doc_embeddings = model.get_embeddings(
                    batch['document'].to(device), 
                    model_type='document'
                )
                
                # Create negative examples for validation
                shift_amount = torch.randint(10, 10000, (1,)).item()
                negative_docs = batch['document'].roll(shifts=shift_amount, dims=0)
                negative_doc_embeddings = model.get_embeddings(
                    negative_docs.to(device),
                    model_type='document'
                )
                
                loss = triplet_loss_function(
                    query_embeddings,
                    positive_doc_embeddings, 
                    negative_doc_embeddings,
                    config['margin']
                )
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model, processor
