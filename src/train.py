import torch
import wandb
from torch.utils.data import DataLoader
from models.two_tower_model import TwoTowerModel
from utils.text_processor import TextProcessor
from data.data_loader import TextMatchingDataset
from tqdm import tqdm

def triplet_loss_function(query_embedding, positive_doc_embedding, negative_doc_embedding, margin):
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
    # Normalize embeddings to ensure cosine similarity calculations are correct
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    positive_doc_embedding = torch.nn.functional.normalize(positive_doc_embedding, p=2, dim=1)
    negative_doc_embedding = torch.nn.functional.normalize(negative_doc_embedding, p=2, dim=1)
    
    # Calculate cosine similarities
    positive_similarity = torch.cosine_similarity(query_embedding, positive_doc_embedding)
    negative_similarity = torch.cosine_similarity(query_embedding, negative_doc_embedding)
    
    # Calculate loss
    loss = torch.relu(positive_similarity - negative_similarity + margin)
    
    return loss.mean()

def train_model(train_data,
                val_data,
                config, device='cuda'):
    # Initialize processor and process data
    processor = TextProcessor(vector_size=config['embed_dim'], word2vec_params=config['word2vec'])
    processor.build_vocab(train_data['documents'] + train_data['queries'])
    
    # Get pretrained embeddings
    pretrained_embeddings = processor.get_embedding_weights()
    
    # Process training and validation data
    train_doc_sequences = processor.encode_text(train_data['documents'])
    train_query_sequences = processor.encode_text(train_data['queries'])
    val_doc_sequences = processor.encode_text(val_data['documents'])
    val_query_sequences = processor.encode_text(val_data['queries'])
    
    # Create dataloaders
    train_dataset = TextMatchingDataset(train_doc_sequences, train_query_sequences, torch.tensor(train_data['labels']))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    val_dataset = TextMatchingDataset(val_doc_sequences, val_query_sequences, torch.tensor(val_data['labels']))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model with pretrained embeddings
    model = TwoTowerModel(
        vocab_size=processor.vocab_size(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{config["epochs"]}')
        
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            # Get embeddings for queries and positive documents
            queries = batch['query'].to(device)
            query_embeddings = model.get_embeddings(
                queries,
                model_type='query'
            )

            positive_docs = batch['document'].to(device)
            positive_doc_embeddings = model.get_embeddings(
                positive_docs, 
                model_type='document'
            )
            
            # Create negative examples by randomly permuting the document indices
            batch_size = queries.size(0)
            perm = torch.randperm(batch_size)
            negative_docs = positive_docs[perm]
            negative_doc_embeddings = model.get_embeddings(
                negative_docs,
                model_type='document'
            )

            # Calculate triplet loss
            loss = triplet_loss_function(
                query_embeddings,
                positive_doc_embeddings, 
                negative_doc_embeddings,
                config['margin']
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch loss to wandb
            wandb.log({"batch_train_loss": loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{config["epochs"]}')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                queries = batch['query'].to(device)
                query_embeddings = model.get_embeddings(
                    queries,
                    model_type='query'
                )

                positive_docs = batch['document'].to(device)
                positive_doc_embeddings = model.get_embeddings(
                    positive_docs, 
                    model_type='document'
                )
                
                batch_size = queries.size(0)
                perm = torch.randperm(batch_size)
                negative_docs = positive_docs[perm]
                negative_doc_embeddings = model.get_embeddings(
                    negative_docs,
                    model_type='document'
                )

                loss = triplet_loss_function(
                    query_embeddings,
                    positive_doc_embeddings, 
                    negative_doc_embeddings,
                    config['margin']
                )
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        print(f"\nEpoch {epoch+1}/{config['epochs']} Summary - "
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
