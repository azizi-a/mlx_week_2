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

def train_epoch(model, train_loader, train_doc_embeddings, optimizer, config, device):
    """Run one epoch of training"""
    model.train()
    total_train_loss = 0
    train_pbar = tqdm(train_loader, desc=f'Training')
    
    for batch_idx, (queries, pos_indices) in enumerate(train_pbar):
        optimizer.zero_grad()
        
        queries = queries.to(device)
        pos_indices = pos_indices.to(device)
        query_embeddings = model.get_embeddings(
            queries,
            model_type='query'
        )
        
        # Get positive document embeddings using indices
        positive_doc_embeddings = train_doc_embeddings[pos_indices]
        
        # Create negative examples for each query using config parameter
        batch_size = queries.size(0)
        neg_indices = torch.randint(
            0, 
            len(train_doc_embeddings), 
            (batch_size * config['num_negative_examples'],), 
            device=device
        )
        negative_doc_embeddings = train_doc_embeddings[neg_indices]
        
        # Reshape query and positive embeddings to match negative samples
        query_embeddings = query_embeddings.repeat_interleave(config['num_negative_examples'], dim=0)
        positive_doc_embeddings = positive_doc_embeddings.repeat_interleave(config['num_negative_examples'], dim=0)
        
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
    
    return total_train_loss / len(train_loader)

def validate(model, val_loader, val_doc_embeddings, config, device):
    """Run validation"""
    model.eval()
    total_val_loss = 0
    val_pbar = tqdm(val_loader, desc=f'Validation')
    
    with torch.no_grad():
        for batch_idx, (queries, pos_indices) in enumerate(val_pbar):
            queries = queries.to(device)
            pos_indices = pos_indices.to(device)
            query_embeddings = model.get_embeddings(
                queries,
                model_type='query'
            )
            
            positive_doc_embeddings = val_doc_embeddings[pos_indices]
            
            # Create negative examples for each query using config parameter
            batch_size = queries.size(0)
            neg_indices = torch.randint(
                0, 
                len(val_doc_embeddings), 
                (batch_size * config['num_negative_examples'],), 
                device=device
            )
            negative_doc_embeddings = val_doc_embeddings[neg_indices]
            
            # Reshape query and positive embeddings to match negative samples
            query_embeddings = query_embeddings.repeat_interleave(config['num_negative_examples'], dim=0)
            positive_doc_embeddings = positive_doc_embeddings.repeat_interleave(config['num_negative_examples'], dim=0)
            
            loss = triplet_loss_function(
                query_embeddings,
                positive_doc_embeddings, 
                negative_doc_embeddings,
                config['margin']
            )
            total_val_loss += loss.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_val_loss / len(val_loader)

def train_model(train_data, val_data, config, device='cuda'):
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
    
    # Initialize model with pretrained embeddings
    model = TwoTowerModel(
        vocab_size=processor.vocab_size(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    # Pre-compute all document embeddings for both training and validation sets
    model.eval()
    with torch.no_grad():
        train_doc_embeddings = model.get_embeddings(
            train_doc_sequences.to(device),
            model_type='document'
        )
        val_doc_embeddings = model.get_embeddings(
            val_doc_sequences.to(device),
            model_type='document'
        )
    
    # Create dataloaders for queries only
    train_dataset = torch.utils.data.TensorDataset(
        train_query_sequences,
        torch.arange(len(train_query_sequences))  # indices for positive docs
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(
        val_query_sequences,
        torch.arange(len(val_query_sequences))
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(config['epochs']):
        # Training phase
        avg_train_loss = train_epoch(model, train_loader, train_doc_embeddings, optimizer, config, device)
        
        # Validation phase
        avg_val_loss = validate(model, val_loader, val_doc_embeddings, config, device)
        
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
