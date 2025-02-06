import torch
import wandb
from torch.utils.data import DataLoader
from models.two_tower_model import TwoTowerModel
from utils.text_processor import TextProcessor
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
        current_avg_loss = total_train_loss / (batch_idx + 1)  # Calculate running average
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{current_avg_loss:.4f}'})
        
        # Log detailed training metrics
        wandb.log({
            "batch_train_loss": loss.item(),
            "running_avg_train_loss": current_avg_loss
        })
    
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

def compute_doc_embeddings(model, doc_sequences, batch_size, device):
    """Compute document embeddings in batches to avoid memory issues"""
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(doc_sequences), batch_size):
            batch = doc_sequences[i:i + batch_size].to(device)
            batch_embeddings = model.get_embeddings(batch, model_type='document')
            embeddings_list.append(batch_embeddings.cpu())  # Move to CPU to free GPU memory
    
    return torch.cat(embeddings_list, dim=0).to(device)

def check_embedding_diversity(embeddings, threshold=0.9):
    if len(embeddings) > 10_000:
        return True
        
    # Compute pairwise similarities
    similarities = torch.matmul(embeddings, embeddings.t())
    # Get upper triangle without diagonal
    upper_tri = torch.triu(similarities, diagonal=1)
    # Count pairs that are too similar
    too_similar_count = (upper_tri > threshold).sum().item()
    total_pairs = (len(embeddings) * (len(embeddings) - 1)) // 2
    
    if too_similar_count > 0:
        print(f"Warning: {too_similar_count}/{total_pairs} of document embedding pairs are too similar (>{threshold})")
        return False
    return True
def train_model(train_data, val_data, config, device='cuda'):
    # Initialize processor and process data
    processor = TextProcessor(vector_size=config['embed_dim'], word2vec_params=config['word2vec'])
    
    # Build vocabulary
    print("Building vocabulary...")
    processor.build_vocab(train_data['documents'] + train_data['queries'])
    
    # Get pretrained embeddings
    pretrained_embeddings = processor.get_embedding_weights()
    
    # Process training and validation data with caching
    print("Processing training data...")
    train_doc_sequences = processor.encode_text(train_data['documents'])
    train_query_sequences = processor.encode_text(train_data['queries'])
    
    print("Processing validation data...")
    val_doc_sequences = processor.encode_text(val_data['documents'])
    val_query_sequences = processor.encode_text(val_data['queries'])
    
    # Initialize model with pretrained embeddings
    model = TwoTowerModel(
        vocab_size=processor.vocab_size(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    # Pre-compute all document embeddings for both training and validation sets in batches
    print("Computing document embeddings...")
    train_doc_embeddings = compute_doc_embeddings(
        model, 
        train_doc_sequences, 
        config['batch_size'],
        device
    )
    
    print(f"\n\ntrain_doc_embeddings: {train_doc_embeddings.shape}")
    check_embedding_diversity(train_doc_embeddings)
    print("---------------------------------------------------------\n")

    val_doc_embeddings = compute_doc_embeddings(
        model, 
        val_doc_sequences, 
        config['batch_size'],
        device
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
        
        # Log epoch metrics together to wandb
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": avg_val_loss,
            "epoch_loss_difference (train - val)": avg_train_loss - avg_val_loss
        }, commit=True)
        
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
