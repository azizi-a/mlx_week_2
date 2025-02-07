import torch
import wandb
from train_model import train_model
from inference import search
from data.data_loader import load_sample_data, flatten_queries_and_documents
from pathlib import Path
from utils.text_processor import TextProcessor

def main():
    # Configuration
    config = {
        'embed_dim': 128,
        'hidden_dim': 64,
        'batch_size': 512,
        'epochs': 10,
        'learning_rate': 0.001,
        'margin': 0.3,
        'num_negative_examples': 3,
        'sample_size': 150_000,
        'top_k': 5,
        'num_workers': 4,
        # Word2Vec configuration
        'word2vec': {
            'window': 5,
            'min_count': 5,
            'workers': 4,
            'epochs': 5
        }
    }

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load sample data
    training_dataset, validation_dataset, _test_dataset = load_sample_data(config['sample_size'])
    train_data = flatten_queries_and_documents(training_dataset)
    val_data = flatten_queries_and_documents(validation_dataset)
    
    print(f"Number of training documents: {len(train_data['documents'])}")
    print(f"Number of training queries: {len(train_data['queries'])}")
    print(f"Number of validation documents: {len(val_data['documents'])}")

    print("Training model...")
    wandb.init(
        project="two_tower_search",
        config=config
    )
    processor = TextProcessor(vector_size=config['embed_dim'], word2vec_params=config['word2vec'])
    processor.build_vocab(train_data['documents'] + train_data['queries'], use_text8=True)
    model, processor = train_model( 
        train_data,
        val_data,
        config, device
    )

    # Save model to wandb
    print('Saving...')
    save_dir = Path("saved_models")
    model_path = save_dir / 'weights.pt'
    torch.save(model.state_dict(), model_path)
    print('Uploading...')
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print('Done!')
    wandb.finish()

    # Save model and processor to local directory
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "two_tower_model.pt")
    torch.save(processor, save_dir / "text_processor.pt")
    print(f"Model saved to {save_dir}")

    # Demonstrate search functionality
    print("\nTesting search functionality:")
    test_queries = [
        "How do animals behave?",
        "Tell me about AI and ML", 
        "What programming language should I learn?",
        train_data['queries'][0],
        train_data['queries'][5],
        train_data['queries'][47]
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(model, processor, query, train_data['documents'], top_k=config['top_k'], device=device)
        print(f"Top {config['top_k']} matching documents:")
        for doc, score in results:
            doc_preview = ' '.join(doc.split()[:20])
            print('-'*100)
            print(f"Score: {score:.4f} | Document: {doc_preview}...")

if __name__ == "__main__":
    main() 