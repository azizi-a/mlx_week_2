import torch
from train import train_model, search
import json
from pathlib import Path

def load_sample_data(data_path):
    """Load sample documents and queries from JSON file"""
    with open(data_path) as f:
        data = json.load(f)
    return data['documents'], data['queries'], data['labels']

def main():
    # Configuration
    config = {
        'max_length': 200,
        'embed_dim': 128,
        'hidden_dim': 64,
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    }

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Sample data (alternatively, load from JSON file)
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning models require significant computational resources",
        "Natural language processing helps computers understand human language",
        "Python is a popular programming language for data science"
    ]
    
    queries = [
        "animal behavior",
        "artificial intelligence",
        "computational requirements",
        "language understanding",
        "programming languages"
    ]
    
    # 1 indicates query matches document, 0 indicates no match
    labels = [
        [1, 0, 0, 0, 0],  # query 1 matches doc 1
        [0, 1, 0, 0, 0],  # query 2 matches doc 2
        [0, 0, 1, 0, 0],  # query 3 matches doc 3
        [0, 0, 0, 1, 0],  # query 4 matches doc 4
        [0, 0, 0, 0, 1],  # query 5 matches doc 5
    ]
    
    # Flatten labels for training
    flat_documents = []
    flat_queries = []
    flat_labels = []
    
    for i, query in enumerate(queries):
        for j, doc in enumerate(documents):
            flat_queries.append(query)
            flat_documents.append(doc)
            flat_labels.append(labels[i][j])

    print("Training model...")
    model, processor = train_model(flat_documents, flat_queries, flat_labels, config, device)

    # Save model and processor
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "two_tower_model.pt")
    torch.save(processor, save_dir / "text_processor.pt")
    print(f"Model saved to {save_dir}")

    # Demonstrate search functionality
    print("\nTesting search functionality:")
    test_queries = [
        "How do animals behave?",
        "Tell me about AI and ML",
        "What programming language should I learn?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(model, processor, query, documents, top_k=2, device=device)
        print("Top 2 matching documents:")
        for doc, score in results:
            print(f"Score: {score:.4f} | Document: {doc}")

if __name__ == "__main__":
    main() 