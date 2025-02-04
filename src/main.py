import torch
from train import train_model
from inference import search
from data.data_loader import load_sample_data, flatten_queries_and_documents
from pathlib import Path

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

    # Load sample data
    training_dataset, _validation_dataset, _test_dataset = load_sample_data(100)
    queries, documents, labels = flatten_queries_and_documents(training_dataset)
    
    print(f"Number of documents: {len(documents)}")
    print(f"Number of queries: {len(queries)}")
    print(f"Number of labels: {len(labels)}")

    print("Training model...")
    model, processor = train_model(queries, documents, labels, config, device)

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
            doc_preview = ' '.join(doc.split()[:80])
            print(f"Score: {score:.4f} | Document: {doc_preview}...")

if __name__ == "__main__":
    main() 