import torch

def search(model, processor, query, documents, top_k=5, device='cuda'):
    """
    Search for the most similar documents to a query.
    
    Args:
        model: Trained TwoTowerModel
        processor: TextProcessor instance used during training
        query: String query to search for
        documents: List of documents to search through
        top_k: Number of top results to return
        device: Device to run the model on
        
    Returns:
        List of tuples (document, similarity_score) for top k matches
    """
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