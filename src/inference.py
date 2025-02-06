import torch

def search(model, processor, query, documents, top_k=5, device='cuda', batch_size=8192):
    """
    Search for the most similar documents to a query.
    
    Args:
        model: Trained TwoTowerModel
        processor: TextProcessor instance used during training
        query: String query to search for
        documents: List of documents to search through
        top_k: Number of top results to return
        device: Device to run the model on
        batch_size: Number of documents to process at once
        
    Returns:
        List of tuples (document, similarity_score) for top k matches
    """
    model.eval()
    with torch.no_grad():
        # Process query
        query_emb = model.get_embeddings(
            processor.encode_text_with_cache([query], cache_key=f'query_{query}').to(device),
            model_type='query'
        )
        
        # Process documents with caching
        doc_encodings = processor.encode_text_with_cache(
            documents,
            cache_key='search_documents'
        )
        
        # Process in batches
        all_similarities = []
        for i in range(0, len(documents), batch_size):
            batch_docs = doc_encodings[i:i + batch_size].to(device)
            doc_emb = model.get_embeddings(batch_docs, model_type='document')
            batch_similarities = torch.matmul(query_emb, doc_emb.t()).cpu().numpy().flatten()
            all_similarities.extend(batch_similarities)
            
        similarities = torch.tensor(all_similarities)
        top_idx = torch.topk(similarities, k=top_k, largest=True).indices
        return [(documents[i], similarities[i].item()) for i in top_idx]