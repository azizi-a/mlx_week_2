from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from pathlib import Path
import sys
from datetime import datetime
from src.data.data_loader import load_sample_data, flatten_queries_and_documents
from contextlib import asynccontextmanager


# Add src directory to path to import local modules
sys.path.append("src")
from inference import search
from models.two_tower_model import TwoTowerModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, processor, documents
    try:
        # Load saved model and processor
        save_dir = Path("saved_models")
        processor = torch.load(save_dir / "text_processor.pt", weights_only=False)
        
        # Initialize and load model
        model = TwoTowerModel(
            vocab_size=processor.vocab_size(),
            embed_dim=128,
            hidden_dim=64
        ).to(device)
        model.load_state_dict(torch.load(save_dir / "two_tower_model.pt"))
        model.eval()
        
        # Load sample documents for searching
        training_dataset, _, _ = load_sample_data(150_000)
        documents = flatten_queries_and_documents(training_dataset)['documents']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    yield
    # Shutdown
    # Add cleanup code here if needed

app = FastAPI(title="Search API", lifespan=lifespan)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Global variables
model = None
processor = None
documents = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = datetime.now()

@app.get("/ping")
async def ping():
    return {
        "status": "healthy",
        "uptime": str(datetime.now() - start_time),
        "device": str(device)
    }

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    if not all([model, processor, documents]):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = search(
            model=model,
            processor=processor,
            query=request.query,
            documents=documents,
            top_k=request.top_k,
            device=device
        )
        
        return {
            "results": [
                {"document": doc, "score": float(score)}
                for doc, score in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
