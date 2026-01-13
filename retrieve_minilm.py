
import os
import sys
import json
import torch
from typing import List, Dict, Any

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from WeaviateManager import WeaviateManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed. Please install it.")
    sys.exit(1)

# Configuration
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries_results")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "MiniLM_Collection"

# Hardcoded Queries
QUERIES = [
    "Common features in people suffering from KPTN Syndrome and their percentage",
    "List of the most common facial and developmental features exhibited by individuals with KPTN Syndrome with percentages",
    "Uncommon features in people with KPTN Syndrome and their percentage"
]

def retrieve_for_queries(queries: List[str], model_name: str, collection_name: str):
    print(f"🚀 Starting Retrieval Process for {len(queries)} queries...")
    
    wm = WeaviateManager(collection_name=collection_name)
    if not wm.client:
        print("❌ Weaviate connection failed.")
        return None

    all_results = []

    try:
        # Check if collection exists
        if not wm.client.collections.exists(collection_name):
            print(f"❌ Collection {collection_name} does not exist. Please run rag_minilm.py first.")
            return None

        # Initialize Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Loading model {model_name} on {device}...")
        embed_model = SentenceTransformer(model_name, device=device)
        
        for query in queries:
            print(f"\n🔍 Querying: '{query}'")
            print("🧠 Encoding query...")
            query_vector = embed_model.encode(query, convert_to_tensor=False)
            
            # Search
            results = wm.search(query_text=query, query_vector=query_vector, limit=8, alpha=0.5)
            
            all_results.append({
                "query": query,
                "results": results
            })
            
        return all_results
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return None
    finally:
        wm.close()

def save_all_results(data: List[Dict[str, Any]], model_name: str):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filename = "minilm_multi_query_results.json"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    output_data = {
        "model": model_name,
        "total_queries": len(data),
        "queries": data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n💾 All results saved to {filepath}")
    return filepath

if __name__ == "__main__":
    results = retrieve_for_queries(QUERIES, MODEL_NAME, COLLECTION_NAME)
    
    if results:
        save_all_results(results, MODEL_NAME)
    else:
        print("❌ Retrieval process failed or returned no results.")
