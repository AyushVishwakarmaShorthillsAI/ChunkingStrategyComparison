
import os
import sys
import json
import torch
from pathlib import Path
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
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_chunks")
ENHANCED_CHUNKS_FILE = os.path.join(OUTPUT_FOLDER, "chunks_tables_enhanced.json")
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries_results")
MODEL_NAME = "tencent/Youtu-Embedding"
COLLECTION_NAME = "Youtu_Collection"
QUERY = "What are the common features associated with people suffering from KPTN Syndrome?"

def load_chunks(json_file):
    print(f"📂 Loading chunks from {json_file}...")
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('chunks', [])

def ingest_data_youtu(chunks, model_name, collection_name):
    print(f"\n🚀 Starting Ingestion Process for model: {model_name}")
    
    if not chunks:
        print("❌ No chunks to ingest.")
        return False

    # Initialize Weaviate with a specific collection name
    wm = WeaviateManager(collection_name=collection_name)
    if not wm.client:
        print("❌ Weaviate connection failed.")
        return False

    try:
        # Create Schema (Force reset)
        wm.create_schema(force_reset=True)

        # Initialize Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Loading model {model_name} on {device}...")
        embed_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        # Generate Embeddings
        print("✍️ Generating embeddings for chunks...")
        texts = [c['content'] for c in chunks]
        embeddings = embed_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        
        # Ingest into Weaviate
        wm.ingest_chunks(chunks, embeddings)
        print("✅ Ingestion complete.")
        return True
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False
    finally:
        wm.close()

def query_rag_youtu(query, model_name, collection_name):
    print(f"\n🔍 Querying RAG with: '{query}'")
    
    wm = WeaviateManager(collection_name=collection_name)
    if not wm.client:
        print("❌ Weaviate connection failed.")
        return None

    try:
        # Load Model again for query embedding
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embed_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        print("🧠 Encoding query...")
        query_vector = embed_model.encode(query, convert_to_tensor=False)
        
        # Search
        results = wm.search(query_text=query, query_vector=query_vector, limit=10, alpha=0.5)
        return results
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return None
    finally:
        wm.close()

def save_results(query, results, model_name):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filename = f"query_youtu_results.json"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    output_data = {
        "query": query,
        "model": model_name,
        "results": results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n💾 Results saved to {filepath}")
    return filepath

if __name__ == "__main__":
    # 1. Load Data
    chunks = load_chunks(ENHANCED_CHUNKS_FILE)
    
    if chunks:
        # 2. Ingest with Youtu embeddings
        success = ingest_data_youtu(chunks, MODEL_NAME, COLLECTION_NAME)
        
        if success:
            # 3. Query
            results = query_rag_youtu(QUERY, MODEL_NAME, COLLECTION_NAME)
            
            if results:
                # 4. Save results
                save_results(QUERY, results, MODEL_NAME)
            else:
                print("❌ No results found.")
    else:
        print("❌ Failed to load chunks.")
