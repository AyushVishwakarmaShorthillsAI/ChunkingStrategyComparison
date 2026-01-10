
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modified classes
try:
    from DocumentChunker import DocumentChunker
    from TableEnhancer import enhance_chunks_async
    from WeaviateManager import WeaviateManager

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Import retrieval libraries
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    print("Error: sentence-transformers not installed. Please install it.")
    sys.exit(1)

import asyncio

# Configuration
INPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_pdfs")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_chunks")
CHUNKS_FILE = os.path.join(OUTPUT_FOLDER, "chunks.json")
ENHANCED_CHUNKS_FILE = os.path.join(OUTPUT_FOLDER, "chunks_tables_enhanced.json")
TABLE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_images")

def process_documents():
    """Runt chunking and table enhancement pipeline"""
    print(f"🚀 Starting Document Processing Pipeline...")
    
    # 1. Create output directories
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TABLE_IMAGES_DIR, exist_ok=True)
    
    # 2. Chunking
    print(f"\n📦 Step 1: Chunking PDFs from {INPUT_FOLDER}...")
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
        chunking_method="enhanced",
        use_docling=True
    )
    
    # Check if we need to run chunking (if chunks.json exists, maybe we can skip, but let's re-run to be safe or check timestamp?)
    # For this script, we'll run it.
    chunks_data = chunker.generate_chunks_from_folder(
        folder_path=INPUT_FOLDER,
        output_file=CHUNKS_FILE
    )
    
    if not chunks_data:
        print("❌ Chunking failed or no chunks generated.")
        return None
    
    print(f"✅ Chunking complete. saved to {CHUNKS_FILE}")
    
    # 3. Table Enhancement
    print(f"\n✨ Step 2: Enhancing Tables...")
    
    # helper for async run
    async def run_enhancement():
        return await enhance_chunks_async(
            chunks_path=CHUNKS_FILE,
            output_path=ENHANCED_CHUNKS_FILE,
            images_dir=TABLE_IMAGES_DIR,
            search_roots=[INPUT_FOLDER]
        )
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            enhanced_file = loop.run_until_complete(run_enhancement())
        else:
            enhanced_file = asyncio.run(run_enhancement())
            
        print(f"✅ Table enhancement complete. Saved to {enhanced_file}")
        return enhanced_file
    except Exception as e:
        print(f"❌ Table enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return CHUNKS_FILE # Fallback to unenhanced

        return CHUNKS_FILE # Fallback to unenhanced

def ingest_data(json_file):
    print(f"\n💾 Step 3: Ingesting data from {json_file} into Weaviate...")
    
    # Load chunks
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        chunks = data.get('chunks', [])
    
    if not chunks:
        print("❌ No chunks to ingest.")
        return

    # Initialize Weaviate
    wm = WeaviateManager()
    if not wm.client:
        print("❌ Weaviate connection failed. Skipping ingestion.")
        return

    # Create Schema (Force reset to ensure clean state for demo)
    wm.create_schema(force_reset=True)

    # Generate Embeddings locally
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Generating embeddings using all-MiniLM-L6-v2 on {device}...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    
    texts = [c['content'] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    
    # Ingest
    wm.ingest_chunks(chunks, embeddings)
    wm.close()
    
    # Save embeddings locally too (optional, but good for nrag_retrieve.py if it needs them, 
    # though nrag_retrieve will use Weaviate search).
    # actually user said: "keep the nrag.py file till chunking and embedding (ingestion) only"
    # so we have done that.

if __name__ == "__main__":
    # 1. Run Pipeline
    final_chunks_file = process_documents()
    
    if final_chunks_file:
        # 2. Ingest
        ingest_data(final_chunks_file)

