import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# =========================
# 🔧 CONFIG (TOGGLES)
# =========================

INPUT_PDF_DIR = "input_pdfs"
SYNDROME_NAME = "KPTN"
FILE_SEARCH_STORE_DISPLAY_NAME = f"{SYNDROME_NAME}_store"
RESULTS_FOLDER = "queries_results"
TOP_K = 8

# Hardcoded Queries
QUERIES = [
    "Common features in people suffering from KPTN Syndrome and their percentage",
    "List of the most common facial and developmental features exhibited by individuals with KPTN Syndrome with percentages",
    "Uncommon features in people with KPTN Syndrome and their percentage"
]

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️ GOOGLE_API_KEY missing")

client = None
if api_key:
    client = genai.Client(api_key=api_key)

def setup_store():
    """Creates store and ingests PDFs"""
    if not client:
        raise RuntimeError("Client not initialized")
        
    print("📦 Creating File Search Store")

    file_search_store = client.file_search_stores.create(
        config={
            "display_name": FILE_SEARCH_STORE_DISPLAY_NAME
        }
    )

    store_name = file_search_store.name
    print(f"✅ Store created: {store_name}")

    # Ingest PDFs
    pdf_paths = list(Path(INPUT_PDF_DIR).glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError("❌ No PDFs found in input_pdfs/")

    print(f"📄 Found {len(pdf_paths)} PDF(s)")

    operations = []

    for pdf in pdf_paths:
        print(f"⬆️ Ingesting: {pdf.name}")

        op = client.file_search_stores.upload_to_file_search_store(
            file=str(pdf),
            file_search_store_name=store_name,
            config={
                "display_name": pdf.name,
                'chunking_config': {
                    'white_space_config': {
                        'max_tokens_per_chunk': 400,
                        'max_overlap_tokens': 50
                    }
                }
            }
        )
        operations.append(op)

    print("⏳ Waiting for ingestion & indexing to complete...")
    for op in operations:
        retries = 5
        while not op.done and retries > 0:
            try:
                time.sleep(2) 
                op = client.operations.get(op)
            except Exception as e:
                print(f"⚠️ Polling error: {e}. Retrying... ({retries} left)")
                retries -= 1
                time.sleep(5)

    print("✅ All PDFs ingested and indexed")
    return store_name

def query_store(store_name, query_text):
    """Runs query against store and returns (answer, chunks_list)"""
    if not client:
        raise RuntimeError("Client not initialized")
        
    print(f"\n🔎 Running query: {query_text}")

    # Note: TOP_K for GFS retrieval is controlled by the model/tool configuration if exposed, 
    # but here we are using standard generate_content which retrieves chunks automatically.
    # We will slice the results to match TOP_K in the output.
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query_text,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ]
        )
    )
    
    answer = response.text
    chunks = []
    
    # Extract chunks
    if response.candidates and response.candidates[0].grounding_metadata:
        grounding = response.candidates[0].grounding_metadata
        if grounding.grounding_chunks:
            for idx, chunk in enumerate(grounding.grounding_chunks):
                if idx >= TOP_K:
                    break
                ctx = chunk.retrieved_context
                chunks.append({
                    "content": ctx.text,
                    "title": ctx.title,
                    "score": 0.0 # GFS doesn't expose score directly in this view easily
                })
                
    return answer, chunks

def save_all_results(data, model_name):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    filename = "gfs_multi_query_results.json"
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
    try:
        store_id = setup_store()
        
        all_query_results = []
        
        for query in QUERIES:
            ans, chunks = query_store(store_id, query)
            
            all_query_results.append({
                "query": query,
                "answer": ans,
                "results": chunks
            })
        
        save_all_results(all_query_results, "Gemini-File-Search")
             
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()