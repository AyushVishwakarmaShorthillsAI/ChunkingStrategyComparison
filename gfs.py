import os
import time
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
TOP_K = 5

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Just print warning here, don't raise unless function called
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
                        'max_tokens_per_chunk': 500,
                        'max_overlap_tokens': 50
                    }
                }
            }
        )
        operations.append(op)

    print("⏳ Waiting for ingestion & indexing to complete...")
    for op in operations:
        while not op.done:
            time.sleep(2) # Faster poll
            op = client.operations.get(op)

    print("✅ All PDFs ingested and indexed")
    return store_name

def query_store(store_name, query_text):
    """Runs query against store and returns (answer, chunks_list)"""
    if not client:
        raise RuntimeError("Client not initialized")
        
    print(f"\n🔎 Running query: {query_text}")

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
                ctx = chunk.retrieved_context
                chunks.append({
                    "chunk_id": idx,
                    "title": ctx.title,
                    "content": ctx.text,
                    "score": 0.0 # GFS doesn't expose score directly in this view easily
                })
                
    return answer, chunks

if __name__ == "__main__":
    # Original behavior
    try:
        store_id = setup_store()
        
        QUERY_TEXT = "Which patients have both frontal bossing and behavioral abnormalities according to the table?"
        
        ans, chunks = query_store(store_id, QUERY_TEXT)
        
        print("🧠 MODEL ANSWER:\n")
        print(ans)
        
        print("\n📚 EXTRACTED RETRIEVED CHUNKS:\n")
        for c in chunks:
             print("\n--- Retrieved Chunk ---")
             print(f"PDF title  : {c['title']}")
             print("Chunk text:")
             print(c['content'][:200])
             
    except Exception as e:
        print(e)
