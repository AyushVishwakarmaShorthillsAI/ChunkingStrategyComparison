import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
import sys
load_dotenv()

# Force UTF-8 for printing
sys.stdout.reconfigure(encoding='utf-8')

# Initialize Client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️ GOOGLE_API_KEY missing")
    exit(1)

client = genai.Client(api_key=api_key)

def check_metadata():
    """Create a temporary store, upload one sample PDF, and print raw metadata."""
    print("🚀 Starting Metadata Investigation...")

    # 1. Create a temporary store
    try:
        store = client.file_search_stores.create(
            config={"display_name": "metadata_test_store"}
        )
        store_name = store.name
        print(f"✅ Store created: {store_name}")

        # 2. Upload a sample PDF (Horn.pdf exists in input_pdfs/)
        pdf_path = "input_pdfs/Horn.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ Sample PDF not found at {pdf_path}")
            return

        print(f"⬆️ Ingesting: {pdf_path}...")
        op = client.file_search_stores.upload_to_file_search_store(
            file=pdf_path,
            file_search_store_name=store_name
        )

        while not op.done:
            print("⏳ Indexing...", end="\r")
            time.sleep(2)
            op = client.operations.get(op)
        print("\n✅ Sample PDF indexed.")

        # 3. Query the store
        query = "What are the common features of KPTN Syndrome?"
        print(f"🔎 Querying: '{query}'")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
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

        # 4. Save and Print the RAW Grounding Metadata
        print("\n" + "="*50)
        print("📋 SAVING RAW GROUNDING METADATA")
        print("="*50)
        
        if response.candidates and response.candidates[0].grounding_metadata:
            # Convert to dict for JSON serialization
            meta_dict = response.candidates[0].grounding_metadata.model_dump()
            with open("raw_gfs_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=4)
            print("✅ Raw metadata saved to raw_gfs_metadata.json")
            
            # Helper: Accessing chunks specifically
            print("\n" + "="*50)
            print("📄 SPECIFIC CHUNKS RETRIEVED")
            print("="*50)
            grounding = response.candidates[0].grounding_metadata
            if grounding.grounding_chunks:
                for i, chunk in enumerate(grounding.grounding_chunks):
                    print(f"\n--- Chunk {i} ---")
                    # Explicitly showing available attributes
                    if chunk.retrieved_context:
                        ctx = chunk.retrieved_context
                        print(f"Title: {ctx.title}")
                        print(f"Text Preview: {ctx.text[:100]}...")
                        # If there were page numbers, they would likely be in a 'uri' or nested field
                        print(f"URI: {ctx.uri if hasattr(ctx, 'uri') else 'N/A'}")
        else:
            print("❌ No grounding metadata found in response.")

    except Exception as e:
        print(f"❌ Error during investigation: {e}")

if __name__ == "__main__":
    check_metadata()
