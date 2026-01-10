
import json
import os
import time
import nrag_retrieve
import gfs

QUERIES = [
    "What clinical conclusions do the authors draw about the variability of macrocephaly in KPTN syndrome, and how does this study modify previous assumptions?"
]

def save_result(filename, query, chunks, summary):
    data = {
        "query": query,
        "chunks": chunks,
        "summary": summary
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved {filename}")

def main():
    print("🚀 Starting Retrieval Comparison...")
    
    # 1. Setup GFS (One-time setup)
    print("\n📦 Setting up GFS Store...")
    try:
        gfs_store_id = gfs.setup_store()
    except Exception as e:
        print(f"❌ GFS Setup Failed: {e}")
        return

    # 2. Run Queries
    for i, query in enumerate(QUERIES, 1):
        print(f"\n\n🔍 [Query {i}/{len(QUERIES)}]: {query}")
        
        # --- NRAG Retrieval ---
        print("\n🔵 Running NRAG (Weaviate + Gemini)...")
        try:
            nrag_chunks = nrag_retrieve.retrieve_from_weaviate(query, top_k=5)
            nrag_summary = nrag_retrieve.generate_answer(query, nrag_chunks)
            save_result(f"query_{i}_nrag.json", query, nrag_chunks, nrag_summary)
            print("⏳ Sleeping 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"❌ NRAG Failed for query {i}: {e}")

        # --- GFS Retrieval ---
        print("\n🟣 Running GFS (Gemini File Search)...")
        try:
            gfs_summary, gfs_chunks = gfs.query_store(gfs_store_id, query)
            save_result(f"query_{i}_gfs.json", query, gfs_chunks, gfs_summary)
            print("⏳ Sleeping 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"❌ GFS Failed for query {i}: {e}")

    print("\n✨ Comparison Complete!")

if __name__ == "__main__":
    main()
