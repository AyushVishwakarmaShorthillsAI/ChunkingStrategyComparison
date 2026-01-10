
import sys
import os
import torch
from WeaviateManager import WeaviateManager
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configure Google GenAI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = None

if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    print("⚠️  GOOGLE_API_KEY not set. LLM synthesis will be skipped.")

def generate_answer(query, results):
    if not client:
        return
        
    print(f"\n🧠 Generating answer using Gemini 2.5 Flash...")
    
    # Context Construction
    context_parts = []
    for i, res in enumerate(results, 1):
        context_parts.append(f"Source {i} ({res['source']}, Page {res['page']}):\n{res['content']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant. Answer the user's query based ONLY on the provided context.
    
Query: {query}

Context:
{context}

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        print("\n🤖 Gemini Answer:")
        print("-" * 40)
        print(response.text)
        print("-" * 40)
        return response.text
        
    except Exception as e:
        print(f"❌ Failed to generate answer: {e}")
        return ""

def retrieve_from_weaviate(query_text, top_k=5, alpha=0.5):

    print(f"🚀 Starting Weaviate Retrieval (Alpha: {alpha})...")
    
    # 1. Connect to Weaviate
    wm = WeaviateManager()
    if not wm.client:
        print("❌ Weaviate connection failed.")
        sys.exit(1)
        
    # 2. Embed Query (using same model as ingestion)
    # Note: Weaviate's text2vec-transformers module could do this if configured,
    # but we are doing client-side vectorization to match nrag.py's ingestion.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Embedder loaded on {device}. Embedding query: '{query_text}'")
    
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    query_vector = embed_model.encode(query_text, convert_to_tensor=False).tolist()
    
    # 3. Search
    results = wm.search(
        query_text=query_text,
        limit=top_k,
        query_vector=query_vector,
        alpha=alpha
    )
    
    if not results:
        print("❌ No results found.")
    else:
        print(f"\n🏆 Top {len(results)} Results from Weaviate:")
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {res['score']:.4f}) ---")
            print(f"Source: {res['source']} | Page: {res['page']}")
            print(f"Content: {res['content']}...")
            
        # Generate Answer
        return results
        
    wm.close()
    return []

if __name__ == "__main__":
    # Hardcoded query as requested
    QUERY = "What is KPTN syndrome"
    
    if len(sys.argv) > 1:
        QUERY = " ".join(sys.argv[1:])
        
    results = retrieve_from_weaviate(QUERY)
    if results:
         generate_answer(QUERY, results)

