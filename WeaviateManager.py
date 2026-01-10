
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import MetadataQuery
import os
import json
import traceback

class WeaviateManager:
    def __init__(self, collection_name="ChunkingStrtComp"):
        self.collection_name = collection_name
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to local embedded Weaviate unless WEAVIATE_URL is set"""
        try:
            print("🔌 Connecting to Weaviate...")
            # Check if environment variables are set for a remote instance
            url = os.getenv("WEAVIATE_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            if url and api_key:
                 self.client = weaviate.connect_to_wcs(
                    cluster_url=url,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key)
                )
                 print(f"✅ Connected to Remote Weaviate at {url}")
            else:
                # Use embedded Weaviate for local development
                self.client = weaviate.connect_to_local()
                print("✅ Connected to Local Weaviate")
                
        except Exception as e:
            print(f"❌ Failed to connect to Weaviate: {e}")
            traceback.print_exc()
            self.client = None

    def close(self):
        if self.client:
            self.client.close()

    def create_schema(self, force_reset=False):
        """Create the collection schema"""
        if not self.client:
            return

        try:
            if self.client.collections.exists(self.collection_name):
                if force_reset:
                    print(f"🗑️ Deleting existing collection: {self.collection_name}")
                    self.client.collections.delete(self.collection_name)
                else:
                    print(f"ℹ️ Collection {self.collection_name} already exists. Skipping creation.")
                    return

            print(f"🛠️ Creating collection schema: {self.collection_name}")
            
            # Simple schema with vectorizer configured (using text2vec-transformers or similar if available, 
            # but for this specific request we are doing embedding externally in nrag.py? 
            # Actually, user said "keep nrag.py till chunking and embedding logic", 
            # so we should likely pass vectors explicitly or use Weaviate's vectorizer.
            # Given we used SentenceTransformers locally in nrag.py, it's safer to pass vectors explicitly 
            # or use text2vec-transformers if we want Weaviate to do it.
            # Let's support explicit vector ingestion since nrag.py already has the model.
            
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    wvc.Property(name="content", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="source", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="page", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="chunk_id", data_type=wvc.DataType.INT),
                    wvc.Property(name="metadata_json", data_type=wvc.DataType.TEXT), # Store full metadata as string
                ],
                # If we want to bring our own vectors, we don't strictly need a vectorizer config, 
                # but explicit configuration is good.
                 vectorizer_config=wvc.Configure.Vectorizer.none(), 
            )
            print("✅ Schema created successfully")

        except Exception as e:
            print(f"❌ Error creating schema: {e}")
            traceback.print_exc()

    def ingest_chunks(self, chunks, embeddings=None):
        """Ingest chunks and optional embeddings"""
        if not self.client:
            return

        collection = self.client.collections.get(self.collection_name)
        
        print(f"🚀 Ingesting {len(chunks)} chunks into Weaviate...")
        
        try:
            with collection.batch.dynamic() as batch:
                for i, chunk in enumerate(chunks):
                    # Prepare properties
                    properties = {
                        "content": chunk.get("content", ""),
                        "source": chunk.get("metadata", {}).get("source", "unknown"),
                        "page": str(chunk.get("metadata", {}).get("page", 0)),
                        "chunk_id": chunk.get("chunk_id", i),
                        "metadata_json": json.dumps(chunk.get("metadata", {}))
                    }
                    
                    vector = None
                    if embeddings is not None and i < len(embeddings):
                        vector = embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else embeddings[i]

                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )
            
            print("✅ Ingestion complete")
            if collection.batch.failed_objects:
                print(f"⚠️ {len(collection.batch.failed_objects)} objects failed to ingest")
                for fail in collection.batch.failed_objects[:3]:
                    print(f"  - Error: {fail.message}")

        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            traceback.print_exc()

    def search(self, query_text, limit=5, query_vector=None, alpha=0.5):
        """Search using hybrid search (combines keyword BM25 and vector search)"""
        if not self.client:
            return []

        collection = self.client.collections.get(self.collection_name)
        print(f"🔍 Searching for: '{query_text}' (Limit: {limit}, Alpha: {alpha})")

        try:
            if query_vector is not None:
                # Hybrid Search
                response = collection.query.hybrid(
                    query=query_text,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit,
                    return_metadata=MetadataQuery(score=True, explain_score=True)
                )
            else:
                # Fallback to BM25 if no vector
                print("⚠️ No query vector provided. Falling back to BM25 keyword search.")
                response = collection.query.bm25(
                    query=query_text,
                    limit=limit
                )

            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content"),
                    "source": obj.properties.get("source"),
                    "page": obj.properties.get("page"),
                    "score": obj.metadata.score if obj.metadata.score else 0
                })
            
            return results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            traceback.print_exc()
            return []
