"""
Endpoint for retrieving relevant chunks from Weaviate
Includes hybrid search, cross-encoder reranking, and quality filtering.

Features:
- Hybrid search (semantic + keyword)
- Cross-encoder reranking for semantic similarity
- Advanced deduplication
- Soft score thresholding (always returns top_k results)
- Result diversity filtering
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from utils.auth_utils import get_current_user
from logger import get_logger
from utils.retrieval_optimizer import (
    preprocess_query,
    get_retrieval_strategy,
    # apply_metadata_boosting,  # DISABLED - using pure rerank scores
    advanced_deduplication,
    apply_score_thresholding,
    ensure_result_diversity,
    log_retrieval_strategy,
    format_query_summary
)
from models import UserInDB
from WeaviateManager import WeaviateManager
from config import WEAVIATE_USE_LOCAL, WEAVIATE_URL, WEAVIATE_API_KEY
from services.azure_storage import storage
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import time

# Initialize logger
logger = get_logger(__name__)

load_dotenv()

router = APIRouter()

# Global variables to cache models
_embedding_model = None
_reranker_model = None

def get_embedding_model():
    """Load and cache the embedding model for generating query vectors"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("🔄 Loading HuggingFace embedding model for hybrid search...")
        # TODO: Upgrade to 'all-mpnet-base-v2' after re-ingesting chunks
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded for hybrid search")
    return _embedding_model

def get_reranker_model():
    """Load and cache the cross-encoder model for reranking"""
    global _reranker_model
    if _reranker_model is None:
        logger.info("🔄 Loading cross-encoder model for reranking...")
        # Using L-6-v2 for faster performance
        _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("✅ Reranker model loaded: ms-marco-MiniLM-L-6-v2 (6 layers, faster performance)")
    return _reranker_model

class QueryItem(BaseModel):
    id: str
    text: str
    selected: bool

class RetrieveChunksRequest(BaseModel):
    user_email: str
    guide_name: str        # User-friendly guide title (for reference only, not used for collection naming)
    syndrome_name: str     # CRITICAL: Medical syndrome from 'concern_name' field (used for Weaviate collection)
    version: str
    queries: List[QueryItem]
    top_k: int = 5  # Number of chunks to return after reranking
    initial_limit: int = 30  # Default initial fetch size (will be overridden by strategy)
    enable_optimizations: bool = True  # Flag to enable/disable optimizations

class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    score: float  # Final score (reranker score)
    hybrid_score: float  # Original Weaviate hybrid score
    rerank_score: float  # Cross-encoder reranker score
    selected: bool
    metadata: Optional[Dict[str, Any]] = {}

class RetrieveChunksResponse(BaseModel):
    success: bool
    message: str
    query_chunks: Dict[str, List[ChunkResult]]  # query_id -> list of chunks
    total_queries: int
    total_chunks_retrieved: int
    optimization_stats: Optional[Dict[str, Any]] = {}  # Performance metrics

@router.post("/retrieve-chunks", response_model=RetrieveChunksResponse)
async def retrieve_chunks_for_queries(
    request: RetrieveChunksRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    ENHANCED: Retrieve top K chunks from Weaviate for each query using optimized hybrid search
    
    New features:
    - Query preprocessing and expansion
    - Dynamic alpha parameter based on query type
    - Adaptive initial retrieval limit
    - Metadata-based score boosting
    - Advanced deduplication
    - Score thresholding
    - Result diversity
    """
    try:
        start_time = time.time()
        logger.info(f"🔍 ENHANCED RETRIEVAL: Processing {len(request.queries)} queries...")
        
        # Initialize Weaviate Manager
        weaviate_url = os.getenv('WEAVIATE_URL') or WEAVIATE_URL
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY') or WEAVIATE_API_KEY
        use_local = WEAVIATE_USE_LOCAL
        
        # For cloud mode, API key is required
        if not use_local and not weaviate_api_key:
            raise HTTPException(
                status_code=500,
                detail="Weaviate Cloud API key not configured"
            )
        
        weaviate_manager = WeaviateManager(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key,
            use_local=use_local
        )
        
        # Connect to Weaviate
        if not weaviate_manager.connect():
            raise HTTPException(
                status_code=500,
                detail="Failed to connect to Weaviate"
            )
        
        # Create collection name
        collection_name = weaviate_manager.create_collection_name(
            user_email=request.user_email,
            syndrome_name=request.syndrome_name,
            version=request.version
        )
        
        logger.info(f"📦 Searching in collection: {collection_name}")
        
        # Check if collection exists
        if not weaviate_manager.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection {collection_name} not found. Please ingest chunks first."
            )
        
        # Get collection
        collection = weaviate_manager.client.collections.get(collection_name)
        
        # Load models
        embedding_model = get_embedding_model()
        reranker_model = get_reranker_model()
        
        # Track optimization stats
        optimization_stats = {
            'queries_processed': 0,
            'total_initial_retrieved': 0,
            'total_after_dedup': 0,
            'total_after_threshold': 0,
            'total_final': 0,
            'avg_latency_per_query_ms': 0,
            'optimizations_enabled': request.enable_optimizations
        }
        
        # Retrieve chunks for each query
        query_chunks: Dict[str, List[ChunkResult]] = {}
        total_chunks = 0
        query_latencies = []
        
        for query_item in request.queries:
            if not query_item.selected:
                logger.info(f"⏭️  Skipping unselected query: {query_item.id}")
                continue
            
            query_start_time = time.time()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔎 Processing Query: {query_item.id}")
            logger.info(f"{'='*80}")
            
            try:
                # Step 1: Preprocess query
                original_query = query_item.text
                
                if request.enable_optimizations:
                    query_text = preprocess_query(original_query, request.syndrome_name)
                    logger.info(f"📝 Query preprocessing:")
                    logger.info(f"   Original: '{original_query[:80]}...'")
                    logger.info(f"   Enhanced: '{query_text[:80]}...'")
                    summary = format_query_summary(original_query, query_text)
                    logger.info(f"   {summary}")
                else:
                    query_text = original_query.replace('[SYNDROME]', request.syndrome_name)
                    query_text = query_text.replace('[Syndrome]', request.syndrome_name)
                    logger.info(f"📝 Query: '{query_text[:80]}...'")
                
                # Step 2: Get retrieval strategy
                if request.enable_optimizations:
                    strategy = get_retrieval_strategy(query_text)
                    log_retrieval_strategy(query_text, strategy)
                    alpha = strategy['alpha']
                    initial_limit = strategy['initial_limit']
                else:
                    alpha = 0.5
                    initial_limit = request.initial_limit
                    logger.info(f"   ⚖️  Using default alpha: {alpha}")
                    logger.info(f"   📊 Using default initial limit: {initial_limit}")
                
                # Step 3: Generate embedding for the query text locally
                query_embedding = embedding_model.encode(query_text).tolist()
                
                # Step 4: Perform hybrid search (semantic + keyword)
                logger.info(f"\n🔍 Executing hybrid search...")
                response = collection.query.hybrid(
                    query=query_text,
                    vector=query_embedding,
                    alpha=alpha,
                    limit=initial_limit,
                    return_metadata=['score']
                )
                
                logger.info(f"   ✓ Retrieved {len(response.objects)} initial chunks")
                optimization_stats['total_initial_retrieved'] += len(response.objects)
                
                # Step 5: Advanced deduplication
                if request.enable_optimizations:
                    deduplicated_chunks = advanced_deduplication(
                        [{'obj': obj} for obj in response.objects],
                        similarity_threshold=0.85
                    )
                else:
                    # Simple deduplication (original method)
                    deduplicated_chunks = []
                    seen_keys = set()
                    for obj in response.objects:
                        content = obj.properties.get('content', '')
                        dedup_key = content[:120] if len(content) >= 120 else content
                        if dedup_key not in seen_keys:
                            seen_keys.add(dedup_key)
                            deduplicated_chunks.append({'obj': obj})
                            if len(deduplicated_chunks) >= initial_limit:
                                break
                    logger.info(f"   🔄 Simple deduplication: {len(response.objects)} → {len(deduplicated_chunks)}")
                
                optimization_stats['total_after_dedup'] += len(deduplicated_chunks)
                
                # Check if we have enough chunks
                if len(deduplicated_chunks) < request.top_k:
                    logger.info(f"\n⚠️  Warning: Only {len(deduplicated_chunks)} unique chunks found (target: {request.top_k})")
                
                # Step 6: Rerank chunks using cross-encoder
                if len(deduplicated_chunks) > 0:
                    logger.info(f"\n🔄 Reranking {len(deduplicated_chunks)} chunks with cross-encoder...")
                    
                    # Prepare query-document pairs for reranking
                    pairs = []
                    for chunk in deduplicated_chunks:
                        content = chunk['obj'].properties.get('content', '')
                        pairs.append([query_text, content])
                    
                    # Get reranking scores from cross-encoder
                    rerank_scores = reranker_model.predict(pairs)
                    
                    # Attach scores AND metadata to chunks
                    chunks_with_scores = []
                    for idx, chunk in enumerate(deduplicated_chunks):
                        obj = chunk['obj']
                        hybrid_score = obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0
                        rerank_score = float(rerank_scores[idx])
                        
                        # Extract metadata for boosting
                        page_nums = obj.properties.get('page', [0])
                        if not isinstance(page_nums, list):
                            page_nums = [page_nums]
                        
                        metadata = {
                            'current_heading': obj.properties.get('current_heading', ''),
                            'page_number': page_nums[0] if page_nums else 0,
                            'page_numbers': page_nums,
                            'source': obj.properties.get('source', ''),
                        }
                        
                        chunks_with_scores.append({
                            'obj': obj,
                            'hybrid_score': hybrid_score,
                            'rerank_score': rerank_score,
                            'metadata': metadata  # ADD METADATA FOR BOOSTING!
                        })
                    
                    # Sort by rerank score (descending)
                    chunks_with_scores.sort(key=lambda x: x['rerank_score'], reverse=True)
                    logger.info(f"   ✓ Reranked by cross-encoder score")
                    
                    # Step 7: Metadata boosting DISABLED - using pure rerank scores only
                    # Keeping original semantic similarity scores from cross-encoder
                    
                    # Step 8: Apply SOFT score thresholding (but always keep at least top_k chunks)
                    if request.enable_optimizations and len(chunks_with_scores) > request.top_k:
                        logger.info(f"\n🎯 Applying soft score thresholds...")
                        chunks_after_threshold, filtered_count = apply_score_thresholding(
                            chunks_with_scores,
                            min_hybrid_score=0.3,
                            min_rerank_score=0.2
                        )
                        
                        # CRITICAL: Always ensure we have at least top_k chunks
                        if len(chunks_after_threshold) < request.top_k:
                            logger.info(f"   ⚠️  Only {len(chunks_after_threshold)} chunks passed threshold, keeping top {request.top_k} original chunks")
                            chunks_with_scores = chunks_with_scores[:request.top_k * 2]  # Keep top 2x for diversity
                        else:
                            chunks_with_scores = chunks_after_threshold
                            if filtered_count > 0:
                                logger.info(f"   ✅ Filtered out {filtered_count} low-quality chunks (keeping {len(chunks_with_scores)} high-quality)")
                        
                        optimization_stats['total_after_threshold'] += len(chunks_with_scores)
                    
                    # Step 9: Take top K after all optimizations
                    top_reranked = chunks_with_scores[:request.top_k * 2]  # Get 2x for diversity filter
                    
                    # Step 10: Ensure result diversity
                    if request.enable_optimizations and len(top_reranked) > request.top_k:
                        logger.info(f"\n🌈 Ensuring result diversity...")
                        diverse_results = ensure_result_diversity(top_reranked, max_per_source=3)
                        
                        # If diversity filtering removed too many, pad back with original top results
                        if len(diverse_results) < request.top_k and len(top_reranked) >= request.top_k:
                            logger.info(f"   ⚠️  Diversity filter left only {len(diverse_results)} chunks, padding to {request.top_k}")
                            # Add back highest-scoring chunks that were filtered out
                            for chunk in top_reranked:
                                if chunk not in diverse_results and len(diverse_results) < request.top_k:
                                    diverse_results.append(chunk)
                        top_reranked = diverse_results
                    
                    # Final top K - ensure we have exactly top_k (or all available if fewer)
                    final_count = min(request.top_k, len(top_reranked))
                    top_reranked = top_reranked[:final_count]
                    
                    logger.info(f"\n✅ Final selection: {len(top_reranked)} chunks (target: {request.top_k})")
                    
                    # Step 11: Process results into ChunkResult objects
                    chunks_for_query = []
                    for idx, item in enumerate(top_reranked):
                        obj = item['obj']
                        hybrid_score = item['hybrid_score']
                        rerank_score = item['rerank_score']
                        
                        # Get actual chunk_id if available, otherwise create one
                        chunk_id = obj.properties.get('chunk_id', f"{query_item.id}_chunk_{idx}")
                        
                        # Get page numbers (handle both array and single value)
                        page_nums = obj.properties.get('page', [0])
                        if not isinstance(page_nums, list):
                            page_nums = [page_nums]
                        
                        # Get authors (handle both array and string)
                        authors = obj.properties.get('authors', [])
                        if isinstance(authors, list):
                            author_str = ', '.join(authors) if authors else ''
                        else:
                            author_str = authors
                        
                        chunk_result = ChunkResult(
                            chunk_id=str(chunk_id),
                            content=obj.properties.get('content', ''),
                            score=round(rerank_score, 4),
                            hybrid_score=round(hybrid_score, 4),
                            rerank_score=round(rerank_score, 4),
                            selected=True,
                            metadata={
                                'source': obj.properties.get('source', ''),
                                'title': obj.properties.get('title', ''),
                                'page_number': page_nums[0] if page_nums else 0,
                                'page_numbers': page_nums,
                                'author': author_str,
                                'current_heading': obj.properties.get('current_heading', ''),
                                'char_count': obj.properties.get('char_count', 0),
                                'mod_date': obj.properties.get('mod_date', ''),
                                'chunk_id': chunk_id,
                            }
                        )
                        chunks_for_query.append(chunk_result)
                        total_chunks += 1
                    
                    optimization_stats['total_final'] += len(chunks_for_query)
                    query_chunks[query_item.id] = chunks_for_query
                    
                    # Log top result
                    if chunks_for_query:
                        top = chunks_for_query[0]
                        logger.info(f"\n🏆 Top Result:")
                        logger.info(f"   Rerank Score: {top.rerank_score:.4f}")
                        logger.info(f"   Hybrid Score: {top.hybrid_score:.4f}")
                        logger.info(f"   Source: {top.metadata.get('source', 'N/A')}")
                        logger.info(f"   Page: {top.metadata.get('page_number', 'N/A')}")
                        preview = top.content[:150] + "..." if len(top.content) > 150 else top.content
                        logger.info(f"   Preview: {preview}")
                else:
                    query_chunks[query_item.id] = []
                    logger.info(f"⚠️  No chunks found for query {query_item.id}")
                
                # Track query latency
                query_latency = (time.time() - query_start_time) * 1000
                query_latencies.append(query_latency)
                optimization_stats['queries_processed'] += 1
                
                logger.info(f"\n⏱️  Query processed in {query_latency:.0f}ms")
                
            except Exception as e:
                logger.info(f"❌ Error searching for query {query_item.id}: {str(e)}")
                import traceback
                # traceback handled by exc_info=True
                query_chunks[query_item.id] = []
        
        # Disconnect from Weaviate
        weaviate_manager.disconnect()
        
        # Calculate final stats
        total_latency = (time.time() - start_time) * 1000
        optimization_stats['avg_latency_per_query_ms'] = (
            sum(query_latencies) / len(query_latencies) if query_latencies else 0
        )
        optimization_stats['total_latency_ms'] = total_latency
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ RETRIEVAL COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total queries processed: {optimization_stats['queries_processed']}")
        logger.info(f"Total chunks retrieved: {total_chunks}")
        logger.info(f"Total time: {total_latency:.0f}ms")
        logger.info(f"Avg time per query: {optimization_stats['avg_latency_per_query_ms']:.0f}ms")
        
        if request.enable_optimizations:
            logger.info(f"\n📊 OPTIMIZATION STATS:")
            logger.info(f"   Initial retrieved: {optimization_stats['total_initial_retrieved']}")
            logger.info(f"   After dedup: {optimization_stats['total_after_dedup']}")
            logger.info(f"   After threshold: {optimization_stats['total_after_threshold']}")
            logger.info(f"   Final results: {optimization_stats['total_final']}")
            logger.info(f"   Quality improvement: {((optimization_stats['total_after_threshold'] / max(optimization_stats['total_initial_retrieved'], 1)) * 100):.1f}%")
        
        return RetrieveChunksResponse(
            success=True,
            message=f"Retrieved {total_chunks} chunks for {len(query_chunks)} queries",
            query_chunks=query_chunks,
            total_queries=len(query_chunks),
            total_chunks_retrieved=total_chunks,
            optimization_stats=optimization_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"❌ Error in retrieve_chunks_for_queries: {str(e)}")
        import traceback
        # traceback handled by exc_info=True
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )

# Keep the save endpoint unchanged
class SaveRetrievedChunksRequest(BaseModel):
    user_email: str
    guide_id: str
    guide_name: str
    syndrome_name: str
    version: str
    retrieved_chunks: Dict[str, List[ChunkResult]]

class SaveRetrievedChunksResponse(BaseModel):
    success: bool
    message: str
    json_file_path: str
    total_chunks_saved: int

@router.post("/save-retrieved-chunks", response_model=SaveRetrievedChunksResponse)
async def save_retrieved_chunks(
    request: SaveRetrievedChunksRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """Save retrieved chunks to JSON file and database"""
    try:
        from db import supabase
        
        sanitized_email = request.user_email.replace('@', '_at_').replace('.', '_')
        sanitized_syndrome = request.syndrome_name.lower().replace(' ', '_').replace('/', '_')
        sanitized_version = request.version.replace('.', '_')
        
        filename = f"{sanitized_email}_{sanitized_syndrome}_v{sanitized_version}_retrieved_chunks.json"
        file_path = f"uploads/{filename}"
        
        chunks_data = {}
        total_chunks = 0
        
        for query_id, chunks in request.retrieved_chunks.items():
            chunks_data[query_id] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score,
                    "hybrid_score": chunk.hybrid_score,
                    "rerank_score": chunk.rerank_score,
                    "selected": chunk.selected,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
                if chunk.selected
            ]
            total_chunks += len(chunks_data[query_id])
        
        # Save to Azure
        storage.write_json({
            "user_email": request.user_email,
            "guide_name": request.guide_name,
            "syndrome_name": request.syndrome_name,
            "version": request.version,
            "total_queries": len(chunks_data),
            "total_chunks": total_chunks,
            "query_chunks": chunks_data
        }, file_path)
        
        logger.info(f"✅ Saved retrieved chunks to {file_path}")
        
        update_response = supabase.table("guides").update({
            "retrieved_chunks": chunks_data,
            "updated_at": "now()"
        }).eq("id", request.guide_id).execute()
        
        if not update_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update database with retrieved chunks"
            )
        
        logger.info(f"✅ Updated database for guide {request.guide_id}")
        
        return SaveRetrievedChunksResponse(
            success=True,
            message=f"Successfully saved {total_chunks} chunks for {len(chunks_data)} queries",
            json_file_path=str(file_path),
            total_chunks_saved=total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in save_retrieved_chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save retrieved chunks: {str(e)}"
        )

# ============================================================================
# NEW BATCH RETRIEVAL ENDPOINTS (Azure-adapted with local fallback)
# ============================================================================

def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename. Normalized to lowercase for consistency."""
    return name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_").replace(".", "_")

# New Pydantic Models for Batch Retrieval
class BatchRetrieveRequest(BaseModel):
    user_email: str
    guide_name: str
    syndrome_name: str
    version: str
    concern_name: Optional[str] = None  # NEW: For saving to raw_unsaved_chunks
    queries: List[QueryItem]
    top_k: int = 15

class BatchRetrieveResponse(BaseModel):
    success: bool
    message: str
    total_queries: int
    total_chunks: int
    cache_folder: str

class GetCachedChunksRequest(BaseModel):
    user_email: str
    guide_name: str
    syndrome_name: str
    version: str
    query_ids: List[str]
    concern_name: Optional[str] = None  # For auto-generated concerns

class CachedChunkResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    hybrid_score: float
    rerank_score: float
    selected: bool
    metadata: Optional[Dict[str, Any]] = {}

class GetCachedChunksResponse(BaseModel):
    success: bool
    message: str
    query_chunks: Dict[str, List[CachedChunkResult]]
    total_queries: int
    total_chunks_retrieved: int

class UpdateQueryChunksRequest(BaseModel):
    user_email: str
    guide_name: str
    syndrome_name: str
    version: str
    query_id: str
    query_text: str
    top_k: int

@router.post("/batch-retrieve-chunks", response_model=BatchRetrieveResponse)
async def batch_retrieve_chunks(
    request: BatchRetrieveRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Retrieve chunks for ALL queries at once and store in retrieved_chunks folder.
    Each query's chunks are stored in a separate JSON file.
    Azure-first with local fallback.
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"📦 BATCH CHUNK RETRIEVAL")
        logger.info(f"{'='*80}")
        logger.info(f"User: {request.user_email}")
        logger.info(f"Guide: {request.guide_name} v{request.version}")
        logger.info(f"Syndrome: {request.syndrome_name}")
        logger.info(f"Queries: {len(request.queries)}")
        logger.info(f"Top K: {request.top_k}")
        
        # Initialize Weaviate Manager
        weaviate_url = os.getenv('WEAVIATE_URL') or WEAVIATE_URL
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY') or WEAVIATE_API_KEY
        use_local = WEAVIATE_USE_LOCAL
        
        if not use_local and not weaviate_api_key:
            raise HTTPException(
                status_code=500,
                detail="Weaviate Cloud API key not configured"
            )
        
        weaviate_manager = WeaviateManager(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key,
            use_local=use_local
        )
        
        if not weaviate_manager.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to Weaviate")
        
        # Create collection name
        collection_name = weaviate_manager.create_collection_name(
            user_email=request.user_email,
            syndrome_name=request.syndrome_name,
            version=request.version
        )
        
        logger.info(f"📚 Collection: {collection_name}")
        
        # Check if collection exists
        if not weaviate_manager.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection {collection_name} not found. Please ingest chunks first."
            )
        
        # Create retrieved_chunks folder path (Azure)
        safe_user = sanitize_filename(request.user_email.replace("@", "_at_").replace(".", "_"))
        safe_guide = sanitize_filename(request.guide_name)
        safe_version = sanitize_filename(request.version if request.version.startswith('v') else f"v{request.version}")
        
        azure_folder = f"uploads/{safe_user}_{safe_guide}_{safe_version}/retrieved_chunks"
        local_folder = Path(os.getenv("UPLOADS_DIR", "./uploads")) / f"{safe_user}_{safe_guide}_{safe_version}" / "retrieved_chunks"
        
        # Local folder will be created only if Azure fails (in exception handler)
        
        logger.info(f"📁 Azure folder: {azure_folder}")
        logger.info(f"📁 Local fallback: {local_folder}")
        
        # Load embedding and reranker models
        embedding_model = get_embedding_model()
        reranker_model = get_reranker_model()
        
        # Get collection
        collection = weaviate_manager.client.collections.get(collection_name)
        
        total_chunks = 0
        
        # Process each query
        for query in request.queries:
            if not query.selected:
                continue
            
            logger.info(f"\n🔍 Processing query: {query.id}")
            logger.info(f"   Query text: {query.text[:100]}...")
            
            # Preprocess query (returns enhanced query string)
            query_text = preprocess_query(query.text, request.syndrome_name)
            
            # Get retrieval strategy (returns dict with alpha and initial_limit)
            strategy = get_retrieval_strategy(query_text)
            log_retrieval_strategy(query_text, strategy)
            
            # Generate query vector
            query_vector = embedding_model.encode(query_text).tolist()
            
            # Hybrid search using Weaviate collection API
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=strategy['alpha'],
                limit=strategy['initial_limit'],
                return_metadata=['score']
            )
            
            if not response.objects:
                logger.warning(f"   ⚠️  No results for query {query.id}")
                # Save empty results
                query_data = {
                    "query_id": query.id,
                    "query_text": query_text,
                    "top_k": request.top_k,
                    "total_chunks": 0,
                    "chunks": []
                }
                
                # Try Azure first
                azure_path = f"{azure_folder}/{query.id}.json"
                try:
                    storage.write_json(query_data, azure_path)
                    logger.info(f"   ✅ Saved to Azure: {azure_path}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Azure failed: {e}, using local fallback")
                    local_folder.mkdir(parents=True, exist_ok=True)
                    local_file = local_folder / f"{query.id}.json"
                    with open(local_file, 'w', encoding='utf-8') as f:
                        json.dump(query_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"   ✅ Saved to local: {local_file}")
                continue
            
            # Convert Weaviate objects to dict format for reranking
            results = []
            for obj in response.objects:
                # Get page numbers (handle both array and single value)
                page_nums = obj.properties.get('page', [0])
                if not isinstance(page_nums, list):
                    page_nums = [page_nums]
                
                # Get authors (handle both array and string)
                authors = obj.properties.get('authors', [])
                if isinstance(authors, list):
                    author_str = ', '.join(authors) if authors else ''
                else:
                    author_str = authors
                
                # Get chunk_id
                chunk_id = obj.properties.get('chunk_id', str(obj.uuid))
                
                results.append({
                    'uuid': str(obj.uuid),
                    'content': obj.properties.get('content', ''),
                    'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                    'metadata': {
                        'source': obj.properties.get('source', ''),
                        'title': obj.properties.get('title', ''),
                        'page_number': page_nums[0] if page_nums else 0,
                        'page_numbers': page_nums,
                        'author': author_str,
                        'current_heading': obj.properties.get('current_heading', ''),
                        'char_count': obj.properties.get('char_count', 0),
                        'mod_date': obj.properties.get('mod_date', ''),
                        'chunk_id': chunk_id
                    }
                })
            
            # Rerank results
            pairs = [[query_text, r['content']] for r in results]
            rerank_scores = reranker_model.predict(pairs, show_progress_bar=False)
            
            # Attach rerank scores
            for result, rerank_score in zip(results, rerank_scores):
                result['rerank_score'] = float(rerank_score)
                result['hybrid_score'] = result.get('score', 0.0)
                result['score'] = float(rerank_score)  # Use rerank score as final score
            
            # Sort by rerank score
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Apply simple deduplication
            deduplicated_results = []
            seen_contents = set()
            for result in results:
                content = result['content']
                dedup_key = content[:120] if len(content) >= 120 else content
                if dedup_key not in seen_contents:
                    seen_contents.add(dedup_key)
                    deduplicated_results.append(result)
            
            logger.info(f"   🔄 Deduplication: {len(results)} → {len(deduplicated_results)} unique")
            
            # Take top K
            top_results = deduplicated_results[:request.top_k]
            
            # Prepare chunks data
            chunks_data = []
            for i, result in enumerate(top_results):
                # Use chunk_id from metadata if available, otherwise use uuid
                chunk_id = result.get('metadata', {}).get('chunk_id', result.get('uuid', f"chunk_{i}"))
                
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "content": result['content'],
                    "score": result['score'],
                    "hybrid_score": result['hybrid_score'],
                    "rerank_score": result['rerank_score'],
                    "selected": i < request.top_k,
                    "metadata": result.get('metadata', {})
                })
            
            # Save to JSON file (Azure first, local fallback)
            query_file_data = {
                "query_id": query.id,
                "query_text": query_text,
                "top_k": request.top_k,
                "total_chunks": len(chunks_data),
                "chunks": chunks_data
            }
            
            azure_path = f"{azure_folder}/{query.id}.json"
            try:
                storage.write_json(query_file_data, azure_path)
                logger.info(f"   ✅ Saved {len(chunks_data)} chunks to Azure: {azure_path}")
            except Exception as e:
                logger.warning(f"   ⚠️  Azure failed: {e}, using local fallback")
                local_folder.mkdir(parents=True, exist_ok=True)
                local_file = local_folder / f"{query.id}.json"
                with open(local_file, 'w', encoding='utf-8') as f:
                    json.dump(query_file_data, f, indent=2, ensure_ascii=False)
                logger.info(f"   ✅ Saved {len(chunks_data)} chunks to local: {local_file}")
            
            total_chunks += len(chunks_data)
        
        # NEW: Save all retrieved chunks to raw_unsaved_chunks/concern_name.json
        # This is for batch mini-summary generation
        if hasattr(request, 'concern_name') and request.concern_name:
            safe_concern = sanitize_filename(request.concern_name)
            raw_chunks_folder = f"uploads/{safe_user}_{safe_guide}_{safe_version}/raw_unsaved_chunks"
            raw_chunks_file = f"{raw_chunks_folder}/{safe_concern}.json"
            
            logger.info(f"\n💾 Saving raw chunks for batch generation")
            logger.info(f"   Concern: {request.concern_name}")
            logger.info(f"   File: {raw_chunks_file}")
            
            # Collect all query chunks from Azure retrieved_chunks folder
            all_query_chunks = {}
            for query in request.queries:
                if not query.selected:
                    continue
                
                # Load the individual query file we just saved
                azure_query_path = f"{azure_folder}/{query.id}.json"
                try:
                    if storage.file_exists(azure_query_path):
                        query_data = storage.read_json(azure_query_path)
                        all_query_chunks[query.id] = {
                            "query_id": query.id,
                            "query_text": query_data.get('query_text', query.text),
                            "chunks": query_data.get('chunks', []),
                            "chunks_count": len(query_data.get('chunks', [])),
                            "top_k": request.top_k,
                            "retrieved_at": query_data.get('retrieved_at', None)
                        }
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not load chunks for {query.id}: {e}")
                    continue
            
            # Create the consolidated file structure
            raw_chunks_data = {
                "concern_name": request.concern_name,
                "guide_name": request.guide_name,
                "syndrome_name": request.syndrome_name,
                "version": request.version,
                "user_email": request.user_email,
                "queries": all_query_chunks,
                "total_queries": len(all_query_chunks),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            # Save to Azure
            try:
                storage.write_json(raw_chunks_data, raw_chunks_file)
                logger.info(f"   ✅ Saved raw chunks to Azure: {raw_chunks_file}")
                logger.info(f"   📊 Total queries: {len(all_query_chunks)}")
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to save raw chunks: {e}")
        
        # Disconnect from Weaviate
        weaviate_manager.disconnect()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ BATCH RETRIEVAL COMPLETE")
        logger.info(f"   Total queries processed: {len([q for q in request.queries if q.selected])}")
        logger.info(f"   Total chunks retrieved: {total_chunks}")
        logger.info(f"   Cache folder: {azure_folder}")
        logger.info(f"{'='*80}\n")
        
        return BatchRetrieveResponse(
            success=True,
            message=f"Successfully retrieved {total_chunks} chunks for {len([q for q in request.queries if q.selected])} queries",
            total_queries=len([q for q in request.queries if q.selected]),
            total_chunks=total_chunks,
            cache_folder=azure_folder
        )
        
    except HTTPException:
        try:
            weaviate_manager.disconnect()
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"❌ Error in batch_retrieve_chunks: {str(e)}", exc_info=True)
        try:
            weaviate_manager.disconnect()
        except:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch retrieve chunks: {str(e)}"
        )

@router.post("/get-cached-chunks", response_model=GetCachedChunksResponse)
async def get_cached_chunks(
    request: GetCachedChunksRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Load chunks from cached JSON files in retrieved_chunks folder.
    Azure-first with local fallback.
    """
    try:
        logger.info(f"\n📂 Loading cached chunks for {len(request.query_ids)} queries")
        
        # Build folder path
        safe_user = sanitize_filename(request.user_email.replace("@", "_at_").replace(".", "_"))
        safe_guide = sanitize_filename(request.guide_name)
        safe_version = sanitize_filename(request.version if request.version.startswith('v') else f"v{request.version}")
        
        azure_folder = f"uploads/{safe_user}_{safe_guide}_{safe_version}/retrieved_chunks"
        local_folder = Path(os.getenv("UPLOADS_DIR", "./uploads")) / f"{safe_user}_{safe_guide}_{safe_version}" / "retrieved_chunks"
        
        query_chunks = {}
        total_chunks = 0
        
        for query_id in request.query_ids:
            azure_path = f"{azure_folder}/{query_id}.json"
            local_file = local_folder / f"{query_id}.json"
            
            # Try Azure first
            data = None
            try:
                if storage.file_exists(azure_path):
                    data = storage.read_json(azure_path)
                    logger.info(f"   ✅ Loaded from Azure: {query_id}")
            except Exception as e:
                logger.warning(f"   ⚠️  Azure failed for {query_id}: {e}, trying local")
            
            # Fall back to local if Azure failed
            if data is None:
                if local_file.exists():
                    with open(local_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"   ✅ Loaded from local: {query_id}")
                else:
                    logger.warning(f"   ⚠️  Cache file not found for query: {query_id}")
                    
                    # For auto-generated concerns, try loading from concern_data
                    if request.concern_name:
                        try:
                            safe_concern = sanitize_filename(request.concern_name)
                            concern_data_path = f"uploads/{safe_user}_{safe_guide}_{safe_version}/concern_data/{safe_concern}.json"
                            
                            if storage.file_exists(concern_data_path):
                                concern_data = storage.read_json(concern_data_path)
                                
                                # Extract chunks for this specific query from concern_data
                                if 'queries' in concern_data and query_id in concern_data['queries']:
                                    query_data = concern_data['queries'][query_id]
                                    if 'chunks' in query_data and query_data['chunks']:
                                        # Format as cached chunk data
                                        data = {
                                            'query_id': query_id,
                                            'chunks': query_data['chunks']
                                        }
                                        logger.info(f"   ✅ Loaded from concern_data (auto-generated): {query_id}")
                        except Exception as e:
                            logger.warning(f"   ⚠️  Failed to load from concern_data: {e}")
                    
                    # If still no data, skip this query
                    if data is None:
                        continue
            
            # All chunks now use same structure (both auto-generated and manual)
            chunks = [
                CachedChunkResult(
                    chunk_id=str(chunk['chunk_id']),
                    content=chunk['content'],  # ✅ Consistent field name
                    score=chunk['score'],
                    hybrid_score=chunk['hybrid_score'],
                    rerank_score=chunk['rerank_score'],
                    selected=chunk['selected'],
                    metadata=chunk.get('metadata', {})
                )
                for chunk in data['chunks']
            ]
            
            query_chunks[query_id] = chunks
            total_chunks += len(chunks)
        
        logger.info(f"✅ Loaded {total_chunks} chunks from cache for {len(query_chunks)} queries\n")
        
        return GetCachedChunksResponse(
            success=True,
            message=f"Successfully loaded {total_chunks} chunks from cache",
            query_chunks=query_chunks,
            total_queries=len(query_chunks),
            total_chunks_retrieved=total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in get_cached_chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load cached chunks: {str(e)}"
        )

@router.post("/update-query-chunks")
async def update_query_chunks(
    request: UpdateQueryChunksRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Update chunks for a single query when user changes top_k.
    Re-retrieves from Weaviate and updates the cached JSON file.
    Azure-first with local fallback.
    """
    try:
        logger.info(f"\n🔄 Updating chunks for query: {request.query_id} (top_k={request.top_k})")
        
        # Initialize Weaviate Manager
        weaviate_url = os.getenv('WEAVIATE_URL') or WEAVIATE_URL
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY') or WEAVIATE_API_KEY
        use_local = WEAVIATE_USE_LOCAL
        
        if not use_local and not weaviate_api_key:
            raise HTTPException(status_code=500, detail="Weaviate Cloud API key not configured")
        
        weaviate_manager = WeaviateManager(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key,
            use_local=use_local
        )
        
        if not weaviate_manager.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to Weaviate")
        
        # Create collection name
        collection_name = weaviate_manager.create_collection_name(
            user_email=request.user_email,
            syndrome_name=request.syndrome_name,
            version=request.version
        )
        
        if not weaviate_manager.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
        
        # Load models
        embedding_model = get_embedding_model()
        reranker_model = get_reranker_model()
        
        # Get collection
        collection = weaviate_manager.client.collections.get(collection_name)
        
        logger.info(f"🔍 Updating chunks for query: {request.query_id}")
        logger.info(f"   Query text: {request.query_text[:100]}...")
        logger.info(f"   New top_k: {request.top_k}")
        
        # Process query (returns enhanced query string)
        query_text = preprocess_query(request.query_text, request.syndrome_name)
        
        # Get retrieval strategy (returns dict with alpha and initial_limit)
        strategy = get_retrieval_strategy(query_text)
        query_vector = embedding_model.encode(query_text).tolist()
        
        # Hybrid search using Weaviate collection API
        response = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=strategy['alpha'],
            limit=strategy['initial_limit'],
            return_metadata=['score']
        )
        
        if not response.objects:
            logger.warning(f"   ⚠️  No results found")
            return {"success": False, "message": "No chunks found", "chunks": []}
        
        # Convert Weaviate objects to dict format
        results = []
        for obj in response.objects:
            # Get page numbers (handle both array and single value)
            page_nums = obj.properties.get('page', [0])
            if not isinstance(page_nums, list):
                page_nums = [page_nums]
            
            # Get authors (handle both array and string)
            authors = obj.properties.get('authors', [])
            if isinstance(authors, list):
                author_str = ', '.join(authors) if authors else ''
            else:
                author_str = authors
            
            # Get chunk_id
            chunk_id = obj.properties.get('chunk_id', str(obj.uuid))
            
            results.append({
                'uuid': str(obj.uuid),
                'content': obj.properties.get('content', ''),
                'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                'metadata': {
                    'source': obj.properties.get('source', ''),
                    'title': obj.properties.get('title', ''),
                    'page_number': page_nums[0] if page_nums else 0,
                    'page_numbers': page_nums,
                    'author': author_str,
                    'current_heading': obj.properties.get('current_heading', ''),
                    'char_count': obj.properties.get('char_count', 0),
                    'mod_date': obj.properties.get('mod_date', ''),
                    'chunk_id': chunk_id
                }
            })
        
        # Rerank
        pairs = [[query_text, r['content']] for r in results]
        rerank_scores = reranker_model.predict(pairs, show_progress_bar=False)
        
        for result, rerank_score in zip(results, rerank_scores):
            result['rerank_score'] = float(rerank_score)
            result['hybrid_score'] = result.get('score', 0.0)
            result['score'] = float(rerank_score)
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply simple deduplication
        deduplicated_results = []
        seen_contents = set()
        for result in results:
            content = result['content']
            dedup_key = content[:120] if len(content) >= 120 else content
            if dedup_key not in seen_contents:
                seen_contents.add(dedup_key)
                deduplicated_results.append(result)
        
        logger.info(f"   🔄 Deduplication: {len(results)} → {len(deduplicated_results)} unique")
        
        top_results = deduplicated_results[:request.top_k]
        
        # Prepare chunks
        chunks_data = []
        for i, result in enumerate(top_results):
            # Use chunk_id from metadata if available, otherwise use uuid
            chunk_id = result.get('metadata', {}).get('chunk_id', result.get('uuid', f"chunk_{i}"))
            
            chunks_data.append({
                "chunk_id": chunk_id,
                "content": result['content'],
                "score": result['score'],
                "hybrid_score": result['hybrid_score'],
                "rerank_score": result['rerank_score'],
                "selected": i < request.top_k,
                "metadata": result.get('metadata', {})
            })
        
        # Update cached JSON file (Azure first, local fallback)
        safe_user = sanitize_filename(request.user_email.replace("@", "_at_").replace(".", "_"))
        safe_guide = sanitize_filename(request.guide_name)
        safe_version = sanitize_filename(request.version if request.version.startswith('v') else f"v{request.version}")
        
        azure_folder = f"uploads/{safe_user}_{safe_guide}_{safe_version}/retrieved_chunks"
        local_folder = Path(os.getenv("UPLOADS_DIR", "./uploads")) / f"{safe_user}_{safe_guide}_{safe_version}" / "retrieved_chunks"
        
        query_file_data = {
            "query_id": request.query_id,
            "query_text": query_text,
            "top_k": request.top_k,
            "total_chunks": len(chunks_data),
            "chunks": chunks_data
        }
        
        azure_path = f"{azure_folder}/{request.query_id}.json"
        try:
            storage.write_json(query_file_data, azure_path)
            logger.info(f"   ✅ Updated {len(chunks_data)} chunks in Azure: {azure_path}\n")
        except Exception as e:
            logger.warning(f"   ⚠️  Azure failed: {e}, using local fallback")
            local_folder.mkdir(parents=True, exist_ok=True)
            local_file = local_folder / f"{request.query_id}.json"
            with open(local_file, 'w', encoding='utf-8') as f:
                json.dump(query_file_data, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Updated {len(chunks_data)} chunks in local: {local_file}\n")
        
        # Disconnect from Weaviate
        weaviate_manager.disconnect()
        
        return {
            "success": True,
            "message": f"Successfully updated {len(chunks_data)} chunks",
            "chunks": chunks_data
        }
        
    except HTTPException:
        try:
            weaviate_manager.disconnect()
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"❌ Error in update_query_chunks: {str(e)}", exc_info=True)
        try:
            weaviate_manager.disconnect()
        except:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update query chunks: {str(e)}"
        )

