# endpoints/chunking.py
# PDF chunking and processing endpoints

from fastapi import APIRouter, HTTPException, Form
from pathlib import Path
import os
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

from utils.document_chunker_wrapper import chunk_pdfs_in_folder
from utils.table_enhancer_wrapper import enhance_table_chunks
from services.azure_storage import storage

# Create router for chunking endpoints
router = APIRouter(tags=["Chunking"])


# -------------------------- PARSE AND CHUNK PDFs --------------------------

@router.post("/parse-chunk")
async def parse_and_chunk_pdfs(
    user_email: str = Form(...),
    guide_name: str = Form(...),
    enhance_tables: bool = Form(True),
    version: str = Form(None),
    guide_id: str = Form(None),
    force_reprocess: bool = Form(False)
):
    """
    Parse and chunk uploaded PDFs for a user's guide.
    
    This endpoint:
    1. Locates the uploaded PDFs in version-specific folder
    2. Checks if chunks already exist (skips reprocessing unless forced)
    3. Uses DocumentChunker to parse and chunk the PDFs
    4. Optionally enhances table extraction using TableEnhancer with Gemini
    5. Saves the final chunks JSON file and updates database
    
    Args:
        user_email: Email of the user who uploaded the PDFs
        guide_name: Name of the guide (syndrome name)
        enhance_tables: Whether to enhance table extraction with Gemini (default: True)
        version: Version of the guide (e.g., "1.0")
        guide_id: Optional guide ID to fetch version and update chunks_file_path
        force_reprocess: Force reprocessing even if chunks exist (default: False)
    
    Returns:
        dict: Processing status, message, output file path, and statistics
    
    Raises:
        HTTPException: 400 for invalid inputs, 404 if PDFs not found, 500 for processing errors
    """
    try:
        # Import here to avoid circular imports
        from db import supabase
        from config import is_demo_mode
        import json
        
        # ============================================================
        # DEMO MODE: Skip PDF processing and use pre-loaded chunks
        # ============================================================
        if is_demo_mode(user_email, guide_name):
            logger.info(f"🎯 DEMO MODE activated for {user_email} / {guide_name}")
            logger.info(f"⚡ Skipping PDF processing, using pre-loaded chunks...")
            
            # Construct paths
            sanitized_email = user_email.replace('@', '_at_').replace('.', '_')
            version_suffix = f"_v{version.replace('.', '_')}" if version else "_v1_0"
            folder_name = f"{sanitized_email}_{guide_name.lower()}{version_suffix}"
            
            # Load pre-processed chunks from local Chunks folder
            # Use Path to get absolute path relative to backend directory
            backend_dir = Path(__file__).parent.parent  # Go up from endpoints/ to backend/
            local_chunks_path = backend_dir / "Chunks" / guide_name.upper() / "chunks_tables_enhanced.json"
            local_chunks_path_str = str(local_chunks_path)
            
            if not os.path.exists(local_chunks_path):
                logger.error(f"❌ Demo chunks file not found: {local_chunks_path}")
                logger.error(f"   Looked in: {local_chunks_path_str}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Demo chunks file not found for {guide_name}. Please add chunks_tables_enhanced.json to backend/Chunks/{guide_name.upper()}/"
                )
            
            # Read local chunks file
            logger.info(f"📂 Loading demo chunks from: {local_chunks_path_str}")
            with open(local_chunks_path_str, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Upload chunks to Azure (so rest of app works normally)
            azure_output_path = f"uploads/{folder_name}/chunks_tables_enhanced.json"
            storage.write_json(chunks_data, azure_output_path)
            logger.info(f"✅ Demo chunks uploaded to Azure: {azure_output_path}")
            
            # Update database with chunks file path if guide_id provided
            if guide_id:
                try:
                    supabase.table("guides").update({
                        "chunks_file_path": azure_output_path
                    }).eq("id", guide_id).execute()
                    logger.info(f"✅ Updated guide {guide_id} with chunks_file_path")
                except Exception as e:
                    logger.warning(f"⚠️  Could not update guide chunks_file_path: {e}")
            
            # Get chunk statistics
            chunks_info = chunks_data.get('document_info', {})
            total_chunks = chunks_info.get('total_chunks', len(chunks_data.get('chunks', [])))
            pdf_count = chunks_info.get('pdf_count', 0)
            
            logger.info(f"🎯 DEMO MODE complete: {total_chunks} chunks loaded instantly!")
            
            return {
                "status": "success",
                "message": f"✅ Chunking complete (DEMO MODE - instant processing)",
                "chunks_file": azure_output_path,
                "total_chunks": total_chunks,
                "pdf_count": pdf_count,
                "pdf_files": [],
                "tables_enhanced": True,
                "folder_path": f"uploads/{folder_name}",
                "demo_mode": True
            }
        
        # ============================================================
        # NORMAL MODE: Continue with regular PDF processing
        # ============================================================
        
        # If guide_id provided, fetch version and concern_name from database
        if guide_id:
            try:
                guide_response = supabase.table("guides")\
                    .select("version, concern_name")\
                    .eq("id", guide_id)\
                    .eq("user_email", user_email)\
                    .execute()
                
                if guide_response.data and len(guide_response.data) > 0:
                    guide_data = guide_response.data[0]
                    version = guide_data['version']
                    guide_name = guide_data.get('concern_name', guide_name)
                    logger.info(f"✅ Fetched guide data: version={version}, guide_name={guide_name}")
            except Exception as e:
                logger.warning(f"⚠️  Could not fetch guide data: {e}")
        
        # If still no version, calculate the next version from database
        if not version:
            try:
                # Query all guides for this user and syndrome to find the next version
                all_guides_response = supabase.table("guides")\
                    .select("version")\
                    .eq("user_email", user_email)\
                    .eq("guide_name", guide_name)\
                    .execute()
                
                if all_guides_response.data and len(all_guides_response.data) > 0:
                    # Extract version numbers
                    existing_versions = []
                    for guide in all_guides_response.data:
                        ver_str = guide['version'].replace('v', '').replace('V', '')
                        try:
                            existing_versions.append(float(ver_str))
                        except:
                            existing_versions.append(1.0)
                    
                    # Calculate next version
                    max_version = max(existing_versions)
                    next_version = f"{int(max_version) + 1}.0"
                    version = next_version
                    logger.info(f"🎯 Calculated next version for {guide_name}: {version}")
                else:
                    # First version for this syndrome
                    version = "1.0"
                    logger.info(f"🆕 First version for {guide_name}: {version}")
            except Exception as e:
                logger.warning(f"⚠️  Error calculating version: {e}")
                version = "1.0"
        
        # Construct local paths
        backend_dir = Path(__file__).parent.parent
        sanitized_email = user_email.replace("@", "_at_").replace(".", "_")
        # Normalize guide name to lowercase for consistency
        sanitized_guide = guide_name.lower().replace(" ", "_").replace("/", "_")
        sanitized_version = version.replace(".", "_")
        
        folder_name = f"{sanitized_email}_{sanitized_guide}_v{sanitized_version}"
        
        # Local folders
        pdf_folder = backend_dir / "input_pdfs" / folder_name
        output_folder = backend_dir / "output_chunks" / folder_name
        table_images_folder = backend_dir / "table_images" / folder_name
        
        # Verify the local PDF folder exists
        if not pdf_folder.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Local PDF folder not found. Please upload PDFs first."
            )
        
        # Create output and table folders
        output_folder.mkdir(parents=True, exist_ok=True)
        table_images_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if chunks already exist locally (even without guide_id)
        if not force_reprocess:
            local_enhanced_chunks = output_folder / "chunks_tables_enhanced.json"
            local_regular_chunks = output_folder / "chunks.json"
            
            if local_enhanced_chunks.exists():
                logger.info(f"✅ Enhanced chunks already exist locally, skipping reprocessing")
                with open(local_enhanced_chunks, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                chunks_info = chunks_data.get('document_info', {})
                return {
                    "status": "success",
                    "message": "Documents already processed. Using existing enhanced chunks.",
                    "chunks_file": str(local_enhanced_chunks),
                    "total_chunks": chunks_info.get('total_chunks', len(chunks_data.get('chunks', []))),
                    "pdf_count": chunks_info.get('pdf_count', 0),
                    "pdf_files": [],
                    "tables_enhanced": True,
                    "folder_path": str(pdf_folder),
                    "already_processed": True
                }
            elif local_regular_chunks.exists():
                logger.info(f"✅ Regular chunks already exist locally, skipping reprocessing")
                with open(local_regular_chunks, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                chunks_info = chunks_data.get('document_info', {})
                return {
                    "status": "success",
                    "message": "Documents already processed. Using existing chunks.",
                    "chunks_file": str(local_regular_chunks),
                    "total_chunks": chunks_info.get('total_chunks', len(chunks_data.get('chunks', []))),
                    "pdf_count": chunks_info.get('pdf_count', 0),
                    "pdf_files": [],
                    "tables_enhanced": False,
                    "folder_path": str(pdf_folder),
                    "already_processed": True
                }
        
        # Get PDF files from local folder
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            raise HTTPException(
                status_code=404,
                detail=f"No PDF files found in the local folder."
            )
        
        pdf_count = len(pdf_files)
        pdf_file_names = [f.name for f in pdf_files]
        
        logger.info(f"📄 Processing {pdf_count} PDF(s) locally for {user_email}/{guide_name}")
        
        # Step 1: Chunk the PDFs using DocumentChunker (save to local output folder)
        chunks_output_path = str(output_folder / "chunks.json")
        
        logger.info("🔄 Step 1/2: Chunking PDFs with DocumentChunker (local)...")
        chunking_result = chunk_pdfs_in_folder(
            pdf_folder_path=str(pdf_folder),
            output_json_path=chunks_output_path
        )
        
        if not chunking_result or chunking_result.get("status") != "success":
            error_msg = chunking_result.get("message", "Unknown error") if chunking_result else "Chunking failed"
            raise HTTPException(
                status_code=500,
                detail=f"PDF chunking failed: {error_msg}"
            )
        
        logger.info(f"✅ Chunking complete: {chunking_result.get('total_chunks', 0)} chunks created")
        
        # Step 2: Enhance tables if requested (using local table_images folder)
        final_output_path = str(chunks_output_path)
        enhancement_result = None
        
        if enhance_tables:
            logger.info("🔄 Step 2/2: Enhancing table extraction with Gemini (local)...")
            enhancement_result = await enhance_table_chunks(
                chunks_json_path=chunks_output_path,
                pdf_folder_path=str(pdf_folder),
                images_dir=str(table_images_folder)
            )
            
            if enhancement_result and enhancement_result.get("status") == "success":
                final_output_path = enhancement_result.get("output_file", final_output_path)
                logger.info(f"✅ Table enhancement complete (saved locally)")
            else:
                logger.warning("⚠️  Table enhancement failed or skipped, using basic chunks")
        else:
            logger.info("⏭️  Skipping table enhancement (enhance_tables=False)")
        
        # Note: We don't update chunks_file_path in database yet
        # It will be updated with Azure path during weaviate ingestion
        logger.info(f"📦 Chunks saved locally: {final_output_path}")
        logger.info(f"⏭️  Next step: Run weaviate ingestion to upload to Azure")
        
        # Prepare response
        response = {
            "status": "success",
            "message": f"Successfully processed {pdf_count} PDF(s) locally",
            "chunks_file": final_output_path,
            "total_chunks": chunking_result.get("total_chunks", 0),
            "pdf_count": pdf_count,
            "pdf_files": pdf_file_names,
            "tables_enhanced": enhance_tables and (enhancement_result and enhancement_result.get("status") == "success"),
            "folder_path": str(pdf_folder),
            "folder_name": folder_name,
            "already_processed": False
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing PDFs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDFs: {str(e)}"
        )

