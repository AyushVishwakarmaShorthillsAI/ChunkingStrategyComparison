# utils/document_chunker_wrapper.py
# Wrapper for DocumentChunker functionality

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
# from services.azure_storage import storage


# Add the backend directory to path to import DocumentChunker
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from DocumentChunker import DocumentChunker
except ImportError:
    logger.info("⚠️  DocumentChunker not found. Please ensure DocumentChunker.py is in the backend directory.")
    DocumentChunker = None


def chunk_pdfs_in_folder(pdf_folder_path: str, output_json_path: str) -> Optional[Dict]:
    """
    Chunk all PDFs in the specified LOCAL folder and save to JSON.
    
    Args:
        pdf_folder_path: Path to LOCAL folder containing uploaded PDFs
        output_json_path: Path where chunks JSON should be saved (local)
        
    Returns:
        Dict with chunking results or None if failed
    """
    if DocumentChunker is None:
        return {
            "status": "error",
            "message": "DocumentChunker not available"
        }
    
    try:
        # logger = get_logger(__name__)

        
        print(f"📄 Starting LOCAL PDF chunking for folder: {pdf_folder_path}")

        
        # Check for PDFs in LOCAL storage
        pdf_folder = Path(pdf_folder_path)
        if not pdf_folder.exists():
            return {
                "status": "error",
                "message": f"Local PDF folder not found: {pdf_folder_path}"
            }
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            return {
                "status": "error",
                "message": "No PDF files found in local folder"
            }
        
        pdf_file_names = [f.name for f in pdf_files]
        print(f"📚 Found {len(pdf_files)} PDF file(s): {pdf_file_names}")

        
        # Initialize DocumentChunker with optimal settings
        chunker = DocumentChunker(
            chunk_size=1024,
            chunk_overlap=100,
            chunking_method="enhanced",
            use_docling=True
        )
        
        # Generate chunks JSON
        chunks_data = chunker.generate_chunks_from_folder(
            folder_path=pdf_folder_path,
            output_file=output_json_path
        )
        
        if chunks_data is None:
            return {
                "status": "error",
                "message": "Failed to generate chunks. Check backend logs for detailed error information. This may be due to: 1) PDF parsing failure, 2) Docling initialization issue, 3) Missing dependencies, or 4) Corrupted PDF files."
            }
        
        # Return success with stats
        return {
            "status": "success",
            "message": f"Successfully chunked {len(pdf_files)} PDF(s) locally",
            "output_file": output_json_path,
            "total_chunks": chunks_data['document_info']['total_chunks'],
            "pdf_count": len(pdf_files),
            "pdf_files": pdf_file_names
        }
        
    except Exception as e:
        print(f"❌ Error during chunking: {e}")

        return {
            "status": "error",
            "message": f"Chunking failed: {str(e)}"
        }

