# utils/table_enhancer_wrapper.py
# Wrapper for TableEnhancer functionality

import os
import sys
from pathlib import Path
from typing import Optional, Dict
# from services.azure_storage import storage


# Add the backend directory to path to import TableEnhancer
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from TableEnhancer import enhance_chunks_async
except ImportError:
    logger.info("⚠️  TableEnhancer not found. Please ensure TableEnhancer.py is in the backend directory.")
    enhance_chunks_async = None


async def enhance_table_chunks(chunks_json_path: str, pdf_folder_path: str, images_dir: str = "table_images") -> Optional[Dict]:
    """
    Enhance table chunks in the chunks JSON using Gemini (LOCAL files).
    
    Args:
        chunks_json_path: Path to the LOCAL chunks.json file
        pdf_folder_path: Path to LOCAL folder containing the original PDFs
        images_dir: LOCAL directory to save rendered page images
        
    Returns:
        Dict with enhancement results or None if failed
    """
    if enhance_chunks_async is None:
        return {
            "status": "error",
            "message": "TableEnhancer not available"
        }
    
    try:
        # logger = get_logger(__name__)

        
        print(f"🔍 Starting LOCAL table enhancement for: {chunks_json_path}")
        print(f"   PDF folder: {pdf_folder_path}")
        print(f"   Images dir: {images_dir}")

        
        # Verify chunks file exists locally
        chunks_path = Path(chunks_json_path)
        print(f"📂 Checking if chunks file exists: {chunks_path}")

        if not chunks_path.exists():
            print(f"❌ Chunks file does not exist!")

            return {
                "status": "error",
                "message": f"Local chunks file not found: {chunks_json_path}"
            }
        print(f"✅ Chunks file exists, size: {chunks_path.stat().st_size} bytes")

        
        # Check for PDFs in LOCAL storage
        pdf_folder = Path(pdf_folder_path)
        print(f"📂 Checking if PDF folder exists: {pdf_folder}")

        if not pdf_folder.exists():
            print(f"❌ PDF folder does not exist!")

            return {
                "status": "error",
                "message": f"Local PDF folder not found: {pdf_folder_path}"
            }
        print(f"✅ PDF folder exists")

        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        print(f"📚 Searching for PDFs in: {pdf_folder}")

        if not pdf_files:
            print(f"❌ No PDF files found in folder!")

            return {
                "status": "error",
                "message": f"No PDF files found in local folder: {pdf_folder_path}"
            }
        print(f"✅ Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

        
        # Prepare output path (same location as input, with _tables_enhanced suffix)
        base, ext = os.path.splitext(chunks_json_path)
        output_path = base + '_tables_enhanced' + (ext or '.json')
        print(f"📝 Output path will be: {output_path}")

        
        # Images directory (LOCAL)
        abs_images_dir = images_dir  # Already an absolute path from chunking.py
        print(f"🖼️  Table images will be saved to: {abs_images_dir}")

        
        # Check if images directory exists, create if needed
        images_dir_path = Path(abs_images_dir)
        if not images_dir_path.exists():
            print(f"📁 Creating images directory: {images_dir_path}")

            images_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Run table enhancement
        search_roots = [pdf_folder_path]
        print(f"🔧 Calling enhance_chunks_async with:")
        print(f"   - chunks_path: {chunks_json_path}")
        print(f"   - output_path: {output_path}")
        print(f"   - images_dir: {abs_images_dir}")
        print(f"   - search_roots: {search_roots}")

        
        enhanced_path = await enhance_chunks_async(
            chunks_path=chunks_json_path,
            output_path=output_path,
            images_dir=abs_images_dir,
            search_roots=search_roots
        )
        
        print(f"🔍 enhance_chunks() returned: {enhanced_path}")

        
        if not enhanced_path:
            print(f"❌ enhance_chunks() returned None or empty!")
            print(f"   Expected output file: {output_path}")
            print(f"   Output file exists: {Path(output_path).exists()}")

            return {
                "status": "error",
                "message": "Table enhancement failed to create output file"
            }
        
        # Verify the output file actually exists
        output_file_path = Path(enhanced_path)
        if not output_file_path.exists():
            print(f"❌ enhance_chunks() returned path but file doesn't exist: {enhanced_path}")

            return {
                "status": "error",
                "message": f"Table enhancement returned path {enhanced_path} but file doesn't exist"
            }
        
        print(f"✅ Table enhancement complete: {enhanced_path}")
        print(f"   Output file size: {output_file_path.stat().st_size} bytes")

        
        return {
            "status": "success",
            "message": "Table enhancement completed successfully (local)",
            "output_file": enhanced_path,
            "images_dir": abs_images_dir
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error during table enhancement: {e}")
        print(f"Traceback: {traceback.format_exc()}")

        return {
            "status": "error",
            "message": f"Table enhancement failed: {str(e)}"
        }

