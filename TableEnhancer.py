#!/usr/bin/env python3
"""
TableEnhancer: Extract tables using Gemini with semantic descriptive format
"""
import os
import sys
import io
import time
import json
import asyncio
import functools
from pathlib import Path
from typing import Dict, Any, List, Optional
# from services.azure_storage import storage


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import fitz
from PIL import Image
# Initialize logger
# logger = get_logger(__name__)



from google import genai
from google.genai import types

# Configure Google GenAI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = None

if not GOOGLE_API_KEY:
    print("⚠️  GOOGLE_API_KEY not set. TableEnhancer will fail.")
else:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("✅ Google GenAI client configured for TableEnhancer")


try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Global rate limiter for Gemini API
_rate_limit_lock = asyncio.Lock()
_last_rate_limit_time = 0

# Gemini concurrency limit (configurable via env)
GEMINI_CONCURRENCY = int(os.getenv('GEMINI_CONCURRENCY', '5'))  # Conservative default for free tier

# Image resample filter (handle Pillow version compatibility)
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # Pillow 10.0+
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS  # Older Pillow


def remove_timestamp_prefix(filename: str) -> str:
    """
    Remove timestamp prefix from filename (e.g., '20251103_082558_Mizuno.pdf' -> 'Mizuno.pdf')
    Timestamp format: YYYYMMDD_HHMMSS_
    """
    import re
    # Match pattern: 8 digits + underscore + 6 digits + underscore
    pattern = r'^\d{8}_\d{6}_'
    return re.sub(pattern, '', filename)


def split_text_into_sentences(text):
    if not text:
        return []
    
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            pass
    
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and current.strip():
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def chunk_table_text_with_overlap(text, max_chars=1024, overlap_sentences=2):
    sentences = split_text_into_sentences(text)
    
    if not sentences:
        return [text] if text else []
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence) + (1 if current_chunk_sentences else 0)
        
        if current_length + sentence_len > max_chars and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
            
            if len(current_chunk_sentences) >= overlap_sentences:
                overlap = current_chunk_sentences[-overlap_sentences:]
                current_chunk_sentences = overlap
                current_length = sum(len(s) + 1 for s in overlap)
            else:
                current_chunk_sentences = []
                current_length = 0
        
        current_chunk_sentences.append(sentence)
        current_length += sentence_len
    
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append(chunk_text)
    
    return chunks


def init_gemini():
    """Initialize Google GenAI client"""
    if not GOOGLE_API_KEY:
        print("⚠️  GOOGLE_API_KEY not set")
        return None
    
    return client


def render_page_image(pdf_source, pdf_name, page_number, output_dir="table_images", dpi=300):
    """Synchronous version of page rendering"""
    base_name = os.path.splitext(pdf_name)[0]
    blob_path = f"{output_dir}/{base_name}_p{page_number}.png"
    
    if os.path.exists(blob_path):
        return blob_path

    
    try:
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(stream=pdf_source, filetype="pdf") if hasattr(pdf_source, 'read') else fitz.open(pdf_source)
        
        page_index = page_number - 1
        
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            return None
        
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img_bytes = pix.tobytes("png")
        doc.close()
        
        with open(blob_path, 'wb') as f:
            f.write(img_bytes)

        print(f"📄 Rendered page {page_number}: {blob_path}")
        return blob_path
    except Exception as e:
        print(f"❌ Failed to render page {page_number}: {e}")
        return None


async def render_page_image_async(pdf_bytes, pdf_name, page_number, output_dir="table_images", dpi=300):
    """Async wrapper for page rendering (runs in executor)"""
    loop = asyncio.get_running_loop()
    # Pass partial function to run_in_executor
    # Note: We pass pdf_bytes to avoid pickling issues with file objects if any
    return await loop.run_in_executor(
        None, 
        functools.partial(render_page_image, pdf_bytes, pdf_name, page_number, output_dir, dpi)
    )


async def extract_table_with_gemini_async(image_path, page_number, retries=3):

    """Async version of Gemini extraction using google-genai"""
    global _last_rate_limit_time
    
    if not client or not image_path:
        return None
    
    try:
        loop = asyncio.get_event_loop()
        img_bytes = await loop.run_in_executor(None, lambda: open(image_path, 'rb').read())
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize if needed
        max_dimension = 3072
        current_max = max(img.size)
        if current_max > max_dimension:
            ratio = max_dimension / current_max
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            print(f"   📐 Resizing page {page_number}: {img.size} → {new_size}")
            img = await loop.run_in_executor(None, lambda: img.resize(new_size, RESAMPLE_FILTER))
        
    except Exception as e:
        print(f"❌ Failed to open/process image {image_path}: {e}")
        return None
    
    prompt = """You are a research paper assistant. This page contains a table. Extract and describe the table content in a format optimized for semantic search and RAG.

**Instructions:**
1. **Table Title/Caption**: State what the table is about
2. **Table Purpose**: Briefly explain what information this table presents
3. **Table Structure**: Describe the columns and what they represent
4. **Table Data**: Present the data in natural language paragraphs (NOT as a raw table)
   - Group related data together
   - Use complete sentences
   - Include all values, measurements, and statistics
5. **Key Findings**: Highlight important patterns, trends, or significant values
6. **Context**: Mention any notes, footnotes, or additional context

**Important:**
- Write in complete, searchable sentences
- Include ALL data points from the table
- Make it easy to find specific information through text search
- Don't use markdown tables or special formatting
- Be thorough and detailed"""
    
    model_name = "gemini-2.5-flash"

    for attempt in range(1, retries + 1):
        try:
            # Global rate limit check
            async with _rate_limit_lock:
                if _last_rate_limit_time > 0:
                    elapsed = time.time() - _last_rate_limit_time
                    if elapsed < 60:
                        wait_time = 60 - elapsed
                        print(f"⏳ Global rate limit cooldown: {wait_time:.1f}s (page {page_number})")
                        await asyncio.sleep(wait_time)

            # Call Gemini via google-genai
            # Convert image to bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            final_img_bytes = buf.getvalue()

            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_bytes(data=final_img_bytes, mime_type="image/png")
                            ]
                        )
                    ]
                )
            )
            
            text = response.text.strip() if response.text else ""
            if text:
                print(f"✅ Extracted table from page {page_number} ({len(text)} chars)")
                return text
            
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'quota' in error_str or 'resource exhausted' in error_str:
                async with _rate_limit_lock:
                    _last_rate_limit_time = time.time()
                print(f"🚨 RATE LIMIT hit on page {page_number} - All tasks will pause")

                if attempt < retries:
                    await asyncio.sleep(60)
                    continue
            else:
                print(f"⚠️  Gemini extraction error (page {page_number}): {e}")
        
        if attempt < retries:
            await asyncio.sleep(attempt * 1)
    
    return None


async def extract_table_chunks_for_pdf_async(pdf_bytes, pdf_name, table_pages, pdf_metadata, images_dir="table_images", semaphore=None):
    if not table_pages:
        return []
    
    print(f"\n🔄 Extracting tables from {len(table_pages)} pages in {pdf_name} (async)...")
    
    if not client:
        print("❌ Google GenAI client not initialized for TableEnhancer")
        return []

    
    if semaphore is None:
        semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)  # Use global config
    
    async def process_page(page_num):
        async with semaphore:
            # 1. Render Image (CPU/IO)
            image_path = await render_page_image_async(pdf_bytes, pdf_name, page_num, images_dir)
            if not image_path:
                return []
            
            # 2. Extract with Gemini (Net IO)
            table_text = await extract_table_with_gemini_async(image_path, page_num)

            if not table_text:
                return []
            
            # 3. Chunk text (CPU)
            print(f"✂️  Splitting table text for page {page_num}...")
            text_chunks = chunk_table_text_with_overlap(table_text, max_chars=1024, overlap_sentences=2)
            
            chunks_data = []
            for idx, chunk_text in enumerate(text_chunks):
                heading = f"Table from page {page_num}" + (f" (part {idx+1}/{len(text_chunks)})" if len(text_chunks) > 1 else "")
                clean_source = remove_timestamp_prefix(pdf_metadata['pdf_name'])
                
                table_chunk = {
                    "chunk_id": 0,
                    "content": chunk_text,
                    "metadata": {
                        "source": clean_source,
                        "title": pdf_metadata['title'],
                        "authors": pdf_metadata['authors'],
                        "page": [page_num],
                        "current_heading": heading,
                        "char_count": len(chunk_text),
                        "mod_date": pdf_metadata['modDate']
                    }
                }
                chunks_data.append(table_chunk)
            return chunks_data

    # Gather all page tasks
    tasks = [process_page(p) for p in table_pages]
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    table_chunks = [chunk for sublist in results for chunk in sublist]
    print(f"   ✓ Extracted total {len(table_chunks)} chunks from {pdf_name}")
    return table_chunks


async def enhance_chunks_async(chunks_path: str, output_path: Optional[str] = None, images_dir: str = 'table_images', search_roots: Optional[List[str]] = None) -> str:
    # Read from LOCAL file system
    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        return None
    
    loop = asyncio.get_running_loop()
    # Async file read (use executor for file IO)
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data.get('chunks', [])
    if not isinstance(chunks, list):
        raise ValueError("Invalid chunks.json format")

    if not output_path:
        chunks_path_norm = chunks_path.replace('\\', '/')
        base, ext = os.path.splitext(chunks_path_norm)
        output_path = base + '_tables_enhanced' + (ext or '.json')

    text_chunks = chunks
    all_table_chunks = []
    
    # Use search_roots (local PDF folders)
    pdf_files = []
    if search_roots:
        for search_root in search_roots:
            search_path = Path(search_root)
            if search_path.exists() and search_path.is_dir():
                pdf_files.extend(list(search_path.glob("*.pdf")))
    
    if not pdf_files:
        print("⚠️  No PDF files found in local folders")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path
    

    print(f"🚀 Starting async table enhancement for {len(pdf_files)} PDFs...")
    # Global semaphore for page-level concurrency (Gemini limits)
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY) 
    
    # Get pdf_details from chunks_data if available
    pdf_details_map = data.get('pdf_details', {})
    
    async def process_pdf_async(pdf_file) -> List[Dict]:
        pdf_name = pdf_file.name
        clean_name = os.path.splitext(pdf_name)[0] # Or however DocumentChunker cleaned it?
        # DocumentChunker uses remove_timestamp_prefix. Let's try direct match or clean match.
        # But here pdf_name has extension. chunks.json keys might depend on logic.
        # DocumentChunker.remove_timestamp_prefix was used for the key.
        # We need to replicate that or do a best-effort match.
        
        # Let's try to find key in pdf_details_map
        # logic: remove timestamp pattern if matches
        def remove_timestamp_prefix(filename):
            import re
            pattern = r'^\d{8}_\d{6}_'
            return re.sub(pattern, '', filename)
            
        key_name = remove_timestamp_prefix(pdf_name)
        
        local_chunks = []
        try:
            # Check if we have pre-computed table pages
            table_pages = []
            if key_name in pdf_details_map:
                table_info = pdf_details_map[key_name]
                table_pages = table_info.get('table_pages', [])
                print(f"✅ Found pre-computed tables for {pdf_name}: {table_pages}")
            else:
                # Fallback: Run Docling (only if metadata missing)
                # This should rarely happen if chunks.json came from updated DocumentChunker
                print(f"⚠️  No pre-computed tables for {pdf_name}, running fallback Docling check...")
                
                from DocumentChunker import DocumentChunker
                def find_tables():
                    chunker = DocumentChunker()
                    return chunker.process_pdf(str(pdf_file), pdf_name)
                
                result = await loop.run_in_executor(None, find_tables)
                table_pages = result['table_pages']

            if not table_pages:
                return []

            # We still need basic metadata for the new chunks
            # Async file read
            pdf_bytes = await loop.run_in_executor(None, lambda: open(pdf_file, 'rb').read())
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            metadata = doc.metadata
            doc.close()
            
            title = metadata.get('title', '').strip() or clean_name
            mod_date = metadata.get('modDate', '').strip()
            author = metadata.get('author', '').strip()
            
            authors = []
            if author:
                authors = [a.strip() for a in author.split(';') if a.strip()]

            pdf_metadata = {
                'title': title,
                'authors': authors,
                'pdf_name': pdf_name,
                'modDate': mod_date
            }
            
            chunks = await extract_table_chunks_for_pdf_async(
                pdf_bytes,
                pdf_name,
                table_pages, 
                pdf_metadata,
                images_dir,
                semaphore
            )
            if chunks:
                local_chunks.extend(chunks)
        
        except Exception as e:
            print(f"⚠️  Error processing {pdf_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return local_chunks

    # Launch all PDF tasks concurrently
    tasks = [process_pdf_async(pdf) for pdf in pdf_files]
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    for res in results:
        all_table_chunks.extend(res)

    
    # Re-index chunks
    start_id = len(text_chunks) + 1
    for idx, chunk in enumerate(all_table_chunks, start_id):
        chunk['chunk_id'] = idx
    
    final_chunks = text_chunks + all_table_chunks
    
    data['document_info']['total_chunks'] = len(final_chunks)
    data['document_info']['text_chunks'] = len(text_chunks)
    data['document_info']['table_chunks'] = len(all_table_chunks)
    data['chunks'] = final_chunks

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Enhanced chunks saved to: {output_path}")
    print(f"📊 Enhancement Summary:")
    print(f"   • Text chunks: {len(text_chunks)}")
    print(f"   • Table chunks: {len(all_table_chunks)}")
    print(f"   • Total chunks: {len(final_chunks)}")
    
    return output_path


def enhance_chunks(chunks_path: str, output_path: Optional[str] = None, images_dir: str = 'table_images', search_roots: Optional[List[str]] = None) -> str:
    """Synchronous entry point that runs the async implementation"""
    try:
        # Check if there's a running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # Optimization: If called from async context, we ideally await
        # But for compatibility with sync wrapper, we can't await here directly without changing signature
        # This is a risk. If this falls here, it means we are blocking the loop with a sync wrapper?
        # Ideally, we should use nesting or just warn
        print("⚠️  Warning: enhance_chunks called synchronously from running loop. Blocking execution.")
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(enhance_chunks_async(chunks_path, output_path, images_dir, search_roots))
    else:
        return asyncio.run(enhance_chunks_async(chunks_path, output_path, images_dir, search_roots))

