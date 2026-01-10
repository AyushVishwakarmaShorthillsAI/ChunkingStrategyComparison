#!/usr/bin/env python3
"""
DocumentChunker with Docling HierarchicalChunker and advanced filtering
"""
import warnings
import os
import sys
import re
import json
import io
import fitz
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
# from services.azure_storage import storage

# Initialize logger
# logger = get_logger(__name__)


# Suppress all common warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*mode.*parameter is deprecated.*')
warnings.filterwarnings('ignore', message='.*builtin type.*has no __module__ attribute.*')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.chunking import HierarchicalChunker
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️  Docling not available")


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
    print("⚠️  NLTK not available")



class DocumentChunker:
    def __init__(self, chunk_size=1024, chunk_overlap=100, chunking_method="enhanced", use_docling=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method
        self.use_docling = use_docling
        self.pages_with_tables = []
    
        
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required but not available")
        
        try:
            # Option B: Minimal internal threads, parallelize across PDFs instead
            pipeline_options = PdfPipelineOptions(
                accelerator_options=AcceleratorOptions(
                    num_threads=1,  # Minimal per-PDF (external parallelism handles speed)
                    device="cpu"    # Explicit CPU mode
                ),
                do_ocr=False,
                do_table_structure=False,  # Disabled (using LLM for tables)
                generate_picture_images=False
            )
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except Exception as e:
            print(f"⚠️  Could not configure pipeline options: {e}")

            self.converter = DocumentConverter()
        
        self.chunker = HierarchicalChunker(max_tokens=2048)
    
    @staticmethod
    def remove_timestamp_prefix(filename: str) -> str:
        """
        Remove timestamp prefix from filename (e.g., '20251103_082558_Mizuno.pdf' -> 'Mizuno.pdf')
        Timestamp format: YYYYMMDD_HHMMSS_
        """
        import re
        # Match pattern: 8 digits + underscore + 6 digits + underscore
        pattern = r'^\d{8}_\d{6}_'
        return re.sub(pattern, '', filename)

    def extract_pdf_metadata(self, pdf_source, pdf_name: str) -> Dict:
        try:
            if isinstance(pdf_source, (str, Path)):
                doc = fitz.open(pdf_source)
                # Don't overwrite pdf_name - use the passed parameter (original name, not temp file)
            else:
                doc = fitz.open(stream=pdf_source, filetype="pdf")
            
            metadata = doc.metadata
            doc.close()
            
            title = metadata.get('title', '').strip()
            author = metadata.get('author', '').strip()
            mod_date = metadata.get('modDate', '').strip()
            
            if mod_date.startswith('D:'):
                try:
                    mod_date_clean = mod_date[2:16]
                    mod_date = datetime.strptime(mod_date_clean, '%Y%m%d%H%M%S').isoformat()
                except ValueError:
                    mod_date = ''
            
            if not title:
                title = os.path.splitext(pdf_name)[0]
            
            authors = []
            if author:
                author_patterns = [';', ',', ' and ', ' & ']
                author_list = [author]
                for pattern in author_patterns:
                    temp_list = []
                    for auth in author_list:
                        temp_list.extend([a.strip() for a in auth.split(pattern)])
                    author_list = temp_list
                authors = [a for a in author_list if a and len(a) > 1]
            
            return {
                'title': title,
                'authors': authors,
                'pdf_name': pdf_name,
                'modDate': mod_date
            }
        except Exception as e:
            return {
                'title': os.path.splitext(pdf_name)[0],
                'authors': [],
                'pdf_name': pdf_name,
                'modDate': ''
            }

    def _split_into_sentences(self, text):
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

    def _apply_sentence_chunking(self, chunks, max_chars=1024, overlap_sentences=2):
        if not chunks:
            return []
        
        result_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['content']
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                continue
            
            overlap_text = ""
            overlap_from_next_page = False
            next_chunk_pages = []
            
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                next_chunk_text = next_chunk['content']
                next_sentences = self._split_into_sentences(next_chunk_text)
                
                next_chunk_pages = next_chunk['metadata'].get('page', [])
                if not isinstance(next_chunk_pages, list):
                    next_chunk_pages = [next_chunk_pages]
                
                overlap_sentences_list = next_sentences[:overlap_sentences]
                if overlap_sentences_list:
                    overlap_text = " ".join(overlap_sentences_list)
                    
                    current_pages = chunk['metadata'].get('page', [])
                    if not isinstance(current_pages, list):
                        current_pages = [current_pages]
                    
                    if next_chunk_pages and set(next_chunk_pages) != set(current_pages):
                        overlap_from_next_page = True
            
            if overlap_text:
                combined_text = text + " " + overlap_text
                combined_sentences = sentences + self._split_into_sentences(overlap_text)
            else:
                combined_text = text
                combined_sentences = sentences
            
            if len(combined_text) <= max_chars:
                metadata_copy = chunk['metadata'].copy()
                metadata_copy['char_count'] = len(combined_text)
                
                if overlap_from_next_page and next_chunk_pages:
                    current_pages = metadata_copy.get('page', [])
                    if not isinstance(current_pages, list):
                        current_pages = [current_pages]
                    merged_pages = sorted(list(set(current_pages + next_chunk_pages)))
                    metadata_copy['page'] = merged_pages
                
                result_chunks.append({
                    "chunk_id": 0,
                    "content": combined_text,
                    "metadata": metadata_copy
                })
            else:
                metadata_for_split = chunk['metadata'].copy()
                if overlap_from_next_page and next_chunk_pages:
                    current_pages = metadata_for_split.get('page', [])
                    if not isinstance(current_pages, list):
                        current_pages = [current_pages]
                    merged_pages = sorted(list(set(current_pages + next_chunk_pages)))
                    metadata_for_split['page'] = merged_pages
                
                split_chunks = self._split_sentences_with_overlap(
                    combined_sentences, 
                    metadata_for_split,
                    max_chars, 
                    overlap_sentences
                )
                result_chunks.extend(split_chunks)
        
        for idx, chunk in enumerate(result_chunks, 1):
            chunk['chunk_id'] = idx
        
        return result_chunks

    def _split_sentences_with_overlap(self, sentences, metadata, max_chars, overlap_count):
        if not sentences:
            return []
        
        all_text = " ".join(sentences)
        if len(all_text) <= max_chars:
            metadata_copy = metadata.copy()
            metadata_copy['char_count'] = len(all_text)
            return [{
                "chunk_id": 0,
                "content": all_text,
                "metadata": metadata_copy
            }]
        
        first_part_sentences = []
        current_length = 0
        split_index = 0
        
        for idx, sentence in enumerate(sentences):
            sentence_len = len(sentence) + (1 if first_part_sentences else 0)
            
            if current_length + sentence_len > max_chars and first_part_sentences:
                split_index = idx
                break
            
            first_part_sentences.append(sentence)
            current_length += sentence_len
            split_index = idx + 1
        
        if not first_part_sentences and sentences:
            first_part_sentences = [sentences[0]]
            split_index = 1
        
        remaining_sentences = sentences[split_index:]
        
        overlap_added = []
        if remaining_sentences:
            if len(remaining_sentences) >= 2:
                test_overlap = remaining_sentences[:overlap_count]
                test_text = " ".join(first_part_sentences + test_overlap)
                
                if len(test_text) <= max_chars:
                    overlap_added = test_overlap
                else:
                    test_overlap = remaining_sentences[:1]
                    test_text = " ".join(first_part_sentences + test_overlap)
                    
                    if len(test_text) <= max_chars:
                        overlap_added = test_overlap
                    else:
                        if len(first_part_sentences) > 1:
                            moved_sentence = first_part_sentences.pop()
                            overlap_added = [moved_sentence]
                            
                            if remaining_sentences:
                                test_with_next = [moved_sentence] + remaining_sentences[:1]
                                test_text = " ".join(first_part_sentences + test_with_next)
                                if len(test_text) <= max_chars:
                                    overlap_added = test_with_next
            else:
                test_overlap = remaining_sentences[:1]
                test_text = " ".join(first_part_sentences + test_overlap)
                
                if len(test_text) <= max_chars:
                    overlap_added = test_overlap
                else:
                    if len(first_part_sentences) > 1:
                        moved_sentence = first_part_sentences.pop()
                        overlap_added = [moved_sentence]
        
        first_chunk_text = " ".join(first_part_sentences + overlap_added)
        metadata_copy = metadata.copy()
        metadata_copy['char_count'] = len(first_chunk_text)
        
        first_chunk = {
            "chunk_id": 0,
            "content": first_chunk_text,
            "metadata": metadata_copy
        }
        
        if remaining_sentences:
            if overlap_added:
                next_chunk_sentences = overlap_added + remaining_sentences
            else:
                next_chunk_sentences = remaining_sentences
            
            remaining_chunks = self._split_sentences_with_overlap(
                next_chunk_sentences, 
                metadata, 
                max_chars, 
                overlap_count
            )
            return [first_chunk] + remaining_chunks
        else:
            return [first_chunk]

    def _is_watermark_or_copyright(self, text):
        if not text or len(text) < 30:
            return False
        
        text_lower = text.lower()
        watermark_patterns = [
            'downloaded from', 'wiley online library', 'terms and conditions',
            'creative commons license', 'see the terms', 'all rights reserved',
            'copyright ©', '© 20', 'published by', 'journal compilation',
            'doi.org', 'onlinelibrary.wiley.com', 'nature.com',
            'springer.com', 'sciencedirect.com', 'governed by the applicable'
        ]
        
        matches = sum(1 for pattern in watermark_patterns if pattern in text_lower)
        if matches >= 2:
            return True
        
        if 'downloaded from' in text_lower and 'http' in text_lower:
            return True
        if 'terms and conditions' in text_lower and 'http' in text_lower:
            return True
        
        return False

    def _is_at_page_bottom(self, doc_item):
        try:
            if not hasattr(doc_item, 'prov') or not doc_item.prov:
                return False
            for prov in doc_item.prov:
                if hasattr(prov, 'bbox') and prov.bbox:
                    bbox_bottom = prov.bbox.b
                    if bbox_bottom < 80:
                        return True
            return False
        except Exception:
            return False

    def _has_small_font(self, doc_item):
        try:
            if not hasattr(doc_item, 'prov') or not doc_item.prov:
                return False
            for prov in doc_item.prov:
                if hasattr(prov, 'bbox') and prov.bbox:
                    height = prov.bbox.t - prov.bbox.b
                    if height < 10:
                        return True
            return False
        except Exception:
            return False

    def _looks_like_footnote(self, text, doc_item=None):
        if not text or len(text) < 20:
            return False
        
        text_lower = text.lower()
        text_length = len(text)
        is_short_text = text_length < 300
        
        if doc_item and is_short_text:
            at_bottom = self._is_at_page_bottom(doc_item)
            has_small_font = self._has_small_font(doc_item)
            if at_bottom and has_small_font:
                return True
        
        institutional_keywords = [
            'department of', 'division of', 'school of', 'college of',
            'university', 'hospital', 'institute', 'center for', 'centre for',
            'faculty of', 'ministry of', 'national', 'laboratory',
            'service d\'', 'service de', 'département',
            'inserm', 'cnrs', 'umr', 'upmc',
            'radboud', 'massachusetts', 'harvard', 'stanford',
            'medical center', 'children\'s hospital', 'general hospital',
            'boston, ma', 'usa', 'france', 'netherlands', 'germany'
        ]
        
        keyword_count = sum(1 for keyword in institutional_keywords if keyword in text_lower)
        has_address = any(pattern in text_lower for pattern in [', ', '; ', ' usa', ' france', ' china', ' uk'])
        has_numbers = any(char.isdigit() for char in text)
        
        if keyword_count >= 2 and has_address and has_numbers:
            return True
        if keyword_count >= 3 and len(text) > 100:
            return True
        
        return False

    def _parent_cref_str(self, parent_ref):
        try:
            if hasattr(parent_ref, "cref"):
                return str(parent_ref.cref).lower()
            return str(parent_ref).lower()
        except Exception:
            return ""

    def _is_child_of_table_or_picture(self, doc, node):
        try:
            parent_ref = getattr(node, "parent", None)
            if parent_ref is not None:
                cref = self._parent_cref_str(parent_ref)
                if "/table" in cref or "/picture" in cref or "/image" in cref or "/figure" in cref:
                    return True

            for group_name in ("tables", "pictures"):
                if hasattr(doc, group_name) and getattr(doc, group_name):
                    for group in getattr(doc, group_name):
                        if hasattr(group, "children") and group.children:
                            for child in group.children:
                                if hasattr(child, "self_ref") and hasattr(node, "self_ref"):
                                    if child.self_ref == node.self_ref:
                                        return True
                                if hasattr(child, "text") and hasattr(node, "text"):
                                    if str(child.text).strip() == str(node.text).strip() and str(node.text).strip():
                                        return True
        except Exception:
            return False
        return False

    def _is_vertical_bbox(self, prov):
        try:
            if not hasattr(prov, 'bbox') or not prov.bbox:
                return False
            width = prov.bbox.r - prov.bbox.l
            height = prov.bbox.t - prov.bbox.b
            is_narrow = width < 50
            is_tall = height > 600
            is_aspect_vertical = height > (width * 10)
            return is_narrow and is_tall and is_aspect_vertical
        except Exception:
            return False

    def _is_table_chunk(self, doc_items):
        if not doc_items:
            return False
        for doc_item in doc_items:
            label = str(getattr(doc_item, 'label', '')).lower()
            if label == 'table':
                return True
            item_type = type(doc_item).__name__.lower()
            if 'table' in item_type:
                return True
        return False

    def _is_picture_chunk(self, doc_items):
        if not doc_items:
            return False
        for doc_item in doc_items:
            label = str(getattr(doc_item, 'label', '')).lower()
            if label in ['picture', 'image', 'figure']:
                return True
            item_type = type(doc_item).__name__.lower()
            if any(keyword in item_type for keyword in ['picture', 'image', 'figure']):
                return True
        return False

    def _extract_normal_bbox_text(self, item):
        try:
            if not hasattr(item, 'prov') or not item.prov or len(item.prov) <= 1:
                return str(getattr(item, 'text', ''))
            
            full_text = str(getattr(item, 'text', ''))
            normal_parts = []
            
            for prov in item.prov:
                if hasattr(prov, 'charspan') and prov.charspan:
                    start, end = prov.charspan
                    text_part = full_text[start:end]
                    if not self._is_vertical_bbox(prov):
                        normal_parts.append(text_part)
            
            return " ".join(normal_parts) if normal_parts else full_text
        except Exception:
            return str(getattr(item, 'text', ''))

    def _process_converted_doc(self, doc, pdf_metadata):
        table_pages_set = set()
        for table_item in getattr(doc, 'tables', []):
            try:
                if hasattr(table_item, 'prov') and table_item.prov:
                    page_num = table_item.prov[0].page_no
                    table_pages_set.add(page_num)
            except Exception:
                continue
        
        chunks = []
        fig_pattern = re.compile(r'^(Fig\.|Figure)\s+\d+', re.IGNORECASE)
        
        for chunk in self.chunker.chunk(dl_doc=doc):
            if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                if self._is_table_chunk(chunk.meta.doc_items):
                    continue
                if self._is_picture_chunk(chunk.meta.doc_items):
                    continue
            
            filtered_parts = []
            pages = set()
            
            if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                for doc_item in chunk.meta.doc_items:
                    label = str(getattr(doc_item, 'label', '')).lower()
                    item_text = str(getattr(doc_item, 'text', '')).strip()
                    
                    is_watermark = self._is_watermark_or_copyright(item_text)
                    if is_watermark:
                        continue
                    if label == 'caption':
                        continue
                    if label == 'footnote':
                        continue
                    if label in ['picture', 'image', 'figure']:
                        continue
                    if self._is_child_of_table_or_picture(doc, doc_item):
                        continue
                    
                    filtered_text = self._extract_normal_bbox_text(doc_item)
                    if filtered_text.strip():
                        filtered_parts.append(filtered_text.strip())
                    
                    if hasattr(doc_item, 'prov') and doc_item.prov:
                        for prov in doc_item.prov:
                            if hasattr(prov, 'page_no'):
                                pages.add(prov.page_no)
            
            txt = " ".join(filtered_parts) if filtered_parts else ""
            
            if not txt or len(txt.strip()) < 50:
                continue
            if fig_pattern.match(txt.strip()):
                continue
            
            heading = ""
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                if isinstance(chunk.meta.headings, list):
                    heading = chunk.meta.headings[-1] if chunk.meta.headings else ""
                elif isinstance(chunk.meta.headings, str):
                    heading = chunk.meta.headings
            if heading is None:
                heading = ""
            
            page_nums = sorted(list(pages)) if pages else [0]
            
            # Remove timestamp prefix from source name for cleaner display
            clean_source = self.remove_timestamp_prefix(pdf_metadata['pdf_name'])
            
            chunk_dict = {
                "chunk_id": len(chunks) + 1,
                "content": txt,
                "metadata": {
                    "source": clean_source,
                    "title": pdf_metadata['title'],
                    "authors": pdf_metadata['authors'],
                    "page": page_nums,
                    "current_heading": heading,
                    "char_count": len(txt),
                    "mod_date": pdf_metadata['modDate']
                }
            }
            chunks.append(chunk_dict)
        
        print(f"✂️  Applying sentence-aware chunking (max 1024 chars, forward 2-sentence overlap)...")

        initial_count = len(chunks)
        chunks = self._apply_sentence_chunking(chunks, max_chars=1024, overlap_sentences=2)
        print(f"   ✓ Chunks before overlap: {initial_count}")
        print(f"   ✓ Chunks after overlap+splitting: {len(chunks)}")

        
        return {
            'chunks': chunks,
            'table_pages': sorted(list(table_pages_set)),
            'metadata': pdf_metadata
        }

    def process_pdf(self, pdf_source, pdf_name=None):
        if pdf_name is None:
            pdf_name = str(pdf_source)
        print(f"\n📄 Converting: {pdf_name}")

        result = self.converter.convert(source=pdf_source)
        doc = result.document
        
        pdf_metadata = self.extract_pdf_metadata(pdf_source, pdf_name)
        
        return self._process_converted_doc(doc, pdf_metadata)

    def chunk_documents(self, folder_path: str) -> tuple[List, Dict]:
        """
        Returns:
            (all_chunks, pdf_details)
            where pdf_details is {pdf_name: {'table_pages': [1, 2]}}
        """

        print(f"Processing documents in folder: {folder_path}")

        
        pdf_details = {}
        all_chunks = []
        
        # Check if folder_path is a local path
        folder_path_obj = Path(folder_path)
        if folder_path_obj.exists() and folder_path_obj.is_dir():
            # LOCAL MODE: Use local file system
            print(f"📂 Using LOCAL folder: {folder_path}")
            pdf_files = list(folder_path_obj.glob("*.pdf"))

            
            if not pdf_files:
                print("No PDF files found in local folder")
                return [], {}

            
            # Option B: Parallel processing with 1 core per PDF
            from concurrent.futures import ThreadPoolExecutor
            pdf_paths = [str(p) for p in pdf_files]
            
            # Use all available cores (1 PDF per core)
            # Use all available cores (1 PDF per core)
            max_workers = min(os.cpu_count() or 4, len(pdf_paths))
            print(f"🚀 Parallel conversion: {len(pdf_paths)} PDFs using {max_workers} workers (1 core per PDF)")

            
            def process_single_pdf(path):
                """Convert single PDF with error handling."""
                try:
                    return path, self.converter.convert(path, raises_on_error=False)
                except Exception as e:

                    print(f"❌ Error converting {Path(path).name}: {e}")
                    return path, None


            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Use executor.map() for deterministic order (maintains input order)
                    results = list(executor.map(process_single_pdf, pdf_paths))
                
                # Process results in original order
                for pdf_path, conv_res in results:
                    if conv_res is None:
                        continue  # Skip failed conversions
                    
                    try:
                        pdf_name = Path(pdf_path).name
                        
                        if conv_res.document:
                            print(f"✅ Converted: {pdf_name}")

                            
                            # Extract PDF metadata (title, authors, etc.)
                            pdf_metadata = self.extract_pdf_metadata(pdf_path, pdf_name)
                            
                            result = self._process_converted_doc(conv_res.document, pdf_metadata)
                            chunks = result['chunks']
                            table_pages = result['table_pages']
                            
                            all_chunks.extend(chunks)
                            
                            # Store table pages for Step 2
                            # Use clean source name as key to match chunk metadata
                            clean_name = self.remove_timestamp_prefix(pdf_name)
                            pdf_details[clean_name] = {
                                'table_pages': table_pages,
                                'original_filename': pdf_name
                            }
                            
                            print(f"   Processed {len(chunks)} chunks, found tables on pages: {table_pages}")

                        else:
                            # If document is None, it means conversion failed? Or we should check errors
                            if conv_res.errors:
                                error_msg = "; ".join([str(e) for e in conv_res.errors])
                                print(f"❌ Error converting {pdf_name}: {error_msg}")

                            else:
                                print(f"❌ Unknown error converting {pdf_name}")

                                
                    except Exception as e:
                        import traceback
                    except Exception as e:
                        import traceback
                        print(f"❌ Error processing result for {pdf_name if 'pdf_name' in locals() else 'unknown'}: {e}")
                        print(traceback.format_exc())

                        
            except Exception as e:
                import traceback
            except Exception as e:
                import traceback
                print(f"❌ Error during batch conversion: {e}")
                print(traceback.format_exc())

        else:
            print("Azure mode disabled in this script.")
            return [], {}

        
        for idx, chunk in enumerate(all_chunks, 1):
            chunk['chunk_id'] = idx
        
        print(f"✅ Total chunks created: {len(all_chunks)}")

        return all_chunks, pdf_details

    def generate_chunks_from_folder(self, folder_path: str, output_file: str = "chunks.json") -> Optional[Dict]:
        try:
            print(f"📄 Generating chunks JSON from folder: {folder_path}")

            
            chunks, pdf_details = self.chunk_documents(folder_path)
            
            if not chunks:
                print("❌ No chunks generated")
                return None

            
            chunks_data = {
                'document_info': {
                    'processed_at': datetime.now().isoformat(),
                    'total_chunks': len(chunks),
                    'chunking_method': 'Docling HierarchicalChunker',
                    'folder_path': folder_path
                },
                'pdf_details': pdf_details,
                'chunks': chunks
            }
            
            # Save to LOCAL file system (not Azure)
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Successfully generated chunks JSON locally: {output_file}")

            print(f"   File size: {output_path.stat().st_size} bytes")
            return chunks_data
            
        except Exception as e:
            print(f"❌ Error generating chunks JSON: {e}")

            return None
