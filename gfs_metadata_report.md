# Gemini File Search (GFS) Metadata Report

This report summarizes the metadata fields available for extraction from the `grounding_metadata` object returned by the Gemini 2.5 Flash File Search tool.

## 📊 Available Metadata Fields

| Field Name | Description | Example Value |
| :--- | :--- | :--- |
| **`title`** | The name of the source PDF file uploaded to the store. | `"Horn.pdf"` |
| **`text`** | The raw text content of the retrieved chunk used for grounding. | `"Comparison of clinical manifestations..."` |
| **`chunk_indices`** | Index mapping used to link specific segments of the answer to the source chunks. | `[0, 1]` |

## ❌ Missing / Unavailable Fields

| Field Name | Status | Reason |
| :--- | :--- | :--- |
| **`page_number`** | **Missing** | GFS performs internal cloud-side chunking and does not currently expose page-level coordinates. |
| **`uri`** | **Null** | For local File Search Stores, the URI is not populated. |
| **`score`** | **Hidden** | GFS does not provide raw similarity scores for individual grounding chunks in the response. |

## 🔗 Grounding Structure

The metadata follows a two-tier structure:
1.  **`grounding_chunks`**: An array of retrieved text snippets and their source titles.
2.  **`grounding_supports`**: Maps segments of the generated response to the specific chunk(s) that provided the information via indices.

### Summary for Strategy Comparison
While GFS provides excellent semantic grounding and context-aware answers, it lacks the **granular metadata (Page Numbers)** that local chunking pipelines (like Docling + Weaviate) can provide. This makes GFS easier to set up but less transparent for medical or legal audits where page-level verification is required.
