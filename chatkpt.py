# chatkpt.py

import time
import textwrap
from google.cloud import bigquery
from google.cloud import storage
from google.cloud.exceptions import NotFound
import json
import traceback
import os
import io
from pypdf import PdfReader
import re
import argparse # For command-line arguments for updates
import tiktoken # New import for token-based chunking

# --- Configuration ---
PROJECT_ID = "analytics-team-testing"
LOCATION = "asia-southeast1"
DATASET_ID = "bqml_rag_testing" # Updated for this version
KNOWLEDGE_BASE_TABLE_ID = "documents_chunked"
EMBEDDING_TABLE_ID = "document_embeddings_chunked"
EMBEDDING_MODEL_ID = "text_embedding_model_gcs"
GENERATIVE_MODEL_ID = "gemini_pro_model_gcs"
CONNECTION_NAME = "vertex-ai-connection"

GCS_BUCKET_NAME = "moe_textbook"
GCS_FOLDER_PATH = ""

EMBEDDING_MODEL_ENDPOINT = "text-embedding-005"
GENERATIVE_MODEL_ENDPOINT = "gemini-1.5-flash-002"

TOP_K = 5
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.2

# CHUNK_SIZE and CHUNK_OVERLAP are now in TOKENS
# Old character-based values:
# CHUNK_SIZE = 1000 # Characters
# CHUNK_OVERLAP = 100 # Characters
# New token-based values (example, adjust as needed):
# text-embedding-005 (Gecko) has a max input token limit of 3072.
# Smaller chunks can sometimes yield more precise retrieval.
CHUNK_SIZE = 1024  # Number of tokens per chunk
CHUNK_OVERLAP = 200 # Number of tokens to overlap between chunks
# --- End Configuration -----

# Fully qualified resource names
DATASET_REF = f"{PROJECT_ID}.{DATASET_ID}"
KNOWLEDGE_BASE_TABLE_REF = f"{DATASET_REF}.{KNOWLEDGE_BASE_TABLE_ID}"
EMBEDDING_TABLE_REF = f"{DATASET_REF}.{EMBEDDING_TABLE_ID}"
EMBEDDING_MODEL_REF = f"{DATASET_REF}.{EMBEDDING_MODEL_ID}"
GENERATIVE_MODEL_REF = f"{DATASET_REF}.{GENERATIVE_MODEL_ID}"
CONNECTION_REF = f"{PROJECT_ID}.{LOCATION}.{CONNECTION_NAME}"

# --- Initialize Clients (Globally for the module) ---
try:
    bq_client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    print(f"BigQuery client initialized for project {PROJECT_ID} in location {LOCATION}")
    gcs_client = storage.Client(project=PROJECT_ID)
    print(f"Google Cloud Storage client initialized for project {PROJECT_ID}")
except Exception as e:
    print(f"CRITICAL: Failed to initialize Google Cloud clients: {e}")
    bq_client = None
    gcs_client = None

# --- Helper Functions ---
def escape_sql_string(value: str) -> str:
    if value is None: return "NULL"
    return value.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

def run_bq_query(sql: str, wait: bool = True):
    if not bq_client: raise ConnectionError("BigQuery client not initialized.")
    try:
        query_job = bq_client.query(sql)
        if wait: results = query_job.result();
        return query_job if not wait else results
    except Exception as e:
        print(f"Error executing BQ query: {e}")
        if hasattr(e, 'errors'):
            for error in e.errors: print(f"  Reason: {error.get('reason', 'N/A')}, Message: {error.get('message', 'N/A')}")
        raise

def bq_resource_exists(resource_type: str, resource_full_id: str) -> bool:
    if not bq_client: raise ConnectionError("BigQuery client not initialized.")
    try:
        if resource_type == "dataset": bq_client.get_dataset(resource_full_id)
        elif resource_type == "table": bq_client.get_table(resource_full_id)
        elif resource_type == "model": bq_client.get_model(resource_full_id)
        else: return False
        return True
    except NotFound: return False
    except Exception: return False # More general exception handling

def sanitize_doc_id_for_bq(doc_id: str) -> str:
    sanitized = re.sub(r'[^\w_.-]', '_', doc_id)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized if sanitized else "unknown_doc"

# --- Core RAG Functions ---

def get_relevant_context_with_ids(query: str) -> list[tuple[str, str]]:
    """
    Retrieves relevant document chunks and their IDs from BigQuery.
    Returns: list of tuples, where each tuple is (chunk_doc_id, text_content).
    """
    if not bq_client: raise ConnectionError("BigQuery client not initialized.")
    print(f"RAG: Retrieving context for query: '{query}'")
    escaped_query = escape_sql_string(query)

    if not bq_resource_exists("table", EMBEDDING_TABLE_REF):
        print(f"RAG Error: Embedding table {EMBEDDING_TABLE_REF} does not exist."); return []
    try:
        table = bq_client.get_table(EMBEDDING_TABLE_REF) # Check existence and get metadata
        if table.num_rows == 0:
            print(f"RAG Warning: Embedding table {EMBEDDING_TABLE_REF} is empty."); return []
    except NotFound: # Explicitly catch NotFound if bq_resource_exists somehow failed before or race condition
        print(f"RAG Error: Embedding table {EMBEDDING_TABLE_REF} does not exist (checked again)."); return []
    except Exception as e: print(f"RAG Warning: Could not verify embedding table status: {e}.")


    sql = f"""
    WITH QueryEmbedding AS (
        SELECT ml_generate_embedding_result AS query_vector
        FROM ML.GENERATE_EMBEDDING(MODEL `{EMBEDDING_MODEL_REF}`, (SELECT '{escaped_query}' AS content))
    )
    SELECT
        base.doc_id,
        base.text_content,
        ML.DISTANCE(base.embedding, (SELECT query_vector FROM QueryEmbedding), 'COSINE') AS distance
    FROM
        `{EMBEDDING_TABLE_REF}` AS base
    ORDER BY
        distance ASC
    LIMIT {TOP_K};"""
    try:
        results = run_bq_query(sql, wait=True)
        context_tuples = []
        for row in results:
            context_tuples.append((str(row['doc_id']), str(row['text_content'])))

        print(f"RAG: Retrieved {len(context_tuples)} context chunks with IDs.")
        return context_tuples
    except Exception as e:
        print(f"RAG Error retrieving context chunks: {e}")
        return []

def generate_answer_with_context(query: str, context_chunks_text: list[str]) -> str:
    """
    Generates an answer for a given query utilizing BigQuery ML.
    Returns answer as string.
    """
    if not bq_client: raise ConnectionError("BigQuery client not initialized.")
    print(f"RAG: Generating answer for query: '{query}' with {len(context_chunks_text)} context pieces.")
    prompt_template = """
    You are a helpful assistant. Answer the following question based ONLY on the provided context pieces.
    If the context pieces do not contain the information needed to answer the question, state that clearly. Do not use any prior knowledge.

    Context Pieces:
    ---
    {context_str}
    ---

    Question: {query}
    Answer:"""
    no_context_prompt_template = """
    You are a helpful assistant. Answer the following question.
    If you do not know the answer, state that clearly. Do not use any prior knowledge.

    Question: {query}
    Answer:"""

    if not context_chunks_text:
        prompt = no_context_prompt_template.format(query=query)
    else:
        context_str = "\n\n---\n\n".join(context_chunks_text)
        prompt = prompt_template.format(context_str=context_str, query=query)
    
    escaped_prompt = escape_sql_string(prompt)
    sql = f"""
    SELECT ml_generate_text_result FROM ML.GENERATE_TEXT(
        MODEL `{GENERATIVE_MODEL_REF}`, (SELECT '{escaped_prompt}' AS prompt),
        STRUCT({TEMPERATURE} AS temperature, {MAX_OUTPUT_TOKENS} AS max_output_tokens));"""
    try:
        results = run_bq_query(sql, wait=True)
        answer = ""
        for row in results:
            model_output_data = row.get('ml_generate_text_result') or (row[0] if len(row) > 0 else None)
            if model_output_data and isinstance(model_output_data, dict): # Handle Gemini 1.5 JSON output
                try:
                    # Standard Gemini API output structure
                    candidate = model_output_data.get('candidates', [{}])[0]
                    content = candidate.get('content', {})
                    text_part = content.get('parts', [{}])[0]
                    extracted_text = text_part.get('text', '').strip()
                    if extracted_text: answer = extracted_text; break
                except (IndexError, KeyError, AttributeError) as e_gemini:
                    # Fallback for potentially different structures or older BQML GEN TEXT output
                    print(f"RAG: Gemini output parsing specific error: {e_gemini}. Trying general prediction parsing.")
                    prediction = model_output_data.get('predictions', [{}])[0] # BQML typical
                    extracted_text = prediction.get('content', '').strip() # BQML typical
                    if extracted_text: answer = extracted_text; break
            elif model_output_data and isinstance(model_output_data, str): # Handle PaLM2 string output
                if model_output_data.strip(): answer = model_output_data.strip(); break
        
        final_answer = answer if answer else "The language model did not provide a specific answer based on the context."
        print(f"RAG: Generated answer: '{final_answer[:100]}...'")
        return final_answer
    except Exception as e:
        print(f"RAG EXCEPTION in generate_answer_with_context: {type(e).__name__}: {e}");
        traceback.print_exc()
        return "An error occurred while trying to generate the answer from the language model."

# --- Functions for Database Setup/Update (to be called independently) ---

def setup_bigquery_dataset():
    if not bq_resource_exists("dataset", DATASET_REF):
        print(f"DB Update: Creating dataset {DATASET_REF}...")
        run_bq_query(f"CREATE SCHEMA IF NOT EXISTS `{DATASET_REF}` OPTIONS(location='{LOCATION}');")
    else: print(f"DB Update: Dataset {DATASET_REF} already exists.")

def setup_knowledge_base_table_from_gcs():
    if not bq_client or not gcs_client:
        print("DB Update Error: BQ or GCS client not initialized.")
        return False

    table_exists = bq_resource_exists("table", KNOWLEDGE_BASE_TABLE_REF)
    if not table_exists:
        print(f"DB Update: Creating knowledge base table {KNOWLEDGE_BASE_TABLE_REF}...")
        run_bq_query(f"CREATE TABLE `{KNOWLEDGE_BASE_TABLE_REF}` (doc_id STRING, text_content STRING);")
    else:
        print(f"DB Update: Knowledge base table {KNOWLEDGE_BASE_TABLE_REF} exists. Will append new documents and skip existing ones.")

    processed_original_doc_ids = set()
    if table_exists:
        print("DB Update: Fetching list of already processed original document IDs...")
        # Regex to extract original ID part from 'original_id_chunk_NUMBER'
        sql_get_processed_ids = f"SELECT DISTINCT REGEXP_EXTRACT(doc_id, r'^(.*?)_chunk_\\d+$') FROM `{KNOWLEDGE_BASE_TABLE_REF}` WHERE REGEXP_CONTAINS(doc_id, r'_chunk_\\d+$')"
        try:
            query_job = bq_client.query(sql_get_processed_ids)
            for row in query_job.result():
                if row[0]: processed_original_doc_ids.add(row[0])
            print(f"DB Update: Found {len(processed_original_doc_ids)} distinct original documents already processed.")
        except Exception as e: print(f"DB Update Warning: Could not fetch processed document IDs: {e}.")

    try:
        # Initialize tokenizer once
        # cl100k_base is used by text-embedding-ada-002 and gpt-3.5-turbo, gpt-4
        # It's a good general-purpose tokenizer. Google models (like Gecko/text-embedding-005)
        # have their own tokenization, but cl100k_base is a reasonable approximation.
        tokenizer_encoding = tiktoken.get_encoding("cl100k_base")
        print("DB Update: Tiktoken tokenizer 'cl100k_base' initialized for token-based chunking.")
    except Exception as e:
        print(f"CRITICAL DB Update Error: Failed to initialize tiktoken tokenizer: {e}. Token-based chunking will not be possible.")
        return False

    print(f"DB Update: Populating table {KNOWLEDGE_BASE_TABLE_REF} from GCS: gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}")
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=GCS_FOLDER_PATH)
        
        original_files_processed_count = 0
        total_chunks_inserted_count = 0
        skipped_files_count = 0
        text_extensions = ('.txt', '.md', '.json', '.html', '.py', '.csv')
        pdf_extension = '.pdf'

        for blob in blobs:
            blob_name = blob.name
            if blob_name.endswith('/') and blob_name == GCS_FOLDER_PATH: continue # Skip the folder itself

            gcs_original_doc_id_base = blob_name
            if GCS_FOLDER_PATH and blob_name.startswith(GCS_FOLDER_PATH): # Remove folder prefix if present
                gcs_original_doc_id_base = blob_name[len(GCS_FOLDER_PATH):]
            if not gcs_original_doc_id_base or gcs_original_doc_id_base.endswith('/'): continue # Skip empty names or subfolders
            
            safe_gcs_original_doc_id_base = sanitize_doc_id_for_bq(gcs_original_doc_id_base)

            if safe_gcs_original_doc_id_base in processed_original_doc_ids:
                skipped_files_count += 1
                continue

            full_text_content = None
            blob_name_lower = blob_name.lower()

            if blob_name_lower.endswith(pdf_extension):
                try:
                    pdf_bytes = blob.download_as_bytes()
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    extracted_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                    full_text_content = "\n\n".join(extracted_pages)
                except Exception as e: print(f"    ERROR reading PDF {gcs_original_doc_id_base}: {e}"); continue
            elif blob_name_lower.endswith(text_extensions):
                try: full_text_content = blob.download_as_text(encoding='utf-8')
                except Exception as e: print(f"    ERROR reading TEXT {gcs_original_doc_id_base}: {e}"); continue
            else: continue # Skip unsupported file types

            if full_text_content and full_text_content.strip():
                print(f"  DB Update: Processing new doc: {gcs_original_doc_id_base} with token-based chunking (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
                
                tokens = tokenizer_encoding.encode(full_text_content)
                
                if not tokens:
                    print(f"    WARNING: Document {gcs_original_doc_id_base} resulted in no tokens. Skipping.")
                    continue
                
                if CHUNK_SIZE <= 0: # Should be caught by config validation ideally
                    print(f"    ERROR: CHUNK_SIZE ({CHUNK_SIZE}) for tokenization must be positive. Skipping document {gcs_original_doc_id_base}.")
                    continue 

                start_token_idx = 0
                chunk_number = 0
                chunks_from_this_doc = 0

                while start_token_idx < len(tokens):
                    end_token_idx = min(start_token_idx + CHUNK_SIZE, len(tokens))
                    current_chunk_tokens = tokens[start_token_idx:end_token_idx]
                    
                    if not current_chunk_tokens: # Should generally not happen if CHUNK_SIZE > 0 and loop condition is met
                        break 

                    current_chunk_text = tokenizer_encoding.decode(current_chunk_tokens)

                    if current_chunk_text.strip(): # Process only if the decoded text is not empty
                        chunk_id_for_bq = f"{safe_gcs_original_doc_id_base}_chunk_{chunk_number}"
                        chunk_row = [{"doc_id": chunk_id_for_bq, "text_content": current_chunk_text}]
                        
                        errors = bq_client.insert_rows_json(KNOWLEDGE_BASE_TABLE_REF, chunk_row)
                        if errors: 
                            print(f"    Errors inserting chunk '{chunk_id_for_bq}': {errors}")
                        else: 
                            total_chunks_inserted_count += 1
                            chunks_from_this_doc += 1
                        
                        chunk_number += 1
                    
                    # Determine next starting point, ensuring progress
                    step = CHUNK_SIZE - CHUNK_OVERLAP
                    if step <= 0: # If overlap is too large or CHUNK_SIZE is small
                        step = 1 # Advance by at least one token to prevent infinite loops
                        if CHUNK_SIZE > 0: # Only warn if chunk_size was meant to be effective
                           print(f"    WARNING: CHUNK_OVERLAP ({CHUNK_OVERLAP}) is >= CHUNK_SIZE ({CHUNK_SIZE}). Advancing by 1 token for doc {gcs_original_doc_id_base}.")
                    
                    next_start_token_idx = start_token_idx + step

                    if next_start_token_idx >= len(tokens): # If the next start is beyond or at the end of total tokens
                        break
                    
                    start_token_idx = next_start_token_idx
                
                if chunks_from_this_doc > 0:
                    original_files_processed_count += 1
                    processed_original_doc_ids.add(safe_gcs_original_doc_id_base)

        print(f"DB Update Summary: New Docs Processed: {original_files_processed_count}, New Chunks Inserted: {total_chunks_inserted_count}, Files Skipped (already processed or errors): {skipped_files_count}")
        return True
    except Exception as e:
        print(f"DB Update Error during GCS processing or chunking: {e}"); traceback.print_exc(); return False


def create_bqml_embedding_model():
    print(f"DB Update: Creating/Replacing embedding model {EMBEDDING_MODEL_REF}...")
    run_bq_query(f"CREATE OR REPLACE MODEL `{EMBEDDING_MODEL_REF}` REMOTE WITH CONNECTION `{CONNECTION_REF}` OPTIONS (ENDPOINT = '{EMBEDDING_MODEL_ENDPOINT}');")

def generate_and_store_embeddings():
    if not bq_resource_exists("table", KNOWLEDGE_BASE_TABLE_REF):
         print(f"DB Update Error: Knowledge base table {KNOWLEDGE_BASE_TABLE_REF} not found. Cannot generate embeddings."); return

    print(f"DB Update: Generating embeddings for all chunks (if any new) and replacing table {EMBEDDING_TABLE_REF}...")
    sql = f"""
    CREATE OR REPLACE TABLE `{EMBEDDING_TABLE_REF}` AS
    SELECT base.doc_id, base.content AS text_content, ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING( MODEL `{EMBEDDING_MODEL_REF}`,
        (SELECT doc_id, text_content AS content FROM `{KNOWLEDGE_BASE_TABLE_REF}` 
         WHERE text_content IS NOT NULL AND LENGTH(TRIM(text_content)) > 0)
    ) AS base;"""
    try: run_bq_query(sql, wait=True); time.sleep(5)
    except Exception as e: print(f"DB Update Error: Failed to generate embeddings: {e}")


def create_vector_index():
    if not bq_resource_exists("table", EMBEDDING_TABLE_REF):
         print(f"DB Update Error: Embedding table {EMBEDDING_TABLE_REF} not found. Cannot create index."); return
    try:
        if bq_client.get_table(EMBEDDING_TABLE_REF).num_rows == 0:
            print(f"DB Update Warning: Embedding table {EMBEDDING_TABLE_REF} is empty. Skipping index creation."); return
    except Exception as e:
        print(f"DB Update Warning: Could not verify embedding table status for index creation: {e}. Proceeding cautiously.");

    index_name = sanitize_doc_id_for_bq(f"{EMBEDDING_TABLE_ID}_idx") # Sanitize index name
    print(f"DB Update: Creating/Replacing vector index '{index_name}' on {EMBEDDING_TABLE_REF}...")
    # Note: BQ Index names must be valid identifiers (letters, numbers, underscores)
    sql_index = f"CREATE OR REPLACE VECTOR INDEX `{index_name}` ON `{EMBEDDING_TABLE_REF}`(embedding) OPTIONS(distance_type='COSINE', index_type='IVF');"
    try: run_bq_query(sql_index, wait=False); print(f"DB Update: Vector index '{index_name}' creation initiated. This may take some time.")
    except Exception as e: print(f"DB Update Error: Could not create/replace vector index: {e}")

def create_bqml_generative_model():
    print(f"DB Update: Creating/Replacing generative model {GENERATIVE_MODEL_REF}...")
    run_bq_query(f"CREATE OR REPLACE MODEL `{GENERATIVE_MODEL_REF}` REMOTE WITH CONNECTION `{CONNECTION_REF}` OPTIONS (ENDPOINT = '{GENERATIVE_MODEL_ENDPOINT}');")


def run_database_update_pipeline(full_setup=False):
    """
    Runs the full pipeline to update the BigQuery knowledge base from GCS,
    generate embeddings, and create indexes/models.
    Set full_setup=True for initial one-time creation of dataset and BQML models.
    """
    if not bq_client or not gcs_client:
        print("CRITICAL: BQ or GCS client not initialized. Aborting database update.")
        return

    print("\n--- Starting Knowledge Base Update Pipeline ---")
    start_time = time.time()

    if full_setup:
        setup_bigquery_dataset()
        create_bqml_embedding_model() # Create embedding model before data processing if it's part of full_setup
        create_bqml_generative_model()

    # Process GCS data into knowledge base table
    if setup_knowledge_base_table_from_gcs():
        # Generate embeddings only if data processing was successful
        generate_and_store_embeddings()
        # Create vector index only if embeddings were generated
        create_vector_index()
    else:
        print("Aborting further DB update steps due to errors in GCS processing, tokenizer init, or table setup.")


    end_time = time.time()
    print(f"--- Knowledge Base Update Pipeline Finished in {end_time - start_time:.2f} seconds ---")

def ensure_bqml_models_exist():
    """
    Checks if BQML models exist and creates them if not.
    This is useful for the API to call on startup if models might be missing,
    without running the full data ingestion pipeline.
    """
    if not bq_client:
        print("API Startup Error: BigQuery client not initialized. Cannot ensure BQML models exist.")
        return
    print("API Startup: Ensuring BQML models exist...")
    try:
        if not bq_resource_exists("model", EMBEDDING_MODEL_REF):
            print(f"API Startup: Embedding model {EMBEDDING_MODEL_REF} not found, creating...")
            create_bqml_embedding_model()
        else:
            print(f"API Startup: Embedding model {EMBEDDING_MODEL_REF} found.")

        if not bq_resource_exists("model", GENERATIVE_MODEL_REF):
            print(f"API Startup: Generative model {GENERATIVE_MODEL_REF} not found, creating...")
            create_bqml_generative_model()
        else:
            print(f"API Startup: Generative model {GENERATIVE_MODEL_REF} found.")
    except Exception as e:
        print(f"API Startup Error: Could not ensure BQML models exist: {e}")


# --- Command-line interface for updates ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatKPT Backend: Manage RAG database.")
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Run the full pipeline to update documents from GCS, embeddings, and index.",
    )
    parser.add_argument(
        "--initial-setup",
        action="store_true",
        help="Run initial setup including dataset and BQML model creation. Use with --update-db for the very first run.",
    )
    args = parser.parse_args()

    if bq_client and gcs_client: # Ensure clients are up before running anything
        if args.update_db:
            run_database_update_pipeline(full_setup=args.initial_setup)
        else:
            print("No action specified. Use --update-db to refresh the knowledge base.")
            print("Example: python chatkpt.py --update-db")
            print("Example (first time): python chatkpt.py --update-db --initial-setup")
    else:
        print("CRITICAL: Failed to initialize Google Cloud clients. Cannot perform operations.")
        print("Please check your Google Cloud credentials and project configuration.")