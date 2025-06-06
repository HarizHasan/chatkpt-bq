# main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import re
import time # For timing requests

# Import the backend RAG logic from chatkpt.py
import chatkpt # This will initialize bq_client and gcs_client in chatkpt

app = FastAPI(
    title="ChatKPT RAG API",
    description="API for querying a Retrieval-Augmented Generation system.",
    version="1.0.0"
)

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    query: str
    # You could add more parameters here like user_id for logging, etc.

class SourceDocument(BaseModel):
    id: str # The original document ID (e.g., filename from GCS)
    # score: Optional[float] # If you had relevance scores for chunks

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list[str] # List of unique original document IDs
    # retrieved_chunks_count: int # Optional: number of chunks used

# --- Helper to extract original doc ID from chunk ID ---
def extract_original_doc_id(chunk_doc_id: str) -> str:
    """
    Extracts the original document ID from a chunk ID.
    Assumes chunk ID format: 'sanitized_original_id_chunk_NUMBER'
    """
    match = re.match(r'^(.*?)_chunk_\d+$', chunk_doc_id)
    if match:
        return match.group(1) # The first captured group is the original ID part
    return chunk_doc_id # Fallback to the full chunk_id if pattern doesn't match

# --- API Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def handle_query(request_data: QueryRequest):
    """
    Accepts a user query, retrieves relevant context,
    generates an answer using an LLM, and cites sources.
    """
    start_time = time.time()
    print(f"API: Received query: '{request_data.query}'")

    if not chatkpt.bq_client: # Check if backend clients initialized
        raise HTTPException(status_code=503, detail="Backend service (BigQuery client) not available.")

    try:
        # 1. Retrieve relevant context chunks (tuples of chunk_id, chunk_text)
        retrieved_chunks_with_ids = chatkpt.get_relevant_context_with_ids(request_data.query)

        if not retrieved_chunks_with_ids:
            # Handle case where no context is found - still try to get an answer
            print("API: No relevant context found. Generating answer without context.")
            llm_answer = chatkpt.generate_answer_with_context(request_data.query, [])
            return QueryResponse(answer=llm_answer, retrieved_sources=[])

        # Separate chunk texts and original document IDs
        chunk_texts = [chunk_text for _, chunk_text in retrieved_chunks_with_ids]
        
        # Extract unique original document IDs from chunk IDs for citation
        original_doc_ids = set()
        for chunk_id, _ in retrieved_chunks_with_ids:
            original_doc_ids.add(extract_original_doc_id(chunk_id))
        
        # 2. Generate answer using the context
        llm_answer = chatkpt.generate_answer_with_context(request_data.query, chunk_texts)

        end_time = time.time()
        print(f"API: Query processed in {end_time - start_time:.2f} seconds.")

        return QueryResponse(
            answer=llm_answer,
            retrieved_sources=sorted(list(original_doc_ids)) # Return sorted unique original IDs
        )

    except ConnectionError as ce:
        print(f"API Error: Backend connection error: {ce}")
        raise HTTPException(status_code=503, detail=f"Backend service connection error: {ce}")
    except Exception as e:
        print(f"API Error: An unexpected error occurred: {e}")
        traceback.print_exc() # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Optional: Basic Health Check ---
@app.get("/health")
async def health_check():
    if chatkpt.bq_client and chatkpt.gcs_client:
        # You could add a simple query to BQ here to confirm connectivity
        return {"status": "healthy", "message": "Backend clients appear initialized."}
    return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "Backend clients not initialized."})

# --- Optional: Startup event to ensure BQML models exist ---
@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup...")
    # This ensures the BQML models (not the data) are present.
    # The data itself is updated by running `python chatkpt.py --update-db`
    chatkpt.ensure_bqml_models_exist()
    print("Startup checks complete.")


if __name__ == "__main__":
    # This is for local development.
    # For Cloud Run, you'd use a Procfile or gunicorn command.
    # Example: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
    uvicorn.run(app, host="0.0.0.0", port=8000)