from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import re
import time

import chatkpt

app = FastAPI(
    title="ChatKPT RAG API",
    description="API for querying a Retrieval-Augmented Generation system.",
    version="1.4.2"
)

# --- Enable CORS ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "https://chatkpt-service-530020478463.asia-southeast1.run.app",
    "http://34.126.113.180",
    "http://35.240.248.248",
    "https://awanlytics.assb-cloud.com",
    "https://awanbotplus.assb-cloud.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic models ---
class QueryRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    id: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list[str]

# --- Helper to extract original doc ID from chunk ID ---
def extract_original_doc_id(chunk_doc_id: str) -> str:
    """
    Extracts the original document ID from a chunk ID.
    Assumes chunk ID format: 'sanitized_original_id_chunk_NUMBER'
    """
    match = re.match(r'^(.*?)_chunk_\d+$', chunk_doc_id)
    if match:
        return match.group(1)
    return chunk_doc_id

# --- API endpoint ---
@app.post("/query", response_model=QueryResponse)
async def handle_query(request_data: QueryRequest):
    """
    Accepts a user query, retrieves relevant context,
    generates an answer using an LLM, and cites sources.
    """
    start_time = time.time()
    print(f"API: Received query: '{request_data.query}'")

    if not chatkpt.bq_client:
        raise HTTPException(status_code=503, detail="Backend service (BigQuery client) not available.")

    try:
        retrieved_chunks_with_ids = chatkpt.get_relevant_context_with_ids(request_data.query)

        if not retrieved_chunks_with_ids:
            print("API: No relevant context found. Generating answer without context.")
            llm_answer = chatkpt.generate_answer_with_context(request_data.query, [])
            return QueryResponse(answer=llm_answer, retrieved_sources=[])

        chunk_texts = [chunk_text for _, chunk_text in retrieved_chunks_with_ids]
        
        original_doc_ids = set()
        for chunk_id, _ in retrieved_chunks_with_ids:
            original_doc_ids.add(extract_original_doc_id(chunk_id))
        
        llm_answer = chatkpt.generate_answer_with_context(request_data.query, chunk_texts)

        end_time = time.time()
        print(f"API: Query processed in {end_time - start_time:.2f} seconds.")

        return QueryResponse(
            answer=llm_answer,
            retrieved_sources=sorted(list(original_doc_ids))
        )

    except ConnectionError as ce:
        print(f"API Error: Backend connection error: {ce}")
        raise HTTPException(status_code=503, detail=f"Backend service connection error: {ce}")
    except Exception as e:
        print(f"API Error: An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Basic health check ---
@app.get("/health")
async def health_check():
    if chatkpt.bq_client and chatkpt.gcs_client:
        return {"status": "healthy", "message": "Backend clients appear initialized."}
    return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "Backend clients not initialized."})

# --- Startup event to ensure BQML models exist ---
@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup...")
    chatkpt.ensure_bqml_models_exist()
    print("Startup checks complete.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
