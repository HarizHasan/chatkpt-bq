# chatkpt-bq
New revision of ChatKPT leveraging BigQuery ML.

# Variables
Environment variables to be defined via dotenv:

**PROJECT_ID** - Name of Google Cloud Project  
**LOCATION** - Location of Google Cloud resources (Cloud Storage, BigQuery, etc.)  
**DATASET_ID** - BigQuery dataset ID  
**KNOWLEDGE_BASE_TABLE_ID** - ID of table where documents will be loaded.  
**EMBEDDING_TABLE_ID** - ID of table where embeddings will be stored.  
**EMBEDDING_MODEL_ID** - ID of BigQuery ML model that will generate embeddings.  
**GENERATIVE_MODEL_ID** - ID of BigQuery ML model that will generate responses based on RAG retrieval.  
**CONNECTION_NAME** - ID of BigQuery-Vertex AI connection.  
**GCS_BUCKET_NAME** - ID of bucket where documents to be processed are stored.  
**GCS_FOLDER_PATH** - ID of folder in bucket where documents are stored, if any (leave as "" if N/A)  
**EMBEDDING_MODEL_ENDPOINT** - Endpoint ID of Google embedding model that will be used (recommended "text-embedding-005")  
**GENERATIVE_MODEL_ENDPOINT** - Endpoint ID of Google LLM that will be used (recommended "gemini-1.5-flash-002")  
