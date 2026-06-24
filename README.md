# Document Ingestion & Conversational RAG API

A backend system that ingests documents, enables semantic search via RAG, and supports multi-turn conversational interviews with booking capability.

## Overview

**Phase 1: Document Ingestion Pipeline**
- Upload PDF or TXT files
- Automatically extract text
- Apply two chunking strategies: fixed-size (overlap-based) or semantic (sentence-aware via NLTK)
- Generate 384-dimensional embeddings using SentenceTransformers
- Store vectors in Qdrant, metadata in SQLite

**Phase 2: Conversational RAG with Interview Booking**
- Multi-turn conversations with chat history via Redis
- Semantic search: retrieve relevant document chunks for user queries
- LLM-powered responses using Ollama (Mistral 7B)
- Automatic booking extraction: parse name, email, date, time from conversation
- Persistent storage of conversations & bookings in SQLite

## Tech Stack

| Component          | Choice                                     |  
|--------------------|--------------------------------------------|
| **API Framework**  | FastAPI 0.138.0+                           |
| **LLM**            | Ollama + Mistral 7B                        |  
| **Embeddings**     | SentenceTransformers (`all-MiniLM-L6-v2`)  |  
| **Vector DB**      | Qdrant (embedded)                          |
| **SQL DB**         | SQLite + SQLAlchemy                        |   
| **Chat Memory**    | Redis                                      |
| **Text Processing**| PyPDF2, NLTK                               |

## Setup & Installation

### Prerequisites
- Python 3.12+
- Docker (for Redis)
- Ollama installed and running

### 1. Clone & Install

```bash
git clone https://github.com/Yokaiiiii/document_ingestion_api.git
cd document_ingestion_api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
# Alternative with uv:
# uv pip install -e .
```

### 2. Environment Setup

```bash
cp .env.example .env
```

Ensure `.env` contains the following (adjust as needed):

```env
APP_NAME=Document Ingestion API
DATABASE_URL=sqlite:///./app.db
UPLOAD_DIR=storage/uploads
QDRANT_PATH=./storage/qdrant_data
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL_NAME=mistral
REDIS_URL=redis://localhost:6379
```

### 3. Start External Services

**Terminal 1 — Ollama:**

```bash
ollama serve
```

In another terminal, pull the model (one-time, ~4.4GB):

```bash
ollama pull mistral
```

**Terminal 2 — Redis:**

```bash
docker-compose up -d redis
# Verify:
docker exec -it rag_redis_cache redis-cli ping
```

**Terminal 3 — FastAPI App:**

```bash
uvicorn app.main:app --reload
```

The app will be available at **http://localhost:8000**

- Interactive API docs: http://localhost:8000/docs (Swagger UI)
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health & Status
- `GET /health` — Server health check

### Documents
- `POST /api/v1/documents/upload` — Upload PDF/TXT with chunking strategy (`fixed_size` or `semantic`)
- `GET /api/v1/documents` — List all documents (newest first)
- `GET /api/v1/documents/{document_id}` — Get document details + all chunks
- `DELETE /api/v1/documents/{document_id}` — Delete document, chunks, and vectors (cascading delete)

### Chat & Booking
- `POST /api/v1/chat` — Send message and receive LLM response + extracted booking (if any)

  **Request body:**
  ```json
  {
    "conversation_id": null | string,
    "message": "string"
  }
  ```

  **Response:**
  ```json
  {
    "conversation_id": "string",
    "assistant_message": "string",
    "booking": { ... } | null,
    "context_used": ["vector_id1", ...]
  }
  ```

## How It Works

### Document Upload Flow

```
User File → Text Extraction → Chunking (fixed-size or semantic) → 
Embedding (384-dim) → Store in Qdrant + SQLite → Return document record
```

### Chat Flow

```
User Message → Load history (Redis) → RAG Retrieval (Qdrant) → 
Context Formatting → LLM Prompt Construction → Ollama Inference → 
Booking Extraction → Persist to SQLite + Redis → Return response
```

## Key Features

- **Smart Chunking**: Semantic (NLTK sentence boundaries) or fixed-size (500 chars + 50 char overlap)
- **Relevance Filtering**: Only includes chunks with similarity score > 0.3
- **Conversation Memory**: Redis with 24h TTL + full archive in SQLite
- **Booking Extraction**: LLM-powered parsing of name, email, date, and time
- **Local-first**: Everything runs locally with no external API costs

## Project Structure

```bash
app/
├── main.py          # FastAPI app & routes
├── config.py        # Pydantic settings
├── database.py      # SQLAlchemy models (Document, Chunk, etc.)
├── schemas.py       # Pydantic models
├── services.py      # Ingestion pipeline
├── rag.py           # Retrieval & context building
├── llm.py           # Ollama client + booking extraction
├── chat.py          # Chat orchestration
├── memory.py        # Redis memory management
└── ...

storage/
└── uploads/         # Uploaded files (gitignored)

docker-compose.yml
pyproject.toml
.env.example
```

## Testing the System

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample.pdf" \
  -F "chunking_strategy=semantic"
```

### 2. Start a Conversation

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": null,
    "message": "What is this document about?"
  }'
```

### 3. Book an Interview

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "prev-conv-id",
    "message": "I am John Doe, my email is john@example.com. I want to book for 2026-06-25 at 14:00"
  }'
```

### 4. Inspect Redis

```bash
docker exec -it rag_redis_cache redis-cli
KEYS *
LRANGE chat_memory:<conversation_id> 0 -1
TTL chat_memory:<conversation_id>
```

## Known Limitations & Future Improvements

**Current Limitations:**
- Mistral 7B has limited reasoning depth
- SQLite and embedded Qdrant are MVP-only
- No authentication or pagination
- Synchronous document processing

**Future Enhancements:**
- Async processing with Celery
- User authentication & multi-tenancy
- Email notifications for bookings
- Larger LLMs (Llama 3, Claude, etc.)
- PostgreSQL + production Qdrant
- Conversation history retrieval endpoint

## License

N/A (Assignment submission)