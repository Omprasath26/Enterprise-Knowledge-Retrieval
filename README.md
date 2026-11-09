# Enterprise Knowledge Retrieval & RAG Assistant

A private, localhost-only document question-answering service. Upload internal
documents, ask questions in a chat interface, and receive grounded answers with
document, section, and page citations.

The stack runs entirely on your machine:

- **Streamlit** – ChatGPT-style chat UI for uploads, questions, and status.
- **FastAPI** – validates uploads, enforces a local-only request boundary, and
  serves the ingestion and query APIs.
- **Docling + HybridChunker** – parses and chunks documents.
- **BGE-M3** – generates normalized 1024-dimensional embeddings on CPU.
- **Qdrant** – stores and retrieves vectors with access-tag filtering.
- **SQLite** – tracks documents, versions, and processing state.
- **Redis** – caches eligible answers, isolated by permission and evidence.
- **OpenAI** – used only by the backend to generate grounded, cited answers.

---

## Prerequisites

- Python 3.11 (a Conda environment named `knowledge_retrieval` is used below).
- Docker Desktop with Docker Compose.
- An OpenAI API key stored in the root `.env` file.

The `.env` file is loaded only by the backend and must never be committed. The
key is never sent to Streamlit or returned by the API. BGE-M3 is downloaded on
first use and runs locally on CPU, so allow extra time and disk space for the
initial setup.

---

## Configuration

Create or update the ignored root `.env` file (see `.env.example` for the full
list of options):

```ini
OPENAI_API_KEY=your-key-here
LOCAL_ACCESS_TAGS=personal
MAX_UPLOAD_SIZE_MB=25
QDRANT_URL=http://127.0.0.1:6335
QDRANT_COLLECTION=enterprise_knowledge
REDIS_URL=redis://127.0.0.1:6380/0
EMBEDDING_MODEL=BAAI/bge-m3
OPENAI_MODEL=gpt-4o-mini
RETRIEVAL_TOP_K=5
RELEVANCE_THRESHOLD=0.45
```

`LOCAL_ACCESS_TAGS` defines the server-owned access scope for this single-user
deployment. Access permissions are never accepted from the browser.

---

## Launch the project

All commands are run from the project root in **PowerShell**.

### 1. Create and activate the environment

```powershell
conda create -n knowledge_retrieval python=3.11 -y
conda activate knowledge_retrieval
python -m pip install -r requirements.txt
```

If the environment already exists, activate it directly:

```powershell
conda activate knowledge_retrieval
```

### 2. Start the Qdrant and Redis services

```powershell
docker compose up -d
```

Qdrant is published on `127.0.0.1:6335` and Redis on `127.0.0.1:6380` to avoid
clashing with other local services. Verify they are running with
`docker compose ps`.

### 3. Start the FastAPI backend (terminal 1)

```powershell
conda activate knowledge_retrieval
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

### 4. Start the Streamlit chat UI (terminal 2)

```powershell
conda activate knowledge_retrieval
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

### 5. Open the application

| Service | URL |
| --- | --- |
| Chat UI | `http://127.0.0.1:8501` |
| API docs | `http://127.0.0.1:8000/docs` |
| Health check | `http://127.0.0.1:8000/health` |

The API rejects non-loopback clients, so it must be reached from the same
machine.

---

## Using the application

1. **Upload a document** – open the sidebar, expand **Add documents**, choose an
   access department, and select a file. Supported types are PDF, DOCX, XLSX,
   XLS, HTML, CSV, TXT, and Markdown.
2. **Ask a question** – type in the chat box at the bottom of the page. Answers
   appear as assistant messages with a **Sources** panel listing the document,
   section, and page for each citation.
3. **Check processing status** – expand **Check processing status** and enter a
   document ID to view its state and any parser warnings.

---

## Data and lifecycle

Original files and the SQLite metadata catalog are stored under the ignored
`data/` directory. Each upload records a content hash, a backend-generated
relative storage reference, the original filename, uploader, timestamp, access
department, and processing state.

Ingestion uses Docling with its HybridChunker, followed by normalized
1024-dimensional BGE-M3 embeddings. Qdrant point IDs are deterministic per
document version and chunk. A replacement becomes current only after successful
indexing, and retrieval verifies the SQLite current version and completed state,
so stale points are never returned. Parser and indexing failures are recorded
without exposing local filesystem paths through the API.

Redis exact and semantic cache entries are isolated by the authorized access
scope and retrieved evidence version. Semantic caching never stores not-found
answers. Back up `data/` and retain the Docker volumes needed to rebuild Qdrant;
Redis and Qdrant volumes are convenience storage, not a replacement for backups.

---

## Tests

Run the local unit tests with:

```powershell
conda activate knowledge_retrieval
python -m pip install pytest
python -m pytest -q
```

GitHub Actions compiles the Python modules and runs the same lightweight unit
tests on every push and pull request. It requires neither the OpenAI key nor
local Qdrant/Redis services.

---

## Security scope

This deployment is intended for a single trusted local user. Loopback binding
and fixed server-side tags are not multi-user authentication or network
security. Add and test proper authentication, authorization, and network
controls before sharing the service.
