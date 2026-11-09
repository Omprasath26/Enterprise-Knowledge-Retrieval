import ipaddress
import logging
import os
import tempfile
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from backend.metadata import MetadataStore
from backend.settings import settings
from backend.uploads import UploadValidationError, safe_filename, validate_upload

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Enterprise Knowledge Retrieval & RAG Assistant",
    description="Local document ingestion and permission-aware RAG API.",
    version="1.0.0")


@lru_cache(maxsize=1)
def get_metadata() -> MetadataStore:
    return MetadataStore(settings.data_dir / "metadata.sqlite3")


@lru_cache(maxsize=1)
def get_qa_service():
    from backend.qa import QuestionAnswerService

    return QuestionAnswerService(get_metadata())


@app.middleware("http")
async def localhost_only(request: Request, call_next):
    host = request.client.host if request.client else ""
    try:
        is_local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_local = False
    if not is_local:
        return JSONResponse(status_code=403, content={"detail": "Local access only."})
    return await call_next(request)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=10000)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "local-rag"}


@app.get("/api/v1/access-tags")
def access_tags():
    return {
        "access_tags": list(settings.access_tags),
        "max_upload_size_mb": settings.max_upload_size_mb,
    }


@app.post("/api/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    document_id: str | None = Form(default=None)):
    if settings.max_upload_size_mb <= 0:
        raise HTTPException(status_code=500, detail="Upload limit is misconfigured.")
    max_size = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read(max_size + 1)
    await file.close()
    source_filename = safe_filename(file.filename or "")
    normalized_department = " ".join(department.split()).casefold()
    if normalized_department not in settings.access_tags:
        raise HTTPException(status_code=403, detail="Department access is not allowed.")
    if document_id:
        try:
            resolved_document_id = str(UUID(document_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid document ID.") from exc
        existing = get_metadata().get_document(resolved_document_id)
        if existing and existing["department"] not in settings.access_tags:
            raise HTTPException(status_code=404, detail="Document not found.")
    else:
        resolved_document_id = str(uuid4())

    version_hash = sha256(content).hexdigest()
    extension = Path(source_filename).suffix.lower()
    relative_path = (
        Path("documents")
        / resolved_document_id
        / f"{version_hash}{extension}"
    )
    destination = settings.data_dir / relative_path
    metadata = get_metadata()
    metadata.record_upload(
        document_id=resolved_document_id,
        version_hash=version_hash,
        original_filename=source_filename,
        department=normalized_department,
        object_path=relative_path.as_posix(),
        file_type=extension.lstrip("."),
        file_size=len(content),
    )
    metadata.set_status(resolved_document_id, version_hash, "validating")
    temporary_path: Path | None = None
    try:
        try:
            validate_upload(source_filename, content, max_size)
        except UploadValidationError as exc:
            metadata.set_status(
                resolved_document_id, version_hash, "failed", str(exc)
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "document_id": resolved_document_id,
                    "processing_status": "failed",
                    "message": str(exc),
                },
            ) from exc

        try:
            from backend.ingestion import ingest_document
        except ImportError as exc:
            metadata.set_status(
                resolved_document_id,
                version_hash,
                "failed",
                "Ingestion dependencies are unavailable.",
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "document_id": resolved_document_id,
                    "processing_status": "failed",
                    "message": "Document ingestion dependencies are not ready.",
                },
            ) from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".upload-", delete=False
        ) as temp_file:
            temporary_path = Path(temp_file.name)
            temp_file.write(content)
        os.replace(temporary_path, destination)
        temporary_path = None
        metadata.set_status(resolved_document_id, version_hash, "stored")
        chunk_count, parser_warnings = ingest_document(
            path=destination,
            document_id=resolved_document_id,
            version_hash=version_hash,
            source_filename=source_filename,
            access_tags=(normalized_department,),
            on_status=lambda state: metadata.set_status(
                resolved_document_id, version_hash, state
            ),
        )
        metadata.set_parser_warnings(
            resolved_document_id, version_hash, parser_warnings
        )
        metadata.set_status(resolved_document_id, version_hash, "completed")
        return {
            "document_id": resolved_document_id,
            "version": version_hash,
            "processing_status": "completed",
            "chunk_count": chunk_count,
            "parser_warnings": parser_warnings,
        }
    except HTTPException:
        raise
    except Exception as exc:
        metadata.set_status(
            resolved_document_id,
            version_hash,
            "failed",
            f"Processing failed ({type(exc).__name__}); see server logs.",
        )
        logger.exception("Document ingestion failed: %s", resolved_document_id)
        raise HTTPException(
            status_code=422,
            detail={
                "document_id": resolved_document_id,
                "processing_status": "failed",
                "message": "Document processing failed; check its status for details.",
            },
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@app.get("/api/v1/documents/{document_id}")
def document_status(document_id: str):
    try:
        resolved_id = str(UUID(document_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID.") from exc
    document = get_metadata().get_document(resolved_id)
    if not document or document["department"] not in settings.access_tags:
        raise HTTPException(status_code=404, detail="Document not found.")
    document.pop("object_path", None)
    document.pop("department", None)
    return document


@app.post("/api/v1/query")
def query_documents(request: QueryRequest):
    try:
        return get_qa_service().answer(request.question, settings.access_tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=503, detail="The query service could not complete the request."
        ) from exc
