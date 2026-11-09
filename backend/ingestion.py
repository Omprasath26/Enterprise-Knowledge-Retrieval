from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (HuggingFaceTokenizer)
from docling_core.types.doc import DoclingDocument
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from backend.settings import settings


@dataclass
class Chunk:
    index: int
    text: str
    headings: list[str]
    pages: list[int]


def _xls_to_xlsx(source: Path, target: Path) -> Path:
    import xlrd
    from openpyxl import Workbook

    source_book = xlrd.open_workbook(str(source))
    target_book = Workbook()
    target_book.remove(target_book.active)
    try:
        for source_sheet in source_book.sheets():
            target_sheet = target_book.create_sheet(source_sheet.name[:31])
            for row in range(source_sheet.nrows):
                for column in range(source_sheet.ncols):
                    target_sheet.cell(
                        row=row + 1,
                        column=column + 1,
                        value=source_sheet.cell_value(row, column),
                    )
        target_book.save(target)
    finally:
        target_book.close()
        source_book.release_resources()
    return target


def _extract_chunks(parsed_document: DoclingDocument) -> list[Chunk]:
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        settings.embedding_model, max_tokens=512
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
    )
    chunks = []
    for item in chunker.chunk(parsed_document):
        text = chunker.contextualize(item).strip()
        if not text:
            continue
        meta = getattr(item, "meta", None)
        headings = list(getattr(meta, "headings", None) or [])
        pages = sorted(
            {
                provenance.page_no
                for doc_item in (getattr(meta, "doc_items", None) or [])
                for provenance in (getattr(doc_item, "prov", None) or [])
                if isinstance(getattr(provenance, "page_no", None), int)
            }
        )
        chunks.append(Chunk(len(chunks), text, headings, pages))
    return chunks


def ingest_document(
    *,
    path: Path,
    document_id: str,
    version_hash: str,
    source_filename: str,
    access_tags: tuple[str, ...],
    on_status: Callable[[str], None]) -> tuple[int, list[str]]:
    on_status("parsing")
    converter = DocumentConverter()
    parser_warnings: list[str] = []
    if path.suffix.lower() == ".xls":
        with TemporaryDirectory(prefix="rag-xls-") as temp_dir:
            parse_path = _xls_to_xlsx(path, Path(temp_dir) / "converted.xlsx")
            converted = converter.convert(str(parse_path))
            parser_warnings = [
                type(item).__name__
                for item in (getattr(converted, "errors", None) or [])
            ]
            parsed = DoclingDocument.model_validate(
                converted.document.export_to_dict()
            )
    else:
        converted = converter.convert(str(path))
        parser_warnings = [
            type(item).__name__
            for item in (getattr(converted, "errors", None) or [])
        ]
        parsed = DoclingDocument.model_validate(
            converted.document.export_to_dict()
        )

    on_status("chunking")
    chunks = _extract_chunks(parsed)
    if not chunks:
        raise ValueError("The document contains no indexable text.")

    on_status("embedding")
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    vectors = np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if (
        vectors.shape != (len(chunks), settings.embedding_dimension)
        or not np.isfinite(vectors).all()):
        raise ValueError("BGE-M3 returned invalid vectors; expected 1024 dimensions.")

    on_status("indexing")
    client = QdrantClient(url=settings.qdrant_url)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{document_id}|{version_hash}|{chunk.index}",
                )
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.astype(float).tolist(),
                    payload={
                        "document_id": document_id,
                        "document_version": version_hash,
                        "chunk_id": point_id,
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                        "source_file": source_filename,
                        "headings": chunk.headings,
                        "page_numbers": chunk.pages,
                        "access_tags": list(access_tags),
                    },
                )
            )
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="document_version",
                            match=models.MatchValue(value=version_hash),
                        )
                    ],
                )
            ),
            wait=True,
        )
    finally:
        client.close()
    return len(chunks), parser_warnings
