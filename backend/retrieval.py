import re
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from backend.metadata import MetadataStore
from backend.settings import settings


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    document_id: str
    version: str
    text: str
    source_file: str
    headings: list[str]
    pages: list[int]
    score: float


class Retriever:
    def __init__(self, metadata: MetadataStore):
        self.metadata = metadata
        self.model = SentenceTransformer(settings.embedding_model, device="cpu")
        self.client = QdrantClient(url=settings.qdrant_url)

    @staticmethod
    def normalize_question(question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")
        return re.sub(r"\s+", " ", question).strip()

    def embed_question(self, question: str) -> list[float]:
        normalized = self.normalize_question(question)
        vector = np.asarray(
            self.model.encode(
                [normalized],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if (
            vector.shape != (1, settings.embedding_dimension)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("BGE-M3 returned an invalid query embedding.")
        return vector[0].astype(float).tolist()

    def search(self, embedding: list[float], access_tags: tuple[str, ...]) -> list[Evidence]:
        if not access_tags:
            return []
        active_versions = self.metadata.current_completed_versions(access_tags)
        if not active_versions:
            return []
        result = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="access_tags",
                        match=models.MatchAny(any=list(access_tags)),
                    )
                ]
            ),
            limit=settings.top_k,
            score_threshold=settings.relevance_threshold,
            with_payload=True,
            with_vectors=False,
        )
        evidence = []
        for point in result.points:
            payload = point.payload or {}
            document_id = payload.get("document_id")
            version = payload.get("document_version")
            chunk_id = payload.get("chunk_id") or str(point.id)
            source = payload.get("source_file")
            text = payload.get("text")
            try:
                valid_hash = (
                    isinstance(version, str)
                    and len(version) == 64
                    and int(version, 16) >= 0
                )
            except ValueError:
                valid_hash = False
            tags = payload.get("access_tags")
            if (
                not isinstance(document_id, str)
                or (document_id, version) not in active_versions
                or not valid_hash
                or not isinstance(chunk_id, str)
                or not chunk_id
                or not isinstance(source, str)
                or not source
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(tags, list)
                or not set(tags).intersection(access_tags)
            ):
                continue
            evidence.append(
                Evidence(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=version,
                    text=text,
                    source_file=source,
                    headings=payload.get("headings") or [],
                    pages=payload.get("page_numbers") or [],
                    score=float(point.score),
                )
            )
        return evidence

    def close(self) -> None:
        self.client.close()


def evidence_fingerprint(evidence: list[Evidence]) -> str:
    material = sorted(
        (item.document_id, item.version, item.chunk_id) for item in evidence
    )
    return sha256(repr(material).encode("utf-8")).hexdigest()


def permission_fingerprint(access_tags: tuple[str, ...]) -> str:
    return sha256(repr(sorted(set(access_tags))).encode("utf-8")).hexdigest()
