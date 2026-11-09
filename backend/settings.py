import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _access_tags() -> tuple[str, ...]:
    tags = tuple(
        dict.fromkeys(
            " ".join(value.split()).casefold()
            for value in os.getenv("LOCAL_ACCESS_TAGS", "personal").split(",")
            if value.strip()
        )
    )
    if not tags:
        raise RuntimeError("LOCAL_ACCESS_TAGS must contain at least one tag.")
    return tags


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6335")
    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION", "enterprise_knowledge"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dimension: int = 1024
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    access_tags: tuple[str, ...] = _access_tags()
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    relevance_threshold: float = float(
        os.getenv("RELEVANCE_THRESHOLD", "0.45")
    )


settings = Settings()
