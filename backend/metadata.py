import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetadataStore:
    """Small SQLite catalog for document identity, versions, and lifecycle."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    department TEXT NOT NULL,
                    uploader TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    current_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL,
                    version_hash TEXT NOT NULL,
                    object_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    failure_reason TEXT,
                    parser_warnings TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (document_id, version_hash),
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(document_versions)"
                ).fetchall()
            }
            if "parser_warnings" not in columns:
                connection.execute(
                    "ALTER TABLE document_versions "
                    "ADD COLUMN parser_warnings TEXT NOT NULL DEFAULT '[]'"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def record_upload(
        self,
        *,
        document_id: str,
        version_hash: str,
        original_filename: str,
        department: str,
        object_path: str,
        file_type: str,
        file_size: int,
        uploader: str = "local-user") -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (document_id, original_filename, department, uploader,
                     created_at, current_version)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    original_filename = excluded.original_filename,
                    department = excluded.department,
                    uploader = excluded.uploader,
                    current_version = excluded.current_version
                """,
                (
                    document_id,
                    original_filename,
                    department,
                    uploader,
                    now,
                    version_hash,
                ),
            )
            connection.execute(
                """
                INSERT INTO document_versions
                    (document_id, version_hash, object_path, file_type,
                     file_size, uploaded_at, processing_status)
                VALUES (?, ?, ?, ?, ?, ?, 'uploaded')
                ON CONFLICT(document_id, version_hash) DO UPDATE SET
                    object_path = excluded.object_path,
                    uploaded_at = excluded.uploaded_at,
                    processing_status = 'uploaded',
                    failure_reason = NULL,
                    parser_warnings = '[]'
                """,
                (
                    document_id,
                    version_hash,
                    object_path,
                    file_type,
                    file_size,
                    now,
                ),
            )

    def set_status(
        self,
        document_id: str,
        version_hash: str,
        status: str,
        failure_reason: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_versions
                SET processing_status = ?, failure_reason = ?
                WHERE document_id = ? AND version_hash = ?
                """,
                (status, failure_reason, document_id, version_hash),
            )

    def set_parser_warnings(self, document_id: str, version_hash: str, warnings: list[str]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_versions
                SET parser_warnings = ?
                WHERE document_id = ? AND version_hash = ?
                """,
                (json.dumps(warnings), document_id, version_hash),
            )

    def current_completed_versions(self, allowed_tags: tuple[str, ...]) -> set[tuple[str, str]]:
        if not allowed_tags:
            return set()
        placeholders = ",".join("?" for _ in allowed_tags)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT d.document_id, d.current_version
                FROM documents AS d
                JOIN document_versions AS v
                  ON v.document_id = d.document_id
                 AND v.version_hash = d.current_version
                WHERE d.department IN ({placeholders})
                  AND v.processing_status = 'completed'
                """,
                allowed_tags,
            ).fetchall()
        return {(row["document_id"], row["current_version"]) for row in rows}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT d.document_id, d.original_filename, d.department,
                       d.current_version, v.file_type, v.file_size,
                       v.object_path, v.uploaded_at, v.processing_status,
                       v.failure_reason, v.parser_warnings
                FROM documents AS d
                JOIN document_versions AS v
                  ON v.document_id = d.document_id
                 AND v.version_hash = d.current_version
                WHERE d.document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if not row:
            return None
        document = dict(row)
        document["parser_warnings"] = json.loads(document["parser_warnings"])
        return document
