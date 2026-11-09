import json
import logging
import re
import time
from uuid import uuid4

import numpy as np
import redis

from backend.settings import settings

logger = logging.getLogger(__name__)


class AnswerCache:
    """Permission- and evidence-scoped exact and semantic Redis cache."""

    def __init__(self):
        self.client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        self.ttl = 3600
        self.semantic_limit = 100
        self.semantic_threshold = 0.92

    @staticmethod
    def _normalized_question(question: str) -> str:
        return re.sub(r"\s+", " ", question).strip().casefold()

    @classmethod
    def _exact_key(cls, question: str, permission: str, evidence: str) -> str:
        import hashlib

        normalized = cls._normalized_question(question)
        question_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"rag:exact:{permission}:{evidence}:{question_hash}"

    @staticmethod
    def _semantic_namespace(permission: str, evidence: str) -> str:
        return f"rag:semantic:{permission}:{evidence}"

    @staticmethod
    def _valid_answer(value: object) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("answer"), str)
            and isinstance(value.get("not_found"), bool)
            and isinstance(value.get("citations"), list)
            and isinstance(value.get("model"), str)
            and all(isinstance(citation, dict) for citation in value["citations"])
            and (
                (value["not_found"] and not value["citations"])
                or (not value["not_found"] and bool(value["citations"]))
            )
        )

    def lookup(
        self,
        question: str,
        embedding: list[float],
        permission: str,
        evidence: str) -> dict | None:
        try:
            exact = self.client.get(self._exact_key(question, permission, evidence))
            if exact:
                exact_answer = json.loads(exact)
                if self._valid_answer(exact_answer):
                    return exact_answer

            namespace = self._semantic_namespace(permission, evidence)
            best_score = self.semantic_threshold
            best_answer = None
            query_vector = np.asarray(embedding, dtype=np.float32)
            for entry_id in self.client.zrevrange(namespace, 0, 99):
                raw = self.client.get(f"{namespace}:{entry_id}")
                if not raw:
                    self.client.zrem(namespace, entry_id)
                    continue
                try:
                    item = json.loads(raw)
                    candidate = np.asarray(item["embedding"], dtype=np.float32)
                    answer = item["answer"]
                    if (
                        not isinstance(answer, dict)
                        or not self._valid_answer(answer)
                        or answer.get("not_found")
                        or candidate.shape != query_vector.shape
                    ):
                        continue
                    score = float(np.dot(query_vector, candidate))
                    if score >= best_score:
                        best_score = score
                        best_answer = answer
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            return best_answer
        except (redis.RedisError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Redis cache lookup unavailable: %s", type(exc).__name__)
            return None

    def store(
        self,
        question: str,
        embedding: list[float],
        permission: str,
        evidence: str,
        answer: dict) -> None:
        if not self._valid_answer(answer):
            return
        try:
            exact_key = self._exact_key(question, permission, evidence)
            self.client.set(exact_key, json.dumps(answer), ex=self.ttl)
            if answer.get("not_found"):
                return
            namespace = self._semantic_namespace(permission, evidence)
            entry_id = uuid4().hex
            self.client.set(
                f"{namespace}:{entry_id}",
                json.dumps({"embedding": embedding, "answer": answer}),
                ex=self.ttl,
            )
            self.client.zadd(namespace, {entry_id: time.time()})
            self.client.expire(namespace, self.ttl)
            overflow = self.client.zcard(namespace) - self.semantic_limit
            if overflow > 0:
                stale = self.client.zrange(namespace, 0, overflow - 1)
                if stale:
                    self.client.zrem(namespace, *stale)
                    self.client.delete(*(f"{namespace}:{key}" for key in stale))
        except (redis.RedisError, OSError, TypeError, ValueError) as exc:
            logger.warning("Redis cache write unavailable: %s", type(exc).__name__)

    def close(self) -> None:
        self.client.close()
