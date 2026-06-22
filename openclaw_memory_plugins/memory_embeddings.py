from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence


_DEFAULT_MODEL_NAME = os.environ.get("OPENCLAW_MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
_ALLOW_DOWNLOAD = os.environ.get("OPENCLAW_MEMORY_EMBEDDING_ALLOW_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no"}
_FALLBACK_DIMENSION = int(os.environ.get("OPENCLAW_MEMORY_EMBEDDING_FALLBACK_DIMENSION", "384"))
_MAX_TOKENS = int(os.environ.get("OPENCLAW_MEMORY_EMBEDDING_MAX_TOKENS", "256"))
_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


_SYNONYM_GROUPS = {
    "concise": "concise",
    "brief": "concise",
    "short": "concise",
    "succinct": "concise",
    "compact": "concise",
    "direct": "direct",
    "straight": "direct",
    "clear": "clear",
    "simple": "clear",
    "reply": "answer",
    "replies": "answer",
    "response": "answer",
    "responses": "answer",
    "answer": "answer",
    "answers": "answer",
    "prefer": "prefer",
    "preference": "prefer",
    "like": "prefer",
    "want": "prefer",
    "need": "prefer",
    "support": "support",
    "integrate": "connect",
    "integrates": "connect",
    "integration": "connect",
    "connect": "connect",
    "works": "work",
    "run": "work",
    "runs": "work",
    "enabled": "enable",
    "supporting": "support",
}


@dataclass(slots=True)
class EmbeddingResult:
    model_name: str
    dimension: int
    vector: list[float]
    backend: str


class _HashingEmbedder:
    def __init__(self, dimension: int = _FALLBACK_DIMENSION) -> None:
        self.dimension = max(64, int(dimension))

    @staticmethod
    def _normalize_text(text: str, *, kind: str) -> str:
        cleaned = " ".join(str(text or "").split()).lower()
        tokens = [_SYNONYM_GROUPS.get(token, token) for token in _WORD_RE.findall(cleaned)]
        if not tokens:
            return f"{kind}: {cleaned}"
        return f"{kind}: {' '.join(tokens)}"

    def _hash_token(self, token: str, *, salt: str) -> tuple[int, float]:
        digest = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest[:4], "little", signed=False)
        sign = -1.0 if digest[4] & 1 else 1.0
        return value % self.dimension, sign

    def encode(self, text: str, *, kind: str = "passage") -> EmbeddingResult:
        vector = [0.0] * self.dimension
        normalized = self._normalize_text(text, kind=kind)
        tokens = _WORD_RE.findall(normalized)
        for index, token in enumerate(tokens):
            slot, sign = self._hash_token(token, salt="token")
            vector[slot] += sign * 1.0
            if index + 1 < len(tokens):
                bigram = f"{token} {tokens[index + 1]}"
                slot, sign = self._hash_token(bigram, salt="bigram")
                vector[slot] += sign * 0.5
        return EmbeddingResult(
            model_name=f"hashing-{self.dimension}",
            dimension=self.dimension,
            vector=self._l2_normalize(vector),
            backend="hashing",
        )

    def encode_many(self, texts: Iterable[str], *, kind: str = "passage") -> list[EmbeddingResult]:
        return [self.encode(text, kind=kind) for text in texts if str(text).strip()]

    @staticmethod
    def _l2_normalize(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return [0.0 for _ in vector]
        return [float(value / norm) for value in vector]


class EmbeddingBackend:
    """Lazy embedding backend with transformer and deterministic fallback layers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or _DEFAULT_MODEL_NAME
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._backend_name = "hashing"
        self._disabled_reason = ""
        self._fallback = _HashingEmbedder()
        self._dimension = self._fallback.dimension
        self._loaded = False

    @property
    def backend_name(self) -> str:
        self._ensure_loaded()
        return self._backend_name

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension

    @property
    def available(self) -> bool:
        return True

    @property
    def disabled_reason(self) -> str:
        self._ensure_loaded()
        return self._disabled_reason

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on environment
            self._disabled_reason = f"transformers unavailable: {exc}"
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=not _ALLOW_DOWNLOAD)
            model = AutoModel.from_pretrained(self.model_name, local_files_only=not _ALLOW_DOWNLOAD)
            model.eval()
            model.to("cpu")
        except Exception as exc:  # pragma: no cover - model download/caching is environment-specific
            self._disabled_reason = f"transformer model unavailable: {exc}"
            return

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._dimension = int(getattr(model.config, "hidden_size", 0) or 0)
        if self._dimension > 0:
            self._backend_name = "transformers"
        else:
            self._disabled_reason = "transformer model reported no hidden size"
            self._dimension = self._fallback.dimension

    @staticmethod
    def _format_text(text: str, *, kind: str) -> str:
        cleaned = " ".join(str(text or "").split())
        return f"{kind}: {cleaned}"

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _l2_normalize(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return [0.0 for _ in vector]
        return [float(value / norm) for value in vector]

    def encode(self, text: str, *, kind: str = "passage") -> EmbeddingResult | None:
        batch = self.encode_many([text], kind=kind)
        return batch[0] if batch else None

    def encode_many(self, texts: Iterable[str], *, kind: str = "passage") -> list[EmbeddingResult]:
        self._ensure_loaded()
        payload = [str(text) for text in texts if str(text).strip()]
        if not payload:
            return []

        if self._backend_name != "transformers" or self._tokenizer is None or self._model is None or self._torch is None:
            return self._fallback.encode_many(payload, kind=kind)

        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        with torch.no_grad():
            batch = tokenizer(
                [self._format_text(text, kind=kind) for text in payload],
                padding=True,
                truncation=True,
                max_length=_MAX_TOKENS,
                return_tensors="pt",
            )
            outputs = model(**batch)
            embeddings = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        result: list[EmbeddingResult] = []
        for vector in embeddings.cpu().tolist():
            normalized = self._l2_normalize(vector)
            result.append(
                EmbeddingResult(
                    model_name=self.model_name,
                    dimension=len(normalized),
                    vector=normalized,
                    backend=self._backend_name,
                )
            )
        return result


@lru_cache(maxsize=4096)
def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    score = 0.0
    for index in range(length):
        score += left[index] * right[index]
    return max(-1.0, min(score, 1.0))


def pack_vector(vector: Sequence[float]) -> bytes:
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def unpack_vector(blob: bytes | memoryview | None) -> tuple[float, ...]:
    if not blob:
        return tuple()
    data = bytes(blob)
    if len(data) % 4 != 0:
        return tuple()
    return struct.unpack(f"<{len(data) // 4}f", data)
