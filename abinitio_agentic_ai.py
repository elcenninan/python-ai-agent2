#!/usr/bin/env python3
"""
Agentic AI helper for Ab Initio failure logs.

What it does:
1. Reads a failure log (text file).
2. Detects likely failure scenario (unique constraint, null column, etc.).
3. Retrieves relevant table/status guidance from a local RAG JSON knowledge base.
4. Calls Ollama local APIs to:
   - analyze log + retrieved context,
   - generate recommendations,
   - generate SQL update queries.

Example:
    python abinitio_agentic_ai.py \
      --log-file ./sample_failure.log \
      --rag-file ./rag_table_details.json \
      --model llama3.1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


@dataclass
class RetrievalResult:
    entry: dict[str, Any]
    score: float


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to call Ollama endpoint '{endpoint}'. Ensure Ollama is running at {self.base_url}."
            ) from exc

    def embedding(self, model: str, text: str) -> list[float]:
        # Ollama currently supports /api/embeddings on stable builds.
        payload = {"model": model, "prompt": text}
        body = self._post_json("/api/embeddings", payload)
        embedding = body.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama embedding response missing 'embedding'.")
        return embedding

    def generate(self, model: str, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        body = self._post_json("/api/generate", payload)
        response = body.get("response", "")
        if not response:
            raise RuntimeError("Ollama generate response missing 'response'.")
        return response.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def detect_failure_hints(log_text: str) -> dict[str, Any]:
    log_lower = log_text.lower()
    hints: dict[str, Any] = {
        "type": "unknown",
        "table": None,
        "column": None,
        "raw_matches": [],
    }

    unique_patterns = [
        r"unique constraint",
        r"duplicate key",
        r"violates unique",
        r"ora-00001",
    ]
    null_patterns = [
        r"cannot be null",
        r"null value",
        r"ora-01400",
        r"not null constraint",
    ]

    for pat in unique_patterns:
        if re.search(pat, log_lower):
            hints["type"] = "unique_constraint"
            hints["raw_matches"].append(pat)
    for pat in null_patterns:
        if re.search(pat, log_lower):
            hints["type"] = "null_value"
            hints["raw_matches"].append(pat)

    table_match = re.search(
        r"(?:table|into|update|merge into)\s+([a-zA-Z0-9_.\"]+)", log_text, re.IGNORECASE
    )
    if table_match:
        hints["table"] = table_match.group(1).strip('"')

    column_match = re.search(
        r"(?:column|field)\s+([a-zA-Z0-9_\"]+)\s+(?:cannot be null|is null)",
        log_text,
        re.IGNORECASE,
    )
    if column_match:
        hints["column"] = column_match.group(1).strip('"')

    return hints


def build_entry_text(entry: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"table_name: {entry.get('table_name', '')}",
            f"failure_type: {entry.get('failure_type', '')}",
            f"status_column: {entry.get('status_column', '')}",
            f"recommended_status: {entry.get('recommended_status', '')}",
            f"notes: {entry.get('notes', '')}",
            f"query_template: {entry.get('query_template', '')}",
        ]
    )


def load_rag_entries(rag_file: Path) -> list[dict[str, Any]]:
    with rag_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("RAG file must be a list or an object with an 'entries' list.")

    if not isinstance(entries, list) or not entries:
        raise ValueError("RAG entries list is empty or invalid.")

    return entries


def retrieve_context(
    ollama: OllamaClient,
    embed_model: str,
    entries: list[dict[str, Any]],
    query_text: str,
    top_k: int = 3,
) -> list[RetrievalResult]:
    query_vec = ollama.embedding(embed_model, query_text)
    scored: list[RetrievalResult] = []
    for entry in entries:
        entry_text = build_entry_text(entry)
        entry_vec = ollama.embedding(embed_model, entry_text)
        scored.append(
            RetrievalResult(
                entry=entry,
                score=cosine_similarity(query_vec, entry_vec),
            )
        )
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def build_prompt(log_text: str, hints: dict[str, Any], context: list[RetrievalResult]) -> str:
    context_json = json.dumps(
        [
            {
                "score": round(item.score, 4),
                "entry": item.entry,
            }
            for item in context
        ],
        indent=2,
    )

    return textwrap.dedent(
        f"""
        You are an expert data support engineer for Ab Initio + RDBMS batch pipelines.

        TASK:
        1) Analyze the failure log.
        2) Recommend a likely root cause.
        3) Use retrieved RAG table metadata to propose SQL update query/query variants.
        4) Focus on unique constraint and null column scenarios.
        5) If status change is required, use the status column + recommended status from RAG.

        RULES:
        - Do not hallucinate table names/columns if not present. If uncertain, say "Needs validation".
        - Return strict JSON with this schema:
          {{
            "failure_type": "...",
            "root_cause": "...",
            "table": "...",
            "column": "...",
            "recommendations": ["..."],
            "sql_updates": ["..."],
            "validation_checks": ["..."]
          }}

        FAILURE_HINTS:
        {json.dumps(hints, indent=2)}

        RAG_CONTEXT_TOP_MATCHES:
        {context_json}

        FAILURE_LOG:
        {log_text}
        """
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ab Initio failure log agent using Ollama + local RAG")
    parser.add_argument("--log-file", required=True, type=Path, help="Path to failure log text file")
    parser.add_argument(
        "--rag-file",
        required=True,
        type=Path,
        help="Path to RAG JSON (list of table/failure metadata entries)",
    )
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="Ollama model for generation")
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Ollama model for embeddings",
    )
    parser.add_argument("--top-k", default=3, type=int, help="How many RAG entries to retrieve")

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"ERROR: Log file not found: {args.log_file}", file=sys.stderr)
        return 2
    if not args.rag_file.exists():
        print(f"ERROR: RAG file not found: {args.rag_file}", file=sys.stderr)
        return 2

    log_text = args.log_file.read_text(encoding="utf-8")
    entries = load_rag_entries(args.rag_file)
    hints = detect_failure_hints(log_text)

    ollama = OllamaClient(args.ollama_url)
    retrieval_query = f"failure_type={hints['type']} table={hints.get('table')} column={hints.get('column')}\n{log_text[:2000]}"
    top_matches = retrieve_context(
        ollama=ollama,
        embed_model=args.embed_model,
        entries=entries,
        query_text=retrieval_query,
        top_k=max(1, args.top_k),
    )

    prompt = build_prompt(log_text, hints, top_matches)
    response = ollama.generate(model=args.model, prompt=prompt, temperature=0.0)

    print("=== DETECTED HINTS ===")
    print(json.dumps(hints, indent=2))
    print("\n=== TOP RAG MATCHES ===")
    for i, item in enumerate(top_matches, start=1):
        print(f"{i}. score={item.score:.4f} table={item.entry.get('table_name')} failure={item.entry.get('failure_type')}")

    print("\n=== AGENT RESPONSE (JSON EXPECTED) ===")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
