from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


_ALLOWED_TYPES = {
    "definition",
    "remark",
    "properties",
    "theorem",
    "lemma",
    "corollary",
}

_TYPE_LABELS = {
    "definition": "Определение",
    "remark": "Замечание",
    "properties": "Свойства",
    "theorem": "Теорема",
    "lemma": "Лемма",
    "corollary": "Следствие",
}


def _clean_text(value: Any) -> str:
    """Return a stripped string; preserve LaTeX/math notation as-is."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize a JSON array of strings."""
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _normalize_source(value: Any) -> dict[str, Any]:
    """Normalize source metadata from the new JSONL schema."""
    if not isinstance(value, dict):
        return {}

    source: dict[str, Any] = {}

    file_name = _clean_text(value.get("file"))
    if file_name:
        source["file"] = file_name

    start_line = value.get("start_line")
    if isinstance(start_line, int) and not isinstance(start_line, bool):
        source["start_line"] = start_line
    elif isinstance(start_line, str) and start_line.strip().isdigit():
        source["start_line"] = int(start_line.strip())

    end_line = value.get("end_line")
    if isinstance(end_line, int) and not isinstance(end_line, bool):
        source["end_line"] = end_line
    elif isinstance(end_line, str) and end_line.strip().isdigit():
        source["end_line"] = int(end_line.strip())

    return source


def _validate_item(item: dict[str, Any], line_number: int) -> None:
    """Validate one record of the new semantic JSONL schema."""
    chunk_id = _clean_text(item.get("chunk_id"))
    if not chunk_id:
        raise ValueError(
            f"Missing 'chunk_id' at JSONL line {line_number}"
        )

    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"Missing/empty 'content' for chunk {chunk_id}"
        )

    item_type = _clean_text(item.get("type"))
    if item_type not in _ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported type {item_type!r} for chunk {chunk_id}; "
            f"expected one of {sorted(_ALLOWED_TYPES)}"
        )

    chapter = _clean_text(item.get("chapter"))
    if not chapter:
        raise ValueError(
            f"Missing 'chapter' for chunk {chunk_id}"
        )

    section = _clean_text(item.get("section"))
    if not section:
        raise ValueError(
            f"Missing 'section' for chunk {chunk_id}"
        )

    section_num = _clean_text(item.get("section_num"))
    if not section_num:
        raise ValueError(
            f"Missing 'section_num' for chunk {chunk_id}"
        )


def _build_page_content(item: dict[str, Any]) -> str:
    """
    Build embedding text.

    `content` is already a complete semantic mathematical unit.
    It is never split or reconstructed from separate pieces.
    """

    chunk_type = _clean_text(item.get("type"))
    type_label = _TYPE_LABELS.get(chunk_type, chunk_type)

    chapter = _clean_text(item.get("chapter"))
    chapter_num = item.get("chapter_num")

    section = _clean_text(item.get("section"))
    section_num = _clean_text(item.get("section_num"))

    content = _clean_text(item.get("content"))

    context: list[str] = []

    if chapter_num is not None and str(chapter_num).strip():
        context.append(
            f"Глава {chapter_num}: {chapter}"
        )
    elif chapter:
        context.append(
            f"Глава: {chapter}"
        )

    if section_num and section:
        context.append(
            f"§ {section_num}. {section}"
        )
    elif section:
        context.append(
            f"Раздел: {section}"
        )

    if type_label:
        context.append(
            f"Тип: {type_label}"
        )

    if not context:
        return content

    return "\n".join(context) + "\n\n" + content


def item_to_document(item: dict[str, Any]) -> Document:
    """Convert one new JSONL record into a LangChain Document."""

    chunk_id = _clean_text(item.get("chunk_id"))
    file_name = _clean_text(item.get("file"))
    chapter = _clean_text(item.get("chapter"))
    section = _clean_text(item.get("section"))
    section_num = _clean_text(item.get("section_num"))
    item_type = _clean_text(item.get("type"))

    equations = _normalize_string_list(
        item.get("equations")
    )
    images = _normalize_string_list(
        item.get("images")
    )

    source = _normalize_source(
        item.get("source")
    )

    # Intentionally preserve the spelling from the JSONL schema.
    contains_proof = item.get(
        "contatins_proof",
        False,
    )

    if not isinstance(contains_proof, bool):
        contains_proof = (
            str(contains_proof).strip().lower()
            in {"true", "1", "yes"}
        )

    proof_value = item.get("proof")

    if isinstance(proof_value, str):
        proof = proof_value.strip()
    else:
        proof = None

    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "file": file_name or None,
        "chapter": chapter or None,
        "chapter_num": item.get("chapter_num"),
        "section": section or None,
        "section_num": section_num or None,
        "type": item_type,
        "equations": equations,
        "images": images,
        "contatins_proof": contains_proof,
        "proof": proof,
        "source": source or None,
    }

    # Also store source line numbers flat for convenient Qdrant filtering.
    if "start_line" in source:
        metadata["source_start_line"] = source["start_line"]

    if "end_line" in source:
        metadata["source_end_line"] = source["end_line"]

    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }

    return Document(
        page_content=_build_page_content(item),
        metadata=metadata,
    )


def load_documents(path: str | Path) -> list[Document]:
    """Load the new semantic JSONL file into LangChain Documents."""

    path = Path(path)

    documents: list[Document] = []
    seen_chunk_ids: set[str] = set()
    invalid_count = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_count += 1

                logger.warning(
                    "Invalid JSON at %s:%d: %s",
                    path,
                    line_number,
                    exc,
                )

                continue

            if not isinstance(item, dict):
                invalid_count += 1

                logger.warning(
                    "Skipping non-object JSON at %s:%d",
                    path,
                    line_number,
                )

                continue

            try:
                _validate_item(
                    item,
                    line_number,
                )

                chunk_id = _clean_text(
                    item["chunk_id"]
                )

                if chunk_id in seen_chunk_ids:
                    raise ValueError(
                        f"Duplicate chunk_id {chunk_id!r}"
                    )

                document_obj = item_to_document(item)

            except Exception:
                invalid_count += 1

                logger.exception(
                    "Failed to convert semantic record at %s:%d",
                    path,
                    line_number,
                )

                continue

            seen_chunk_ids.add(chunk_id)
            documents.append(document_obj)

    logger.info(
        "Loaded %d documents from %s; skipped %d records",
        len(documents),
        path,
        invalid_count,
    )

    return documents


__all__ = [
    "item_to_document",
    "load_documents",
]