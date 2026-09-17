from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*(.*?)(?:\.)?\s*$")
_CHAPTER_RE = re.compile(r"^\s*(?:Глава|Chapter)\s+(\d+)\s*$", re.IGNORECASE)

_TYPE_LABELS = {
    "theorem": "теорема",
    "definition": "определение",
    "lemma": "лемма",
    "corollary": "следствие",
    "property": "свойство",
    "remark": "замечание",
    "example": "пример",
    "rule": "правило",
    "bullet": "утверждение",
    "text": "текст",
}


def _clean_text(value: Any) -> str:
    """Normalize text without changing mathematical notation."""
    if value is None:
        return ""
    return str(value).strip()


def _iter_elements(item: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield semantic elements in the original order."""
    if isinstance(item.get("statement"), list):
        yield from item["statement"]
    if isinstance(item.get("content"), list):
        yield from item["content"]
    if isinstance(item.get("proof"), list):
        yield from item["proof"]


def element_to_text(element: dict[str, Any]) -> str:
    """Convert one parser element to embedding-friendly text."""
    element_type = element.get("type", "text")
    text = _clean_text(element.get("text"))

    if element_type == "text":
        return text

    if element_type == "formula":
        return f"FORMULA:\n{text}" if text else ""

    if element_type == "image":
        path = _clean_text(element.get("path"))
        return f"IMAGE: {path}" if path else "IMAGE"

    if text:
        return f"{element_type.upper()}:\n{text}"

    return ""


def _elements_to_text(elements: Any) -> list[str]:
    if not isinstance(elements, list):
        return []

    result: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        text = element_to_text(element)
        if text:
            result.append(text)
    return result


def _extract_equations(item: dict[str, Any]) -> list[str]:
    """Extract formula payloads from statement/content/proof, preserving order."""
    equations: list[str] = []

    for element in _iter_elements(item):
        if element.get("type") == "formula":
            text = _clean_text(element.get("text"))
            if text:
                equations.append(text)

    return equations


def _extract_images(item: dict[str, Any]) -> list[str]:
    images: list[str] = []

    for element in _iter_elements(item):
        if element.get("type") == "image":
            path = _clean_text(element.get("path"))
            if path:
                images.append(path)

    return images


def _parse_section(section: str) -> tuple[str, str]:
    """Return ('2.4', 'Скалярное произведение векторов')."""
    section = _clean_text(section)
    if not section:
        return "", ""

    match = _SECTION_RE.match(section)
    if not match:
        return section, section

    number = match.group(1)
    title = match.group(2).strip()
    return number, title


def _parse_chapter(chapter: str, chapter_name: str) -> tuple[str, str]:
    """Normalize chapter metadata while retaining source wording."""
    chapter_raw = _clean_text(chapter)
    chapter_title = _clean_text(chapter_name)

    match = _CHAPTER_RE.match(chapter_raw)
    chapter_number = match.group(1) if match else chapter_raw

    return chapter_number, chapter_title


def _part_name(item: dict[str, Any]) -> str | None:
    """Use an explicitly provided part; never invent one from chapter number."""
    for key in ("part", "part_name"):
        value = _clean_text(item.get(key))
        if value:
            return value
    return None


def _build_semantic_body(item: dict[str, Any]) -> str:
    item_type = _clean_text(item.get("type")) or "text"
    label = _TYPE_LABELS.get(item_type, item_type)

    parts: list[str] = [f"Тип знания: {label}"]

    part = _part_name(item)
    if part:
        parts.append(f"Часть: {part}")

    chapter = _clean_text(item.get("chapter_name")) or _clean_text(item.get("chapter"))
    if chapter:
        parts.append(f"Глава: {chapter}")

    section = _clean_text(item.get("section"))
    section_number, section_title = _parse_section(section)
    if section_number and section_title:
        parts.append(f"Раздел: {section_number} — {section_title}")
    elif section:
        parts.append(f"Раздел: {section}")

    if item_type == "theorem":
        statement = _elements_to_text(item.get("statement"))
        proof = _elements_to_text(item.get("proof"))

        if statement:
            parts.append("Формулировка:")
            parts.extend(statement)

        if proof:
            parts.append("Доказательство:")
            parts.extend(proof)

    else:
        content = _elements_to_text(item.get("content"))
        if content:
            parts.append("Содержание:")
            parts.extend(content)

    return "\n\n".join(part for part in parts if part)


def item_to_document(item: dict[str, Any], document_name: str | None = None) -> Document:
    """Convert one semantic JSONL record into a LangChain Document."""
    item_type = _clean_text(item.get("type")) or "text"
    section_raw = _clean_text(item.get("section"))
    section_number, section_title = _parse_section(section_raw)
    chapter_number, chapter_title = _parse_chapter(
        _clean_text(item.get("chapter")),
        _clean_text(item.get("chapter_name")),
    )

    equations = _extract_equations(item)
    images = _extract_images(item)
    source = item.get("source") if isinstance(item.get("source"), dict) else {}

    metadata = {
        "knowledge_id": _clean_text(item.get("id")) or None,
        "document": (
            _clean_text(item.get("document"))
            or document_name
            or Path(source.get("file", "")).stem
            or None
        ),
        "part": _part_name(item),
        "type": item_type,
        "chapter": chapter_title or None,
        "chapter_number": chapter_number or None,
        "chapter_name": chapter_title or None,
        "section": section_number or section_raw or None,
        "section_raw": section_raw or None,
        "section_title": section_title or None,
        "parent_section": section_number or section_raw or None,
        "source_file": source.get("file"),
        "start_line": source.get("start_line"),
        "end_line": source.get("end_line"),
        "equations": equations,
        "equation_count": len(equations),
        "images": images,
        "image_count": len(images),
        "has_proof": bool(item.get("proof")),
        "page": item.get("page"),
    }

    # Keep metadata values JSON/Qdrant friendly; remove only absent optional fields.
    metadata = {key: value for key, value in metadata.items() if value is not None}

    return Document(
        page_content=_build_semantic_body(item),
        metadata=metadata,
    )


def load_documents(path: str | Path, document: str | None = None) -> list[Document]:
    """Load semantic records from JSONL into LangChain Documents."""
    path = Path(path)
    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON at %s:%d: %s", path, line_number, exc)
                continue

            if not isinstance(item, dict):
                logger.warning("Skipping non-object JSON at %s:%d", path, line_number)
                continue

            try:
                document_obj = item_to_document(item, document_name=document)
            except Exception:
                logger.exception("Failed to convert record at %s:%d", path, line_number)
                continue

            if not document_obj.page_content.strip():
                logger.warning("Skipping empty semantic record at %s:%d", path, line_number)
                continue

            documents.append(document_obj)

    logger.info("Loaded %d semantic documents from %s", len(documents), path)
    return documents


__all__ = [
    "element_to_text",
    "item_to_document",
    "load_documents",
]