from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.]?\s*(.*?)\s*$")
_CHAPTER_NUMBER_RE = re.compile(r"^\s*(?:Глава|Chapter)\s+(\d+)\s*$", re.IGNORECASE)

_TYPE_LABELS = {
    "theorem": "теорема",
    "definition": "определение",
    "lemma": "лемма",
    "corollary": "следствие",
    "property": "свойство",
    "remark": "замечание",
    "example": "пример",
    "rule": "правило",
    "scheme": "схема",
    "bullet": "утверждение",
    "text": "текст",
}


def _clean_text(value: Any) -> str:
    """Convert a value to a trimmed string without touching math notation."""
    if value is None:
        return ""
    return str(value).strip()


def _parse_section(section: str, section_title: str | None = None) -> tuple[str, str]:
    """
    Normalize section metadata.

    New parser schema:
        section = "8.2"
        section_title = "Деление с остатком в кольце многочленов"

    Legacy schema:
        section = "8.2 Деление с остатком в кольце многочленов."
    """
    section_raw = _clean_text(section)
    explicit_title = _clean_text(section_title)

    if not section_raw:
        return "", explicit_title

    match = _SECTION_RE.match(section_raw)
    if not match:
        return section_raw, explicit_title or section_raw

    number = match.group(1)
    parsed_title = match.group(2).strip().rstrip(".")
    return number, explicit_title or parsed_title


def _parse_chapter(
    chapter: str,
    chapter_number: str | None = None,
    chapter_name: str | None = None,
) -> tuple[str, str]:
    """
    Normalize chapter metadata for both the new and legacy JSONL schemas.

    New schema:
        chapter = "Кольцо многочленов"
        chapter_number = "8"

    Legacy schema:
        chapter = "Глава 8"
        chapter_name = "Кольцо многочленов"
    """
    chapter_raw = _clean_text(chapter)
    explicit_number = _clean_text(chapter_number)
    explicit_name = _clean_text(chapter_name)

    if explicit_number:
        number = explicit_number
    else:
        match = _CHAPTER_NUMBER_RE.match(chapter_raw)
        number = match.group(1) if match else chapter_raw

    if explicit_name:
        name = explicit_name
    elif _CHAPTER_NUMBER_RE.match(chapter_raw):
        # Legacy form: chapter="Глава 8", chapter_name may be absent.
        name = ""
    else:
        # New form: chapter already contains the human-readable title.
        name = chapter_raw

    return number, name


def _iter_elements(item: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield structured elements in their semantic order."""
    for key in ("statement", "content", "proof"):
        value = item.get(key)
        if isinstance(value, list):
            for element in value:
                if isinstance(element, dict):
                    yield element


def element_to_text(element: dict[str, Any]) -> str:
    """Convert one structured element into embedding text."""
    element_type = _clean_text(element.get("type")) or "text"

    if element_type == "image":
        path = _clean_text(element.get("path"))
        return f"IMAGE: {path}" if path else "IMAGE"

    text = _clean_text(element.get("text"))
    if not text:
        return ""

    if element_type == "formula":
        return f"FORMULA:\n{text}"

    if element_type == "text":
        return text

    return f"{element_type.upper()}:\n{text}"


def _elements_to_text(elements: Any) -> list[str]:
    """Render a structured element list; tolerate malformed entries."""
    if not isinstance(elements, list):
        return []

    result: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        rendered = element_to_text(element)
        if rendered:
            result.append(rendered)
    return result


def _content_to_text(item: dict[str, Any]) -> str:
    """
    Read semantic content from either schema:

    New schema:
        content = "..."

    Legacy schema:
        content = [{"type": "text", "text": "..."}, ...]
    """
    content = item.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n\n".join(_elements_to_text(content)).strip()

    # Extra backward-compatible fallback for legacy theorem representation.
    sections: list[str] = []

    statement = _elements_to_text(item.get("statement"))
    proof = _elements_to_text(item.get("proof"))

    if statement:
        sections.append("\n\n".join(statement))
    if proof:
        sections.append("\n\n".join(proof))

    return "\n\n".join(sections).strip()


def _extract_equations(item: dict[str, Any]) -> list[str]:
    """Extract equations from explicit metadata or structured elements."""
    equations = item.get("equations")
    if isinstance(equations, list):
        normalized = [_clean_text(eq) for eq in equations if _clean_text(eq)]
        if normalized:
            return normalized

    result: list[str] = []
    for element in _iter_elements(item):
        if _clean_text(element.get("type")) == "formula":
            formula = _clean_text(element.get("text"))
            if formula:
                result.append(formula)
    return result


def _extract_images(item: dict[str, Any]) -> list[str]:
    """Extract image paths from explicit metadata or structured elements."""
    images = item.get("images")
    if isinstance(images, list):
        normalized = [_clean_text(path) for path in images if _clean_text(path)]
        if normalized:
            return normalized

    result: list[str] = []
    for element in _iter_elements(item):
        if _clean_text(element.get("type")) == "image":
            path = _clean_text(element.get("path"))
            if path:
                result.append(path)
    return result


def _build_page_content(
    item: dict[str, Any],
    *,
    part_name: str,
    chapter_name: str,
    section_number: str,
    section_title: str,
) -> str:
    """
    Prefer parser-produced retrieval_text because it already contains the
    intended semantic context. Fall back to reconstructing it for legacy JSONL.
    """
    retrieval_text = _clean_text(item.get("retrieval_text"))
    if retrieval_text:
        return retrieval_text

    item_type = _clean_text(item.get("type")) or "text"
    label = _TYPE_LABELS.get(item_type, item_type)

    parts: list[str] = [f"Тип знания: {label}"]

    if part_name:
        parts.append(f"Часть: {part_name}")

    if chapter_name:
        parts.append(f"Глава: {chapter_name}")

    if section_number and section_title:
        parts.append(f"Раздел: {section_number} — {section_title}")
    elif section_number:
        parts.append(f"Раздел: {section_number}")
    elif section_title:
        parts.append(f"Раздел: {section_title}")

    # New semantic JSONL keeps the complete semantic unit in `content`.
    content_text = _content_to_text(item)
    if content_text:
        parts.append(content_text)

    return "\n\n".join(part for part in parts if part).strip()


def item_to_document(item: dict[str, Any], document_name: str | None = None) -> Document:
    """Convert one semantic JSONL record into a LangChain Document."""
    item_type = _clean_text(item.get("type")) or "text"

    part_name = _clean_text(item.get("part")) or _clean_text(item.get("part_name"))
    part_number = _clean_text(item.get("part_number"))

    chapter_number, chapter_name = _parse_chapter(
        _clean_text(item.get("chapter")),
        _clean_text(item.get("chapter_number")),
        _clean_text(item.get("chapter_name")),
    )

    section_number, section_title = _parse_section(
        _clean_text(item.get("section")),
        _clean_text(item.get("section_title")),
    )

    section_raw = _clean_text(item.get("section_raw")) or _clean_text(item.get("section"))
    parent_section = _clean_text(item.get("parent_section")) or section_number or section_raw

    equations = _extract_equations(item)
    images = _extract_images(item)

    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    source_file = _clean_text(source.get("file"))

    record_document = _clean_text(item.get("document"))
    inferred_document = (
        record_document
        or document_name
        or (Path(source_file).stem if source_file else "")
    )

    metadata: dict[str, Any] = {
        "knowledge_id": _clean_text(item.get("id")) or None,
        "document": inferred_document or None,
        "part": part_name or None,
        "part_number": part_number or None,
        "type": item_type,
        "title": _clean_text(item.get("title")) or None,
        "chapter": chapter_name or None,
        "chapter_number": chapter_number or None,
        "chapter_name": chapter_name or None,
        "section": section_number or section_raw or None,
        "section_raw": section_raw or None,
        "section_title": section_title or None,
        "parent_section": parent_section or None,
        "source_file": source_file or None,
        "start_line": source.get("start_line"),
        "end_line": source.get("end_line"),
        "page": item.get("page"),
        "equations": equations,
        "equation_count": len(equations),
        "images": images,
        "image_count": len(images),
        "has_proof": bool(item.get("contains_proof", item.get("proof"))),
    }

    # Qdrant metadata should contain only supported scalar/list values.
    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }

    page_content = _build_page_content(
        item,
        part_name=part_name,
        chapter_name=chapter_name,
        section_number=section_number,
        section_title=section_title,
    )

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def load_documents(path: str | Path, document: str | None = None) -> list[Document]:
    """Load semantic JSONL records into LangChain Documents."""
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
