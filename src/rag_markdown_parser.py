#!/usr/bin/env python3
"""High-fidelity Markdown -> hierarchical JSON + RAG JSONL parser.

Designed for lecture notes / mathematical Markdown where structure matters:
Part -> Chapter -> Section, definitions, theorems, proofs, corollaries,
properties, lists, LaTeX display math and images.

No third-party dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

# ----------------------------- regexes -------------------------------------

RE_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
RE_FENCE = re.compile(r"^\s*(```+|~~~+)(.*)$")
RE_MATH_START = re.compile(r"^\s*\$\$\s*$")
RE_MATH_INLINE_BLOCK = re.compile(r"^\s*\$\$.*\$\$\s*$")
RE_IMAGE_ONLY = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")
RE_HR = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
RE_UL = re.compile(r"^(?P<indent>[ \t]{0,})(?P<mark>[-+*])[ \t]+(?P<body>.*)$")
RE_OL = re.compile(r"^(?P<indent>[ \t]{0,})(?P<num>\d+)[.)][ \t]+(?P<body>.*)$")
RE_BOLD_LABEL = re.compile(r"^\s*\*\*(?P<label>.+?)\*\*(?P<punct>[:.]?)[ \t]*(?P<rest>.*)$")
RE_PART = re.compile(r"^Часть\s+(?P<number>[IVXLCDM]+)\.\s*(?P<title>.+?)\s*$", re.I)
RE_CHAPTER = re.compile(r"^Глава\s+(?P<number>\d+)\.\s*(?P<title>.+?)\s*$", re.I)
RE_SECTION = re.compile(r"^(?P<number>\d+\.\d+)\s+(?P<title>.+?)\s*$")
RE_CHAPTER_SHORT = re.compile(r"^(?P<number>\d+)\.\s+\*\*(?P<title>.+?)\*\*\s*$")
RE_SECTION_TOC = re.compile(r"^(?P<number>\d+\.\d+)\s+(?P<title>.+?)\.?$")
RE_NUMBERED_CHAPTER_TOC = re.compile(r"^(?P<number>\d+)\.\s+\*\*(?P<title>.+?)\*\*\s*$")
RE_URL = re.compile(r"https?://[^\s)]+")
RE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-ZЁ])")

SEMANTIC_LABELS = [
    ("theorem", re.compile(r"^теорема(?:\b|\s)", re.I)),
    ("lemma", re.compile(r"^лемма(?:\b|\s)", re.I)),
    ("corollary", re.compile(r"^следствие(?:\b|\s)", re.I)),
    ("proof", re.compile(r"^доказательство(?:\b|\s)", re.I)),
    ("remark", re.compile(r"^замечание(?:\b|\s)", re.I)),
    ("definition", re.compile(r"^определение(?:\b|\s)", re.I)),
    ("example", re.compile(r"^пример(?:\b|\s)", re.I)),
    ("property", re.compile(r"^свойств(?:о|а|а:)|^основные свойства", re.I)),
    ("rule", re.compile(r"^правило(?:\b|\s)", re.I)),
    ("scheme", re.compile(r"^схема(?:\b|\s)", re.I)),
]

# Labels that naturally attach to the previous theorem/lemma/corollary/etc.
PROOF_TYPES = {"proof"}
ATTACH_TO_PREVIOUS = {"proof"}

@dataclass
class SourceSpan:
    line_start: int
    line_end: int

@dataclass
class Block:
    type: str
    lines: list[str]
    span: SourceSpan
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")

@dataclass
class SemanticUnit:
    unit_id: str
    type: str
    label: Optional[str]
    title: Optional[str]
    blocks: list[Block]
    span: SourceSpan
    contains_proof: bool = False
    contains_formula: bool = False
    contains_image: bool = False

@dataclass
class Section:
    level: int
    title: str
    number: Optional[str]
    kind: str
    span: SourceSpan
    children: list[Any] = field(default_factory=list)

# ----------------------------- helpers -------------------------------------

def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(x) for x in parts)
    return prefix + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_md_for_preview(s: str, limit: int = 180) -> str:
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\$\$.*?\$\$", "[formula]", s, flags=re.S)
    s = normalize_space(s)
    return s[:limit] + ("…" if len(s) > limit else "")


def classify_label(label: str) -> str:
    clean = label.strip().rstrip(":.")
    for kind, rx in SEMANTIC_LABELS:
        if rx.search(clean):
            return kind
    return "label"


def parse_heading(text: str) -> dict[str, Any]:
    m = RE_PART.match(text)
    if m:
        return {"kind": "part", "number": m.group("number"), "title": m.group("title")}
    m = RE_CHAPTER.match(text)
    if m:
        return {"kind": "chapter", "number": m.group("number"), "title": m.group("title")}
    m = RE_SECTION.match(text)
    if m:
        return {"kind": "section", "number": m.group("number"), "title": m.group("title")}
    return {"kind": "heading", "number": None, "title": text}


def is_list_start(line: str) -> bool:
    return bool(RE_UL.match(line) or RE_OL.match(line))


def is_block_start(lines: list[str], i: int) -> bool:
    line = lines[i]
    if RE_HEADING.match(line) or RE_FENCE.match(line) or RE_MATH_START.match(line) or RE_MATH_INLINE_BLOCK.match(line):
        return True
    if RE_IMAGE_ONLY.match(line) or RE_HR.match(line) or is_list_start(line):
        return True
    return False

# ----------------------------- block parsing -------------------------------

def parse_blocks(lines: list[str], start_line_no: int) -> list[Block]:
    """Parse one section body. Blank lines are separators, not content blocks."""
    blocks: list[Block] = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        line = lines[i]
        global_line = start_line_no + i

        # Heading inside section/body (rare but legal)
        m_h = RE_HEADING.match(line)
        if m_h:
            blocks.append(Block("heading", [line], SourceSpan(global_line, global_line), {"level": len(m_h.group(1)), "title": m_h.group(2)}))
            i += 1
            continue

        # Fenced code (preserve exactly)
        mf = RE_FENCE.match(line)
        if mf:
            fence = mf.group(1)[0:3]
            j = i + 1
            while j < n and not re.match(rf"^\s*{re.escape(fence)}+", lines[j]):
                j += 1
            if j < n:
                j += 1
            blocks.append(Block("code", lines[i:j], SourceSpan(global_line, start_line_no + j - 1), {"info": mf.group(2).strip()}))
            i = j
            continue

        # Display math: single-line $$...$$ or multi-line $$ delimiters
        if RE_MATH_INLINE_BLOCK.match(line) or RE_MATH_START.match(line):
            if line.count("$$") >= 2 and not RE_MATH_START.match(line):
                j = i + 1
            else:
                j = i + 1
                while j < n and "$$" not in lines[j]:
                    j += 1
                if j < n:
                    j += 1
            blocks.append(Block("math", lines[i:j], SourceSpan(global_line, start_line_no + j - 1), {}))
            i = j
            continue

        # Standalone image
        mi = RE_IMAGE_ONLY.match(line)
        if mi:
            blocks.append(Block("image", [line], SourceSpan(global_line, global_line), {"alt": mi.group(1), "url": mi.group(2)}))
            i += 1
            continue

        # Horizontal rule
        if RE_HR.match(line):
            blocks.append(Block("hr", [line], SourceSpan(global_line, global_line), {}))
            i += 1
            continue

        # Lists: collect until next non-list block. Indented continuation lines are included.
        if is_list_start(line):
            j = i + 1
            while j < n:
                nxt = lines[j]
                if not nxt.strip():
                    # Keep blank if followed by an indented/list continuation; otherwise list ends.
                    if j + 1 < n and (is_list_start(lines[j + 1]) or lines[j + 1].startswith((" ", "\t"))):
                        j += 1
                        continue
                    break
                if is_list_start(nxt) or nxt.startswith((" ", "\t")):
                    j += 1
                    continue
                break
            raw = lines[i:j]
            items = extract_list_items(raw)
            blocks.append(Block("list", raw, SourceSpan(global_line, start_line_no + j - 1), {"items": items}))
            i = j
            continue

        # Paragraph / prose block. Stop before the next structural block start.
        j = i + 1
        while j < n:
            if not lines[j].strip():
                break
            if is_block_start(lines, j):
                break
            j += 1
        blocks.append(Block("paragraph", lines[i:j], SourceSpan(global_line, start_line_no + j - 1), {}))
        i = j

    return blocks


def extract_list_items(lines: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for idx, line in enumerate(lines):
        m = RE_UL.match(line) or RE_OL.match(line)
        if m:
            if current:
                current["text"] = "\n".join(current["lines"]).strip()
                items.append(current)
            if RE_UL.match(line):
                current = {"kind": "unordered", "marker": m.group("mark"), "indent": len(m.group("indent").replace("\t", "    ")), "lines": [line]}
            else:
                current = {"kind": "ordered", "marker": m.group("num"), "indent": len(m.group("indent").replace("\t", "    ")), "lines": [line]}
        elif current is not None:
            current["lines"].append(line)
    if current:
        current["text"] = "\n".join(current["lines"]).strip()
        items.append(current)
    return items

# ----------------------------- document parsing ----------------------------

def parse_document(text: str, source_name: str) -> dict[str, Any]:
    lines = text.splitlines()
    # We treat each heading as structural and parse all non-heading material below it.
    heading_positions: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines, 1):
        m = RE_HEADING.match(line)
        if m:
            heading_positions.append((idx, len(m.group(1)), m.group(2).strip()))

    title = heading_positions[0][2] if heading_positions and heading_positions[0][1] == 1 else source_name
    doc_id = stable_id("doc_", source_name, text[:1000], len(text))

    root: dict[str, Any] = {
        "schema_version": "1.0",
        "parser": "rag_markdown_parser",
        "source": {"name": source_name},
        "document": {
            "id": doc_id,
            "title": title,
            "line_count": len(lines),
            "char_count": len(text),
            "sections": [],
            "preamble": [],
            "toc": [],
        },
    }

    # Parse top-level headings in source order. Blocks between headings go to the active section.
    node_stack: list[dict[str, Any]] = []
    pending_blocks: list[Block] = []
    first_part_seen = False
    current_section_node: Optional[dict[str, Any]] = None
    current_chapter_node: Optional[dict[str, Any]] = None
    current_part_node: Optional[dict[str, Any]] = None
    current_toc_node: Optional[dict[str, Any]] = None

    def attach_block(block: Block) -> None:
        if current_section_node is not None:
            current_section_node.setdefault("blocks", []).append(block_to_json(block))
        elif current_chapter_node is not None:
            current_chapter_node.setdefault("blocks", []).append(block_to_json(block))
        elif current_part_node is not None:
            current_part_node.setdefault("blocks", []).append(block_to_json(block))
        elif current_toc_node is not None and not first_part_seen:
            current_toc_node.setdefault("blocks", []).append(block_to_json(block))
        else:
            root["document"]["preamble"].append(block_to_json(block))

    # We parse body ranges between headings.
    cursor = 1
    for h_idx, (line_no, level, heading_text) in enumerate(heading_positions):
        body_start = cursor
        if line_no > cursor:
            body = lines[cursor - 1: line_no - 1]
            if body:
                attachable = parse_blocks(body, cursor)
                if current_toc_node is not None and not first_part_seen:
                    for b in attachable:
                        current_toc_node.setdefault("blocks", []).append(block_to_json(b))
                elif current_section_node is not None or current_chapter_node is not None or current_part_node is not None:
                    for b in attachable:
                        attach_block(b)
                else:
                    for b in attachable:
                        root["document"]["preamble"].append(block_to_json(b))
        cursor = line_no + 1

        parsed = parse_heading(heading_text)
        kind = parsed["kind"]

        # Root title H1 is document title, not a part.
        if not first_part_seen and level == 1 and kind == "heading" and line_no == heading_positions[0][0]:
            continue

        if kind == "part":
            first_part_seen = True
            part = make_section_node(kind, parsed, line_no)
            root["document"]["sections"].append(part)
            current_part_node = part
            current_chapter_node = None
            current_section_node = None
            current_toc_node = None
        elif kind == "chapter":
            if current_part_node is None:
                # Fallback: implicit part.
                current_part_node = make_section_node("part", {"kind": "part", "number": None, "title": "Без части"}, line_no)
                root["document"]["sections"].append(current_part_node)
                first_part_seen = True
            chapter = make_section_node(kind, parsed, line_no)
            current_part_node["children"].append(chapter)
            current_chapter_node = chapter
            current_section_node = None
            current_toc_node = None
        elif kind == "section":
            if current_part_node is None:
                current_part_node = make_section_node("part", {"kind": "part", "number": None, "title": "Без части"}, line_no)
                root["document"]["sections"].append(current_part_node)
                first_part_seen = True
            if current_chapter_node is None:
                current_chapter_node = make_section_node("chapter", {"kind": "chapter", "number": None, "title": "Без главы"}, line_no)
                current_part_node["children"].append(current_chapter_node)
            section = make_section_node(kind, parsed, line_no)
            current_chapter_node["children"].append(section)
            current_section_node = section
            current_toc_node = None
        elif heading_text.strip().lower() == "оглавление":
            # Represent TOC heading explicitly and keep its contents out of RAG sections.
            toc = make_section_node("toc", parsed, line_no)
            root["document"]["toc_section"] = toc
            current_part_node = current_chapter_node = current_section_node = None
            current_toc_node = toc
        else:
            # Generic heading is attached to the nearest active structure. During TOC,
            # preserve it there so the source hierarchy is not silently flattened.
            if current_toc_node is not None and not first_part_seen:
                current_toc_node.setdefault("children", []).append(make_section_node("heading", parsed, line_no))
            else:
                target = current_section_node or current_chapter_node or current_part_node
                if target is not None:
                    target.setdefault("children", []).append(make_section_node("heading", parsed, line_no))

    # Tail after last heading
    if cursor <= len(lines):
        tail = parse_blocks(lines[cursor - 1:], cursor)
        for b in tail:
            attach_block(b)

    # Normalize spans for headings based on actual child coverage later.
    return root


def make_section_node(kind: str, parsed: dict[str, Any], line_no: int) -> dict[str, Any]:
    return {
        "id": stable_id("sec_", kind, parsed.get("number"), parsed.get("title"), line_no),
        "kind": kind,
        "number": parsed.get("number"),
        "title": parsed.get("title"),
        "source": {"line_start": line_no, "line_end": line_no},
        "children": [],
        "blocks": [],
    }


def block_to_json(b: Block) -> dict[str, Any]:
    out = {
        "type": b.type,
        "source": {"line_start": b.span.line_start, "line_end": b.span.line_end},
        "markdown": b.text,
    }
    if b.meta:
        out["meta"] = b.meta
    return out

# ----------------------------- semantic units ------------------------------

def iter_sections(doc: dict[str, Any]) -> Iterable[tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]]:
    for part in doc["document"]["sections"]:
        for chapter in part.get("children", []):
            if chapter.get("kind") != "chapter":
                continue
            for section in chapter.get("children", []):
                if section.get("kind") != "section":
                    continue
                yield section, part, chapter


def block_from_json(b: dict[str, Any]) -> Block:
    return Block(
        type=b["type"],
        lines=b["markdown"].splitlines(),
        span=SourceSpan(b["source"]["line_start"], b["source"]["line_end"]),
        meta=b.get("meta", {}),
    )


def semantic_info_for_block(block: Block) -> tuple[Optional[str], Optional[str], str]:
    if block.type in {"paragraph", "list"}:
        first = block.lines[0].strip() if block.lines else ""
        m = RE_BOLD_LABEL.match(first)
        if m:
            label = normalize_space(m.group("label"))
            kind = classify_label(label)
            rest = m.group("rest").strip()
            title = strip_md_for_preview(rest or label)
            return kind, label, title
        # Definition-like bullets: '- **...** ... называется/...' etc.
        if block.type == "list":
            first_item = block.meta.get("items", [{}])[0]
            item_text = first_item.get("text", "") if first_item else ""
            m2 = RE_BOLD_LABEL.match(re.sub(r"^\s*[-+*]\s+", "", item_text.splitlines()[0] if item_text else ""))
            if re.search(r"\b(называется|называются|называют|будем|это)\b|^[-+*]?\s*говорят,?\s+что\b", item_text, re.I):
                label = normalize_space(m2.group("label")) if m2 else None
                rest = m2.group("rest") if m2 else item_text
                return "definition", label, strip_md_for_preview(rest or label or item_text)
    return None, None, None


def build_semantic_units(section: dict[str, Any]) -> list[SemanticUnit]:
    blocks = [block_from_json(b) for b in section.get("blocks", [])]
    units: list[SemanticUnit] = []
    current: Optional[SemanticUnit] = None
    proof_active = False

    def new_unit(first: Block, kind: str, label: Optional[str], title: Optional[str]) -> SemanticUnit:
        uid = stable_id("unit_", section["id"], first.span.line_start, first.span.line_end, first.text)
        return SemanticUnit(uid, kind or first.type, label, title, [first], SourceSpan(first.span.line_start, first.span.line_end), False, first.type == "math", first.type == "image")

    def append_to_current(b: Block) -> None:
        assert current is not None
        current.blocks.append(b)
        current.span.line_end = b.span.line_end
        current.contains_formula |= b.type == "math"
        current.contains_image |= b.type == "image"

    def starts_definition(item: Block) -> bool:
        k, _, _ = semantic_info_for_block(item)
        return k == "definition"

    for idx, b in enumerate(blocks):
        kind, label, title = semantic_info_for_block(b)
        next_block = blocks[idx + 1] if idx + 1 < len(blocks) else None
        proof_closed_here = False

        # An explicit proof belongs to the preceding claim.
        if kind == "proof" and current is not None:
            append_to_current(b)
            current.contains_proof = True
            proof_active = True
            continue

        # During a proof, keep formulas, figures, lists and proof-style prose together.
        # If a prose setup is immediately followed by a definition block, close the proof first.
        if proof_active and current is not None:
            if b.type in {"math", "image", "list"}:
                append_to_current(b)
                continue
            if kind is None and b.type == "paragraph":
                txt = b.text.strip()
                proof_starters = (
                    "Пусть ", "Рассмотрим ", "Заметим ", "Так как ", "Следовательно",
                    "Теорема остается", "Теорема остаётся", "Утверждение остается", "Утверждение остаётся",
                    "Отсюда", "Аналогично", "Обозначим ", "Покажем ", "Докажем ",
                    "Имеем ", "Тогда ", "Получим ", "При этом", "Из ", "Для ",
                    "Если ", "Поскольку ", "Подстав", "Преобраз", "Найдем ", "Найдём ",
                )
                is_likely_proof = txt.startswith(proof_starters) or bool(re.search(r"\\Rightarrow|\\Leftarrow|\\iff|\\implies", txt))
                if starts_definition(next_block) if next_block is not None else False:
                    is_likely_proof = False
                if is_likely_proof:
                    append_to_current(b)
                    continue
            # New explicit semantic anchor or non-proof prose closes the proof-backed unit.
            proof_active = False
            proof_closed_here = True

        # Formula/image immediately following an anchor stays with it.
        if current is not None and kind is None and b.type in {"math", "image"} and not proof_closed_here:
            append_to_current(b)
            continue

        # Any new semantic anchor starts a fresh unit.
        if kind is not None:
            current = new_unit(b, kind, label, title)
            units.append(current)
            continue

        # A standalone definition-like paragraph/list starts a semantic unit rather than being
        # swallowed by the previous theorem/proof.
        if current is not None and current.contains_proof and starts_definition(b):
            current = new_unit(b, "definition", None, strip_md_for_preview(b.text))
            units.append(current)
            continue

        if current is None or proof_closed_here:
            current = new_unit(b, b.type, label, title or strip_md_for_preview(b.text))
            units.append(current)
        else:
            append_to_current(b)

    return units

# ----------------------------- chunking ------------------------------------

def extract_images(text: str) -> list[dict[str, str]]:
    return [{"alt": alt, "url": url} for alt, url in RE_IMAGE_ONLY.findall(text)]


def has_math(text: str) -> bool:
    return "$$" in text or bool(re.search(r"(?<!\\)\$[^$]+\$", text))


def token_estimate(text: str) -> int:
    # Cheap, language-agnostic approximation. Intentionally conservative for Russian+LaTeX.
    return max(1, round(len(text) / 3.7))


def join_unit_markdown(units: list[SemanticUnit]) -> str:
    parts = []
    for u in units:
        parts.append("\n\n".join(b.text for b in u.blocks if b.text.strip()))
    return "\n\n".join(p for p in parts if p.strip())


def build_context_header(part: dict[str, Any], chapter: dict[str, Any], section: dict[str, Any]) -> str:
    p = part.get("title")
    c = chapter.get("title")
    s = section.get("title")
    parts = []
    if p:
        parts.append(f"Часть {part.get('number')}. {p}" if part.get("number") else f"Часть: {p}")
    if c:
        parts.append(f"Глава {chapter.get('number')}. {c}" if chapter.get("number") else f"Глава: {c}")
    if s:
        parts.append(f"§ {section.get('number')}. {s}" if section.get("number") else f"§ {s}")
    return "\n".join(parts)


def split_long_markdown(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split oversized markdown while respecting display-math blocks and paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    # First split by paragraph, keeping math blocks atomic.
    raw_parts = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buf = ""
    for part in raw_parts:
        candidate = part if not buf else buf + "\n\n" + part
        if len(candidate) <= max_chars or not buf:
            buf = candidate
        else:
            pieces.append(buf)
            buf = part
    if buf:
        pieces.append(buf)

    expanded: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            expanded.append(piece)
            continue
        # Extremely long paragraph: sentence-oriented splitting, then hard fallback by words.
        sentences = RE_SENTENCE.split(piece)
        cur = ""
        for sent in sentences:
            candidate = sent if not cur else cur + " " + sent
            if len(candidate) <= max_chars or not cur:
                cur = candidate
            else:
                expanded.append(cur)
                cur = sent
        if cur:
            expanded.append(cur)

    if overlap_chars <= 0:
        return expanded

    overlapped: list[str] = []
    for i, piece in enumerate(expanded):
        if i == 0:
            overlapped.append(piece)
            continue
        prev = expanded[i - 1]
        # Add a short textual tail as overlap, but avoid duplicating huge formulas.
        tail = prev[-overlap_chars:]
        cut = tail.find("\n")
        if cut >= 0 and cut < len(tail) // 2:
            tail = tail[cut + 1:]
        overlapped.append((tail.strip() + "\n\n" + piece).strip())
    return overlapped


def generate_chunks(
    parsed: dict[str, Any],
    source_name: str,
    max_chars: int = 5000,
    overlap_chars: int = 450,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for section, part, chapter in iter_sections(parsed):
        units = build_semantic_units(section)
        if not units:
            continue
        context = build_context_header(part, chapter, section)

        # Pack complete semantic units. A theorem+proof remains together unless it exceeds max_chars.
        packed: list[SemanticUnit] = []
        packed_text = ""

        def flush_pack() -> None:
            nonlocal packed, packed_text, chunk_index
            if not packed:
                return
            source_markdown = join_unit_markdown(packed)
            # If pack itself is oversized, split and emit continuations.
            pieces = split_long_markdown(source_markdown, max_chars=max_chars, overlap_chars=overlap_chars)
            unit_types = sorted(set(u.type for u in packed))
            labels = [u.label for u in packed if u.label]
            unit_ids = [u.unit_id for u in packed]
            source_start = min(u.span.line_start for u in packed)
            source_end = max(u.span.line_end for u in packed)
            for part_no, body in enumerate(pieces, 1):
                cid = stable_id("chunk_", parsed["document"]["id"], source_start, source_end, part_no, body)
                content = f"{context}\n\n{body}" if context else body
                chunks.append({
                    "id": cid,
                    "document_id": parsed["document"]["id"],
                    "chunk_index": chunk_index,
                    "unit_ids": unit_ids,
                    "section_id": section["id"],
                    "source": {
                        "file": source_name,
                        "line_start": source_start,
                        "line_end": source_end,
                    },
                    "hierarchy": {
                        "part": {"number": part.get("number"), "title": part.get("title")},
                        "chapter": {"number": chapter.get("number"), "title": chapter.get("title")},
                        "section": {"number": section.get("number"), "title": section.get("title")},
                        "path": [x for x in [part.get("title"), chapter.get("title"), section.get("title")] if x],
                    },
                    "semantic_types": unit_types,
                    "semantic_labels": labels,
                    "contains_proof": any(u.contains_proof for u in packed),
                    "contains_formula": has_math(body),
                    "images": extract_images(body),
                    "token_estimate": token_estimate(content),
                    "content": content,
                    "source_markdown": body,
                    "continuation": {"part": part_no, "of": len(pieces)} if len(pieces) > 1 else None,
                })
                chunk_index += 1
            packed = []
            packed_text = ""

        for u in units:
            utext = join_unit_markdown([u])
            candidate = utext if not packed else packed_text + "\n\n" + utext
            # Don't pack across proof-bearing units: retrieval is usually cleaner this way.
            if packed and (len(candidate) > max_chars or any(x.contains_proof for x in packed)):
                flush_pack()
            packed.append(u)
            packed_text = utext if not packed_text else packed_text + "\n\n" + utext
        flush_pack()

    return chunks

# ----------------------------- validation ----------------------------------

def validate(parsed: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {"parts": 0, "chapters": 0, "sections": 0, "chunks": len(chunks), "theorems": 0, "lemmas": 0, "corollaries": 0, "proofs_attached": 0, "proof_blocks": 0, "formulas": 0, "images": 0}
    for part in parsed["document"]["sections"]:
        if part.get("kind") == "part":
            stats["parts"] += 1
            for chapter in part.get("children", []):
                if chapter.get("kind") == "chapter":
                    stats["chapters"] += 1
                    for section in chapter.get("children", []):
                        if section.get("kind") == "section":
                            stats["sections"] += 1
                            for u in build_semantic_units(section):
                                key = {"theorem": "theorems", "lemma": "lemmas", "corollary": "corollaries"}.get(u.type)
                                if key:
                                    stats[key] = stats.get(key, 0) + 1
                                stats["proofs_attached"] += int(u.contains_proof)
                                stats["proof_blocks"] += sum(b.type == "paragraph" and semantic_info_for_block(b)[0] == "proof" for b in u.blocks)
                                stats["formulas"] += sum(b.type == "math" for b in u.blocks)
                                stats["images"] += sum(b.type == "image" for b in u.blocks)
    if stats["chunks"] == 0:
        raise ValueError("Парсер не создал ни одного RAG-чанка. Проверьте структуру Markdown.")
    return stats

# ----------------------------- JSON serialization --------------------------

def json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

# ----------------------------- CLI -----------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="High-fidelity Markdown parser for RAG on math notes")
    ap.add_argument("input", type=Path, help="input Markdown file")
    ap.add_argument("--jsonl", type=Path, default=None, help="output JSONL chunks")
    ap.add_argument("--tree", type=Path, default=None, help="output hierarchical JSON tree")
    ap.add_argument("--max-chars", type=int, default=5000, help="target maximum chunk size in characters")
    ap.add_argument("--overlap", type=int, default=450, help="overlap for oversized semantic units")
    ap.add_argument("--stats", action="store_true", help="print parser statistics")
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8")
    parsed = parse_document(text, args.input.name)
    chunks = generate_chunks(parsed, args.input.name, args.max_chars, args.overlap)
    stats = validate(parsed, chunks)

    jsonl_path = args.jsonl or args.input.with_suffix(".chunks.jsonl")
    tree_path = args.tree or args.input.with_suffix(".structure.json")
    write_jsonl(jsonl_path, chunks)
    write_json(tree_path, parsed)

    print(f"OK: {args.input}")
    print(f"JSONL: {jsonl_path} ({len(chunks)} chunks)")
    print(f"TREE:  {tree_path}")
    if args.stats:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
