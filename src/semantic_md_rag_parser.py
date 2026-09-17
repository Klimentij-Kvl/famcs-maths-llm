#!/usr/bin/env python3
"""Semantic Markdown -> JSONL parser for mathematical lecture notes.

The parser treats the author's Markdown structure as the primary segmentation signal:
Part -> Chapter -> Section -> semantic block (definition/theorem/lemma/corollary/
property/etc.). It deliberately avoids generic fixed-size chunking unless an
individual semantic block is exceptionally large.

Output records are optimized for RAG and preserve the original Markdown, equations,
source line spans and hierarchy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
PART_RE = re.compile(r"^Часть\s+(?P<num>[IVXLCDM]+)\.\s*(?P<title>.+?)\s*$", re.I)
CHAPTER_RE = re.compile(r"^Глава\s+(?P<num>\d+)\.\s*(?P<title>.+?)\s*$", re.I)
SECTION_RE = re.compile(r"^(?P<num>\d+\.\d+)\s+(?P<title>.+?)\s*$")
TOC_HEADING_RE = re.compile(r"^Оглавление\s*$", re.I)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
DISPLAY_MATH_RE = re.compile(r"(?s)\$\$(.+?)\$\$")
MATH_BRACKET_RE = re.compile(r"(?s)\\\[(.+?)\\\]")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
LIST_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-+*]|\d+[.)])[ \t]+(?P<body>.*)$")
BOLD_LABEL_RE = re.compile(r"^\s*\*\*(?P<label>.+?)\*\*(?P<rest>.*)$")

SEMANTIC_PATTERNS = [
    ("theorem", re.compile(r"^теорем[ауые]\b|^теорема\.?$", re.I)),
    ("lemma", re.compile(r"^лемм[ауые]\b|^лемма\.?$", re.I)),
    ("corollary", re.compile(r"^следстви[ея]\b|^следствие\.?$", re.I)),
    ("proof", re.compile(r"^доказательств(?:о|а)\b|^доказательство\.?$", re.I)),
    ("definition", re.compile(r"^определени[ея]\b|^определение\.?$", re.I)),
    ("remark", re.compile(r"^замечани[ея]\b|^замечание\.?$", re.I)),
    ("example", re.compile(r"^пример\b|^пример\.?$", re.I)),
    ("property", re.compile(r"^свойств(?:о|а)\b|^основные свойства\b", re.I)),
    ("rule", re.compile(r"^правило\b|^правило\.?$", re.I)),
    ("scheme", re.compile(r"^схема\b|^схема\.?$", re.I)),
    ("method", re.compile(r"^метод\b|^алгоритм\b", re.I)),
]

PROOF_LIKE_STARTS = (
    "Пусть ", "Рассмотрим ", "Заметим ", "Так как ", "Следовательно", "Аналогично",
    "Обозначим ", "Покажем ", "Докажем ", "Имеем ", "Тогда ", "Получим ", "Отсюда",
    "Поскольку ", "Если ", "Из ", "Для ", "При этом", "Таким образом", "По ",
    "Возьмём ", "Возьмем ", "Построим ", "Пусть дана", "Пусть дан", "Пусть задан",
)

@dataclass
class Block:
    kind: str
    text: str
    start: int
    end: int

@dataclass
class Semantic:
    type: str
    blocks: list[Block]
    start: int
    end: int
    label: Optional[str] = None
    title: Optional[str] = None
    has_proof: bool = False


def stable_id(*parts: object) -> str:
    s = "\x1f".join(str(x) for x in parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_") or "document"


def parse_heading(text: str):
    text = text.strip()
    m = PART_RE.match(text)
    if m:
        return "part", m.group("num"), m.group("title").strip()
    m = CHAPTER_RE.match(text)
    if m:
        return "chapter", m.group("num"), m.group("title").strip()
    m = SECTION_RE.match(text)
    if m:
        return "section", m.group("num"), m.group("title").strip()
    return "heading", None, text


def classify_label(label: str) -> str:
    clean = re.sub(r"[*_`]", "", label).strip().rstrip(":.")
    for kind, rx in SEMANTIC_PATTERNS:
        if rx.search(clean):
            return kind
    return "text"


def label_info(block: Block):
    # Explicit bold label at the beginning of a paragraph/list item.
    first = block.text.splitlines()[0].strip() if block.text else ""
    m = BOLD_LABEL_RE.match(first)
    if m:
        label = m.group("label").strip()
        kind = classify_label(label)
        if kind != "text":
            return kind, label, m.group("rest").strip()

    # Definition-like list item: "- **X** называется ..." or "- ... называется ...".
    if block.kind == "list":
        first_item = first
        body = re.sub(r"^\s*[-+*]\s+", "", first_item)
        m2 = BOLD_LABEL_RE.match(body)
        if m2 and classify_label(m2.group("label")) != "text":
            return classify_label(m2.group("label")), m2.group("label").strip(), m2.group("rest").strip()
        if re.search(r"\b(называется|называются|называют|будем называться|будем считать)\b", body, re.I):
            return "definition", None, body
    return None, None, None


def is_structural_line(line: str) -> bool:
    return bool(HEADING_RE.match(line) or FENCE_RE.match(line))


def parse_blocks(lines: list[str], start_offset: int) -> list[Block]:
    """Parse section body into atomic Markdown blocks with source line spans."""
    out: list[Block] = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        line = lines[i]
        g = start_offset + i

        if is_structural_line(line):
            out.append(Block("heading", line, g, g))
            i += 1
            continue

        # Fenced code.
        mf = FENCE_RE.match(line)
        if mf:
            fence = mf.group(1)[:3]
            j = i + 1
            while j < n and not re.match(rf"^\s*{re.escape(fence)}+", lines[j]):
                j += 1
            if j < n:
                j += 1
            out.append(Block("code", "\n".join(lines[i:j]), g, start_offset + j - 1))
            i = j
            continue

        # Display math: $$...$$, including multiline.
        if "$$" in line:
            count = line.count("$$")
            j = i + 1
            if count < 2:
                while j < n and "$$" not in lines[j]:
                    j += 1
                if j < n:
                    j += 1
            else:
                j = i + 1
            out.append(Block("math", "\n".join(lines[i:j]), g, start_offset + j - 1))
            i = j
            continue

        # Standalone image line(s).
        if IMAGE_RE.fullmatch(line.strip()):
            out.append(Block("image", line, g, g))
            i += 1
            continue

        # Lists. Preserve all list lines inside the same block; blank lines that
        # clearly continue the list are retained.
        if LIST_RE.match(line):
            j = i + 1
            while j < n:
                if not lines[j].strip():
                    if j + 1 < n and (LIST_RE.match(lines[j + 1]) or lines[j + 1].startswith((" ", "\t"))):
                        j += 1
                        continue
                    break
                if LIST_RE.match(lines[j]) or lines[j].startswith((" ", "\t")):
                    j += 1
                    continue
                break
            out.append(Block("list", "\n".join(lines[i:j]), g, start_offset + j - 1))
            i = j
            continue

        # Paragraph/prose block.
        j = i + 1
        while j < n:
            if not lines[j].strip():
                break
            if is_structural_line(lines[j]) or "$$" in lines[j] or IMAGE_RE.fullmatch(lines[j].strip()) or LIST_RE.match(lines[j]):
                break
            j += 1
        out.append(Block("paragraph", "\n".join(lines[i:j]), g, start_offset + j - 1))
        i = j
    return out


def looks_like_proof_paragraph(text: str) -> bool:
    t = text.strip()
    return t.startswith(PROOF_LIKE_STARTS) or bool(re.search(r"\\(?:Rightarrow|Leftarrow|iff|implies|Longrightarrow)", t))


def is_explicit_anchor(block: Block):
    return label_info(block)[0]


def build_semantics(blocks: list[Block]) -> list[Semantic]:
    """Create semantic units without imposing arbitrary token/character chunks."""
    units: list[Semantic] = []
    current: Optional[Semantic] = None
    proof_mode = False

    def start(b: Block, typ: str, label=None, title=None):
        nonlocal current, proof_mode
        current = Semantic(typ, [b], b.start, b.end, label, title, False)
        units.append(current)
        proof_mode = typ == "proof"

    def append(b: Block):
        assert current is not None
        current.blocks.append(b)
        current.end = b.end

    for idx, b in enumerate(blocks):
        kind, label, title = label_info(b)
        nxt = blocks[idx + 1] if idx + 1 < len(blocks) else None

        # Explicit proof attaches to the immediately preceding claim.
        if kind == "proof" and current is not None and current.type in {"theorem", "lemma", "corollary", "definition", "property", "remark", "example", "rule", "scheme", "method"}:
            append(b)
            current.has_proof = True
            proof_mode = True
            continue

        if proof_mode and current is not None:
            # Continue the proof through ordinary prose, equations, figures and lists.
            # A new semantic anchor terminates it, while a definition-like list/paragraph
            # that is plainly not proof text also terminates it.
            if kind in {"theorem", "lemma", "corollary", "definition", "property", "remark", "example", "rule", "scheme", "method"}:
                proof_mode = False
                start(b, kind, label, title)
                continue
            if kind == "proof":
                append(b)
                continue
            if b.kind in {"math", "image", "list"} or (b.kind == "paragraph" and looks_like_proof_paragraph(b.text)):
                append(b)
                continue
            # Short prose without a new anchor is still usually part of a proof in
            # this note style; keep it unless it looks like a new definition.
            if b.kind == "paragraph" and not re.search(r"\b(называется|называются|называем|определяется)\b", b.text, re.I):
                append(b)
                continue
            proof_mode = False

        # Explicit semantic anchor starts a unit.
        if kind in {"theorem", "lemma", "corollary", "definition", "property", "remark", "example", "rule", "scheme", "method"}:
            start(b, kind, label, title)
            continue

        # Formula/image immediately following a semantic unit belongs to that unit.
        if current is not None and b.kind in {"math", "image"}:
            append(b)
            continue

        # In lists, keep consecutive numbered/property content with the current unit.
        if current is not None and b.kind == "list":
            append(b)
            continue

        # Ordinary prose continues current unit unless the previous unit has ended with proof.
        if current is not None:
            # A new definition-like bullet/prose should not get swallowed.
            if b.kind in {"paragraph", "list"}:
                looks_def = bool(re.search(r"\b(называется|называются|определяется как|говорят, что)\b", b.text, re.I))
                if looks_def and current.type not in {"property", "definition"}:
                    start(b, "definition", None, None)
                    continue
            append(b)
        else:
            start(b, "text", None, None)

    # Repair accidental standalone proof units: fold into previous semantic claim.
    repaired: list[Semantic] = []
    for u in units:
        if u.type == "proof" and repaired and repaired[-1].type in {"theorem", "lemma", "corollary", "definition", "property", "remark", "example", "rule", "scheme", "method"}:
            prev = repaired[-1]
            prev.blocks.extend(u.blocks)
            prev.end = u.end
            prev.has_proof = True
        else:
            repaired.append(u)
    return repaired


def extract_equations(markdown: str) -> list[str]:
    equations: list[str] = []
    for m in DISPLAY_MATH_RE.finditer(markdown):
        equations.append(m.group(1).strip())
    for m in MATH_BRACKET_RE.finditer(markdown):
        equations.append(m.group(1).strip())
    return equations


def extract_images(markdown: str):
    return [{"alt": a, "url": u} for a, u in IMAGE_RE.findall(markdown)]


def clean_heading_title(title: str) -> str:
    return title.strip().rstrip(".")


def infer_page(lines: list[str], start: int, end: int) -> Optional[int]:
    # Deliberately conservative: do not invent page numbers when the Markdown
    # has none. Supports common explicit page comments if a source has them.
    rx = re.compile(r"(?:page|страница)\s*[:=]\s*(\d+)", re.I)
    for line in lines[max(0, start - 3):min(len(lines), end + 2)]:
        m = rx.search(line)
        if m:
            return int(m.group(1))
    return None


def parse(text: str, source_name: str, document_id: Optional[str] = None):
    lines = text.splitlines()
    first_h1 = next((ln.strip()[2:].strip() for ln in lines if re.match(r"^#\s+", ln)), source_name)
    doc_id = document_id or slugify(Path(source_name).stem)

    # Find structural headings, then only parse bodies under actual sections.
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines, 1):
        m = HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    records: list[dict] = []
    toc_mode = False
    current_part = None
    current_chapter = None
    current_section = None
    record_index = 0

    for hi, (line_no, level, raw_title) in enumerate(headings):
        next_line = headings[hi + 1][0] if hi + 1 < len(headings) else len(lines) + 1
        kind, num, title = parse_heading(raw_title)

        # Skip document title and TOC hierarchy.
        if level == 1 and line_no == headings[0][0]:
            continue
        if TOC_HEADING_RE.match(raw_title):
            toc_mode = True
            current_part = current_chapter = current_section = None
            continue
        if kind == "part" and level == 1:
            # A top-level Part heading marks the end of the table of contents.
            toc_mode = False
        elif toc_mode:
            # Everything in the TOC is ignored for RAG records.
            continue

        if kind == "part":
            current_part = {"number": num, "title": clean_heading_title(title)}
            current_chapter = None
            current_section = None
            continue
        if kind == "chapter":
            if toc_mode:
                continue
            current_chapter = {"number": num, "title": clean_heading_title(title)}
            current_section = None
            continue
        if kind != "section" or toc_mode:
            continue

        current_section = {"number": num, "title": clean_heading_title(title), "line": line_no}
        body_start = line_no + 1
        body_end = next_line - 1
        body = lines[body_start - 1:body_end]
        blocks = parse_blocks(body, body_start)
        units = build_semantics(blocks)

        for u in units:
            raw_content = "\n\n".join(b.text.strip() for b in u.blocks if b.text.strip())
            if not raw_content.strip():
                continue
            # Remove accidental nested heading blocks from content.
            if all(b.kind == "heading" for b in u.blocks):
                continue
            equations = extract_equations(raw_content)
            images = extract_images(raw_content)
            start_line, end_line = u.start, u.end
            part_title = current_part["title"] if current_part else None
            chapter_title = current_chapter["title"] if current_chapter else None
            breadcrumb_parts = [x for x in [part_title, chapter_title, f"§ {num}. {current_section['title']}"] if x]
            breadcrumb = "\n".join(breadcrumb_parts)
            retrieval_text = breadcrumb + "\n\n" + raw_content if breadcrumb else raw_content

            rec = {
                "id": stable_id(doc_id, num, start_line, end_line, raw_content),
                "document": doc_id,
                "part": part_title,
                "part_number": current_part["number"] if current_part else None,
                "chapter": chapter_title,
                "chapter_number": current_chapter["number"] if current_chapter else None,
                "section": num,
                "section_title": current_section["title"],
                "parent_section": num,
                "type": u.type,
                "title": u.title,
                "page": infer_page(lines, start_line, end_line),
                "content": raw_content,
                "retrieval_text": retrieval_text,
                "equations": equations,
                "images": images,
                "contains_proof": u.has_proof,
                "source": {
                    "file": source_name,
                    "start_line": start_line,
                    "end_line": end_line,
                },
                "order": record_index,
            }
            records.append(rec)
            record_index += 1

    if not records:
        raise ValueError("Не удалось извлечь semantic parts из Markdown.")

    return {
        "schema_version": "2.0-semantic",
        "document": doc_id,
        "title": first_h1,
        "source": source_name,
        "records": records,
    }


def validate(data: dict):
    records = data["records"]
    errors = []
    for i, r in enumerate(records):
        required = ["id", "document", "section", "section_title", "type", "content", "equations", "parent_section", "source"]
        missing = [k for k in required if k not in r]
        if missing:
            errors.append((i, f"missing {missing}"))
        if r["section"] != r["parent_section"]:
            errors.append((i, "parent_section mismatch"))
        if not isinstance(r["equations"], list):
            errors.append((i, "equations must be list"))
        if r["source"]["start_line"] > r["source"]["end_line"]:
            errors.append((i, "bad source span"))
    # No record may be an isolated proof when the previous record can absorb it.
    for i, r in enumerate(records):
        if r["type"] == "proof":
            errors.append((i, "standalone proof record"))
    if errors:
        raise ValueError("Validation errors: " + repr(errors[:20]))


def stats(data: dict):
    rs = data["records"]
    from collections import Counter
    return {
        "records": len(rs),
        "sections": len({(r["chapter"], r["section"]) for r in rs}),
        "types": dict(Counter(r["type"] for r in rs)),
        "with_proof": sum(r["contains_proof"] for r in rs),
        "with_equations": sum(bool(r["equations"]) for r in rs),
        "equations": sum(len(r["equations"]) for r in rs),
        "with_images": sum(bool(r["images"]) for r in rs),
        "max_content_chars": max(len(r["content"]) for r in rs),
    }


def write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Semantic Markdown parser for math-note RAG")
    ap.add_argument("input", type=Path)
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--document", default=None, help="Stable document id, e.g. linear_algebra")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    text = args.input.read_text(encoding="utf-8")
    data = parse(text, args.input.name, args.document)
    validate(data)
    out = args.jsonl or args.input.with_suffix(".semantic.jsonl")
    write_jsonl(out, data["records"])
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({k: v for k, v in data.items() if k != "records"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: {args.input}")
    print(f"JSONL: {out}")
    print(f"META:  {meta}")
    if args.stats:
        print(json.dumps(stats(data), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
