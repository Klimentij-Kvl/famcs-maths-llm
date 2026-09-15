import json
from langchain_core.documents import Document

def element_to_text(element: dict) -> str:
    element_type = element["type"]

    if element_type == "text":
        return element["text"]

    if element_type == "formula":
        return f"FORMULA:\n{element['text']}"

    if element_type == "image":
        return f"IMAGE: {element['path']}"

    return str(element)


def theorem_to_text(item: dict) -> str:
    parts = []

    parts.append(
        f"Тип знания: теорема"
    )

    parts.append(
        f"Глава: {item.get('chapter', '')}"
    )

    parts.append(
        f"Тема: {item.get('chapter_name', '')}"
    )

    parts.append(
        f"Раздел: {item.get('section', '')}"
    )

    statement = []

    for element in item.get("statement", []):
        statement.append(
            element_to_text(element)
        )

    if statement:
        parts.append(
            "Формулировка теоремы:\n"
            + "\n".join(statement)
        )

    proof = []

    for element in item.get("proof", []):
        proof.append(
            element_to_text(element)
        )

    if proof:
        parts.append(
            "Доказательство:\n"
            + "\n".join(proof)
        )

    return "\n\n".join(parts)


def definition_to_text(item: dict) -> str:
    parts = []

    parts.append(
        "Тип знания: определение"
    )

    parts.append(
        f"Глава: {item.get('chapter', '')}"
    )

    parts.append(
        f"Тема: {item.get('chapter_name', '')}"
    )

    parts.append(
        f"Раздел: {item.get('section', '')}"
    )

    content = []

    for element in item.get("content", []):
        content.append(
            element_to_text(element)
        )

    if content:
        parts.append(
            "Содержание:\n"
            + "\n".join(content)
        )

    return "\n\n".join(parts)


def generic_to_text(item: dict) -> str:
    return json.dumps(
        item,
        ensure_ascii=False
    )


def item_to_document(item: dict) -> Document:

    item_type = item.get("type")

    if item_type == "theorem":
        text = theorem_to_text(item)

    elif item_type == "definition":
        text = definition_to_text(item)

    else:
        text = generic_to_text(item)

    metadata = {
        "knowledge_id": item.get("id"),
        "type": item_type,
        "chapter": item.get("chapter"),
        "chapter_name": item.get("chapter_name"),
        "section": item.get("section"),
        "source_file": item.get("source", {}).get("file"),
        "start_line": item.get("source", {}).get("start_line"),
        "end_line": item.get("source", {}).get("end_line"),
    }

    return Document(
        page_content=text,
        metadata=metadata,
    )

def load_documents(path: str) -> list[Document]:

    documents = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as e:
                print(
                    f"WARNING: invalid JSON "
                    f"at line {line_number}: {e}"
                )
                continue

            documents.append(
                item_to_document(item)
            )

    return documents