import re
import json
import hashlib
from pathlib import Path
from typing import Optional


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/fil_ag.md"
OUTPUT_FILE = "data/processed/fil_ag.jsonl"


# ============================================================
# LOAD
# ============================================================

text = Path(INPUT_FILE).read_text(encoding="utf-8")


# ============================================================
# HELPERS
# ============================================================

def make_id(source_file: str, line_number: int, prefix: str = "") -> str:
    raw = f"{source_file}:{line_number}:{prefix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_markup(text: str) -> str:
    """
    Осторожно нормализует Markdown-обёртку.

    Примеры:

        **## Глава 1**
        -> ## Глава 1

        **Теорема.**
        -> Теорема.

        **\*\*Теорема.\*\***
        -> Теорема.
    """

    text = text.strip()

    # В исходном файле встречается \*\*
    text = text.replace(r"\*\*", "**")

    # Убираем несколько внешних пар **
    changed = True

    while changed:
        changed = False

        if text.startswith("**") and text.endswith("**"):
            text = text[2:-2].strip()
            changed = True

    return text


def normalize_heading(line: str) -> str:
    """
    Убирает внешние ** и ##.
    """

    line = normalize_markup(line)

    if line.startswith("##"):
        line = re.sub(r"^##\s*", "", line)

    return line.strip()


# ============================================================
# CLASSIFICATION
# ============================================================

def is_chapter_no(line: str) -> bool:
    title = normalize_heading(line)
    return bool(
        re.fullmatch(r"Глава\s+\d+", title)
    )


def is_section(line: str) -> bool:
    title = normalize_heading(line)
    return bool(
        re.match(r"^\d+\.\d+\s+", title)
    )


def is_chapter_name(line: str) -> bool:
    line = line.strip()

    # Заголовок ## ...
    normalized = normalize_markup(line)

    return (
        normalized.startswith("## ")
        and not is_chapter_no(line)
        and not is_section(line)
    )


def is_heading(line: str) -> bool:
    normalized = normalize_markup(line)
    return normalized.startswith("## ")


def is_definition(line: str) -> bool:
    """
    В твоём формате определения начинаются с ∙.
    Мы считаем новым смысловым блоком любой такой bullet,
    содержащий 'называется/называются'.

    Примеры:

        ∙ Отрезок называется...
        ∙ Вектором называется...
        ∙ Цилиндрическими координатами ... называются...
    """

    return bool(
        re.match(
            r"^∙\s+.*?\bназыва(?:ется|ются)\b",
            line,
            flags=re.IGNORECASE | re.DOTALL
        )
    )


def is_bullet(line: str) -> bool:
    return line.strip().startswith("∙")


def is_theorem(line: str) -> bool:
    """
    Теоремы у тебя выделяются жирным.

    Ловит:

        **Теорема.**
        **Теорема Шаля.**
        **Теорема.** Координата...

    Но не ловит:

        Теорема остаётся справедливой...
    """

    raw = line.strip()

    # Нормализуем только escaped **
    raw = raw.replace(r"\*\*", "**")

    return bool(
        re.match(
            r"^\*\*\s*Теорема\b",
            raw,
            flags=re.IGNORECASE
        )
    )


def is_formula_delimiter(line: str) -> bool:
    return line.strip() == "$$"


def is_image(line: str) -> bool:
    """
    Поддерживает и:

        ![](images/a.jpg)

    и твой вариант:

        ![]\(images/a.jpg)
    """

    return bool(
        re.match(
            r"^!\[[^\]]*\]\\?\(([^)]+)\)\s*$",
            line.strip()
        )
    )


def extract_image_path(line: str) -> Optional[str]:
    match = re.match(
        r"^!\[[^\]]*\]\\?\(([^)]+)\)\s*$",
        line.strip()
    )

    if not match:
        return None

    return match.group(1)


# ============================================================
# MARKERS
# ============================================================

def has_proof_start(line: str) -> bool:
    return "♦" in line


def has_proof_end(line: str) -> bool:
    return "⊠" in line


# ============================================================
# OBJECT FACTORIES
# ============================================================

def create_definition(
    chapter: str,
    chapter_name: str,
    section: str,
    text: str,
    line_number: int,
) -> dict:

    return {
        "id": make_id(INPUT_FILE, line_number, "definition"),
        "type": "definition",

        "chapter": chapter,
        "chapter_name": chapter_name,
        "section": section,

        "content": [
            {
                "type": "text",
                "text": text,
                "line": line_number,
            }
        ],

        "source": {
            "file": INPUT_FILE,
            "start_line": line_number,
            "end_line": line_number,
        },
    }


def create_bullet(
    chapter: str,
    chapter_name: str,
    section: str,
    text: str,
    line_number: int,
) -> dict:

    return {
        "id": make_id(INPUT_FILE, line_number, "bullet"),
        "type": "bullet",

        "chapter": chapter,
        "chapter_name": chapter_name,
        "section": section,

        "content": [
            {
                "type": "text",
                "text": text,
                "line": line_number,
            }
        ],

        "source": {
            "file": INPUT_FILE,
            "start_line": line_number,
            "end_line": line_number,
        },
    }


def create_theorem(
    chapter: str,
    chapter_name: str,
    section: str,
    statement: str,
    line_number: int,
) -> dict:

    return {
        "id": make_id(INPUT_FILE, line_number, "theorem"),
        "type": "theorem",

        "chapter": chapter,
        "chapter_name": chapter_name,
        "section": section,

        "statement": [
            {
                "type": "text",
                "text": statement,
                "line": line_number,
            }
        ],

        "proof": [],

        "source": {
            "file": INPUT_FILE,
            "start_line": line_number,
            "end_line": line_number,
        },
    }


def create_text(
    chapter: str,
    chapter_name: str,
    section: str,
    text: str,
    line_number: int,
) -> dict:

    return {
        "id": make_id(INPUT_FILE, line_number, "text"),
        "type": "text",

        "chapter": chapter,
        "chapter_name": chapter_name,
        "section": section,

        "content": [
            {
                "type": "text",
                "text": text,
                "line": line_number,
            }
        ],

        "source": {
            "file": INPUT_FILE,
            "start_line": line_number,
            "end_line": line_number,
        },
    }


# ============================================================
# PARSER STATE
# ============================================================

blocks = []

current_chapter = ""
current_chapter_name = ""
current_section = ""

current_object = None

# Возможные режимы:

# normal
# theorem_statement
# theorem_proof
# formula
mode = "normal"

# Режим перед входом в formula
previous_mode = None

# Буфер формулы
formula_buffer = []

# Текущая строка
current_line_number = 0


# ============================================================
# OBJECT HELPERS
# ============================================================

def save_current_object():
    """
    Сохраняет текущий объект, если он существует.
    """

    global current_object

    if current_object is None:
        return

    current_object["source"]["end_line"] = current_line_number

    blocks.append(current_object)

    current_object = None


def update_current_object_end_line():
    if current_object is not None:
        current_object["source"]["end_line"] = current_line_number


def append_theorem_statement_text(text: str, line_number: int):
    if not text.strip():
        return

    current_object["statement"].append({
        "type": "text",
        "text": text.strip(),
        "line": line_number,
    })


def append_theorem_proof_text(text: str, line_number: int):
    if not text.strip():
        return

    current_object["proof"].append({
        "type": "text",
        "text": text.strip(),
        "line": line_number,
    })


def append_definition_text(text: str, line_number: int):
    if not text.strip():
        return

    current_object["content"].append({
        "type": "text",
        "text": text.strip(),
        "line": line_number,
    })


def append_formula(formula_text: str, line_number: int):
    """
    Кладёт формулу туда, где parser находился до входа
    в режим formula.
    """

    global current_object

    element = {
        "type": "formula",
        "text": formula_text.strip(),
        "line": line_number,
    }

    if current_object is None:

        blocks.append({
            "id": make_id(INPUT_FILE, line_number, "formula"),
            "type": "formula",
            "chapter": current_chapter,
            "chapter_name": current_chapter_name,
            "section": current_section,
            "text": formula_text.strip(),
            "source": {
                "file": INPUT_FILE,
                "start_line": line_number,
                "end_line": line_number,
            },
        })

        return

    if current_object["type"] == "theorem":

        if previous_mode == "theorem_statement":
            current_object["statement"].append(element)

        elif previous_mode == "theorem_proof":
            current_object["proof"].append(element)

    elif current_object["type"] in ("definition", "bullet", "text"):

        current_object["content"].append(element)

    update_current_object_end_line()


def append_image(line: str, line_number: int):
    path = extract_image_path(line)

    if path is None:
        return

    element = {
        "type": "image",
        "path": path,
        "line": line_number,
    }

    if current_object is None:

        blocks.append({
            "id": make_id(INPUT_FILE, line_number, "image"),
            "type": "image",

            "chapter": current_chapter,
            "chapter_name": current_chapter_name,
            "section": current_section,

            "path": path,

            "source": {
                "file": INPUT_FILE,
                "start_line": line_number,
                "end_line": line_number,
            },
        })

        return

    if current_object["type"] == "theorem":

        if mode == "theorem_statement":
            current_object["statement"].append(element)

        elif mode == "theorem_proof":
            current_object["proof"].append(element)

    else:

        current_object["content"].append(element)

    update_current_object_end_line()


# ============================================================
# CONTENT PROCESSING
# ============================================================

def append_plain_text(text: str, line_number: int):
    """
    Добавляет обычный текст в зависимости от текущего состояния.
    """

    global current_object

    text = text.strip()

    if not text:
        return

    # ----------------------------------------
    # THEOREM
    # ----------------------------------------

    if (
        current_object is not None
        and current_object["type"] == "theorem"
    ):

        if mode == "theorem_statement":

            append_theorem_statement_text(
                text,
                line_number
            )

            return

        if mode == "theorem_proof":

            append_theorem_proof_text(
                text,
                line_number
            )

            return

    # ----------------------------------------
    # DEFINITION / BULLET / TEXT
    # ----------------------------------------

    if current_object is not None:

        append_definition_text(
            text,
            line_number
        )

        return

    # ----------------------------------------
    # Нет текущего объекта
    # Создаём text-object
    # ----------------------------------------

    current_object = create_text(
        current_chapter,
        current_chapter_name,
        current_section,
        text,
        line_number,
    )


# ============================================================
# FORMULA HANDLER
# ============================================================

def start_formula():
    global mode
    global previous_mode
    global formula_buffer

    previous_mode = mode
    mode = "formula"
    formula_buffer = []


def finish_formula(line_number: int):
    global mode
    global previous_mode
    global formula_buffer

    formula_text = "\n".join(
        formula_buffer
    ).strip()

    append_formula(
        formula_text,
        line_number
    )

    formula_buffer = []

    mode = previous_mode
    previous_mode = None


# ============================================================
# INLINE TOKENIZER
# ============================================================

MARKER_RE = re.compile(
    r"(\$\$|♦|⊠)"
)


def split_markers(text: str):
    """
    Делит строку:

        abc $$ def ♦ ghi ⊠ jkl

    на:

        abc
        $$
        def
        ♦
        ghi
        ⊠
        jkl
    """

    return [
        part
        for part in MARKER_RE.split(text)
        if part != ""
    ]


# ============================================================
# MAIN LINE PROCESSOR
# ============================================================

def process_line(line: str, line_number: int):

    global mode
    global formula_buffer
    global current_object

    # --------------------------------------------------------
    # Если внутри formula
    # --------------------------------------------------------

    if mode == "formula":

        parts = split_markers(line)

        for part in parts:

            if part == "$$":

                finish_formula(line_number)

            else:

                if mode == "formula":
                    formula_buffer.append(part)

                else:
                    process_line(
                        part,
                        line_number
                    )

        return


    # --------------------------------------------------------
    # Context headings
    # --------------------------------------------------------

    if is_heading(line):

        # Закрываем текущий объект перед новым разделом
        save_current_object()

        heading = normalize_heading(line)

        if is_chapter_no(line):

            current_chapter_update = heading

            global current_chapter
            current_chapter = current_chapter_update

        elif is_section(line):

            global current_section
            current_section = heading

        else:

            global current_chapter_name
            current_chapter_name = heading

        return


    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    if is_image(line):

        append_image(
            line,
            line_number
        )

        return


    # --------------------------------------------------------
    # THEOREM
    # --------------------------------------------------------

    if is_theorem(line):

        # Закрываем предыдущий объект
        save_current_object()

        statement = normalize_markup(line)

        current_object = create_theorem(
            current_chapter,
            current_chapter_name,
            current_section,
            statement,
            line_number,
        )

        mode = "theorem_statement"

        # Теорема может содержать ♦ на той же строке
        if "♦" in statement:

            before_proof, after_proof = statement.split(
                "♦",
                1
            )

            current_object["statement"] = []

            if before_proof.strip():

                current_object["statement"].append({
                    "type": "text",
                    "text": normalize_markup(
                        before_proof
                    ),
                    "line": line_number,
                })

            mode = "theorem_proof"

            if after_proof.strip():

                if "⊠" in after_proof:

                    proof_text, after_end = after_proof.split(
                        "⊠",
                        1
                    )

                    if proof_text.strip():

                        current_object["proof"].append({
                            "type": "text",
                            "text": proof_text.strip(),
                            "line": line_number,
                        })

                    save_current_object()

                    if after_end.strip():

                        append_plain_text(
                            after_end,
                            line_number
                        )

                    mode = "normal"

                else:

                    current_object["proof"].append({
                        "type": "text",
                        "text": after_proof.strip(),
                        "line": line_number,
                    })

        return


    # --------------------------------------------------------
    # START PROOF
    # --------------------------------------------------------

    if (
        "♦" in line
        and current_object is not None
        and current_object["type"] == "theorem"
        and mode == "theorem_statement"
    ):

        before_proof, after_proof = line.split(
            "♦",
            1
        )

        if before_proof.strip():

            append_theorem_statement_text(
                before_proof,
                line_number
            )

        mode = "theorem_proof"

        if after_proof.strip():

            if "⊠" in after_proof:

                proof_text, after_end = after_proof.split(
                    "⊠",
                    1
                )

                if proof_text.strip():

                    append_theorem_proof_text(
                        proof_text,
                        line_number
                    )

                save_current_object()

                mode = "normal"

                if after_end.strip():

                    append_plain_text(
                        after_end,
                        line_number
                    )

            else:

                append_theorem_proof_text(
                    after_proof,
                    line_number
                )

        return


    # --------------------------------------------------------
    # END PROOF
    # --------------------------------------------------------

    if (
        "⊠" in line
        and current_object is not None
        and current_object["type"] == "theorem"
        and mode == "theorem_proof"
    ):

        before_end, after_end = line.split(
            "⊠",
            1
        )

        if before_end.strip():

            append_theorem_proof_text(
                before_end,
                line_number
            )

        # Полностью закрываем theorem
        save_current_object()

        mode = "normal"

        # Если после ⊠ был текст,
        # он относится уже к следующему объекту
        if after_end.strip():

            append_plain_text(
                after_end,
                line_number
            )

        return


    # --------------------------------------------------------
    # START DEFINITION
    # --------------------------------------------------------

    if is_definition(line):

        save_current_object()

        current_object = create_definition(
            current_chapter,
            current_chapter_name,
            current_section,
            line.strip(),
            line_number,
        )

        mode = "normal"

        return


    # --------------------------------------------------------
    # START BULLET
    # --------------------------------------------------------

    if is_bullet(line):

        # Если это просто bullet, который не прошёл
        # is_definition(), всё равно создаём отдельный объект.
        save_current_object()

        current_object = create_bullet(
            current_chapter,
            current_chapter_name,
            current_section,
            line.strip(),
            line_number,
        )

        mode = "normal"

        return


    # --------------------------------------------------------
    # FORMULA
    # --------------------------------------------------------

    # В случае строки с $$:
    if "$$" in line:

        parts = split_markers(line)

        for part in parts:

            if part == "$$":

                if mode != "formula":
                    start_formula()

                else:
                    finish_formula(
                        line_number
                    )

            elif part.strip():

                if mode == "formula":

                    formula_buffer.append(
                        part
                    )

                else:

                    append_plain_text(
                        part,
                        line_number
                    )

        return


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if is_image(line):

        append_image(
            line,
            line_number
        )

        return


    # --------------------------------------------------------
    # NORMAL TEXT
    # --------------------------------------------------------

    append_plain_text(
        line,
        line_number
    )


# ============================================================
# RUN
# ============================================================

for line_number, line in enumerate(
    text.splitlines(),
    start=1
):

    current_line_number = line_number

    if not line.strip():
        continue

    process_line(
        line,
        line_number
    )


# ------------------------------------------------------------
# Если файл закончился внутри formula
# ------------------------------------------------------------

if mode == "formula":

    finish_formula(
        current_line_number
    )


# ------------------------------------------------------------
# Сохраняем последний объект
# ------------------------------------------------------------

save_current_object()


# ============================================================
# POSTPROCESSING
# ============================================================

def clean_object(obj: dict) -> dict:
    """
    Убирает лишние пробелы и пустые элементы.
    """

    if obj["type"] == "theorem":

        new_statement = []

        for element in obj["statement"]:

            if element["type"] == "text":

                text_value = element["text"].strip()

                if text_value:
                    element["text"] = text_value
                    new_statement.append(element)

            else:

                new_statement.append(element)

        obj["statement"] = new_statement


        new_proof = []

        for element in obj["proof"]:

            if element["type"] == "text":

                text_value = element["text"].strip()

                if text_value:
                    element["text"] = text_value
                    new_proof.append(element)

            else:

                new_proof.append(element)

        obj["proof"] = new_proof

    elif "content" in obj:

        new_content = []

        for element in obj["content"]:

            if element["type"] == "text":

                text_value = element["text"].strip()

                if text_value:
                    element["text"] = text_value
                    new_content.append(element)

            else:

                new_content.append(element)

        obj["content"] = new_content

    return obj


blocks = [
    clean_object(block)
    for block in blocks
]


# ============================================================
# SAVE JSONL
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for block in blocks:

        f.write(
            json.dumps(
                block,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# REPORT
# ============================================================

type_counts = {}

for block in blocks:

    block_type = block["type"]

    type_counts[block_type] = (
        type_counts.get(block_type, 0)
        + 1
    )


print()
print("=" * 60)
print("PARSER FINISHED")
print("=" * 60)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Objects: {len(blocks)}")

print()
print("Object types:")

for block_type, count in sorted(
    type_counts.items()
):

    print(
        f"  {block_type:15} {count}"
    )

print()
print("First objects:")

for block in blocks[:5]:

    print(
        json.dumps(
            block,
            ensure_ascii=False,
            indent=2
        )
    )

    print("-" * 60)