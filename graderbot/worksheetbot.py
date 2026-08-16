#!/usr/bin/env python3
"""
Worksheet generator agent.

Pipeline:
  prompt -> LLM generates structured question JSON
         -> render each question via \\Question{id}{text}, LaTeX-escaping the id;
            laid out as a multi-column tabular grid when there are enough
            questions of similar estimated width, else a single vertical
            column (see choose_grid_columns)
         -> insert into template at %%QUESTIONS%% marker
         -> compile with latexmk
         -> on LaTeX compile error, feed log back to LLM to repair, retry

Requires: ANTHROPIC_API_KEY env var, `anthropic` python package, a LaTeX
distribution with `latexmk` on PATH.

Usage:
  python -m graderbot.worksheetbot --template tex/worksheet_template.tex \
      --prompt "10 question algebra worksheet on solving linear equations, grade 9" \
      --out worksheet --num-questions 10 --max-repairs 3
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import anthropic

from graderbot import storage
from graderbot.worksheet_qr import generate_worksheet_id, render_qr_png
from graderbot.worksheet_synth import _texinputs_env

# Models the worksheet-generation pipeline can run against. The first entry is
# the default (issue #22). Threaded through every client.messages.create call
# so the model is a runtime choice rather than a hardcoded constant.
AVAILABLE_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
MODEL = AVAILABLE_MODELS[0]
QUESTIONS_MARKER = "%%QUESTIONS%%"
WORKSHEET_ID_MARKER = "%%WORKSHEET_ID%%"
TITLE_BLOCK_MARKER = "%%TITLE_BLOCK%%"

OnStep = Callable[[str, Optional[str]], None]


def _print_step(msg: str, detail: Optional[str] = None) -> None:
	"""Default callback for step progress: prints to stderr, optionally with detail."""
	print(msg, file=sys.stderr)
	if detail:
		for line in detail.split("\n"):
			print(f"  {line}", file=sys.stderr)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    text: str
    answer: str
    open_ended: bool = False  # True if there's no single correct answer (issue #65); never graded right/wrong


@dataclass
class WorksheetDocument:
    """A worksheet's full body content (issue #42): a title, a free-text
    header/instructions paragraph, and its questions. Assembled by
    `generate_worksheet`/`create_worksheet_from_questions` before rendering,
    so title and header are always resolved by the time the LaTeX is built."""
    title: str
    header: str
    questions: "list[Question]"


# --------------------------------------------------------------------------
# Step 1: generate structured question data from the prompt
# --------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """You are a arithmetic and algebra worksheet author.
Given a topic/prompt, produce exam-quality questions as JSON only — no prose,
no markdown fences, no commentary.

Output schema (a JSON array):
[
  {"id": "1", "text": "<question text, real LaTeX>", "answer": "<final answer>", "open_ended": false}
]

Rules:
- Write question text as real LaTeX math notation, delimited with $...$
  (e.g. "$x^2 + 3x - 4 = 0$", "$\\frac{1}{4}+\\frac{1}{2}=$"). It is inserted
  into the worksheet .tex source verbatim, so it must be valid LaTeX.
- You are writing LaTeX inside a JSON string, so every backslash in a LaTeX
  command must be written as two backslashes so the JSON parses correctly:
  \\times, \\div, \\left, \\right, \\cdot, \\le, \\ge, \\sqrt, etc. A single
  backslash (e.g. "\times") produces invalid JSON and will be rejected.
- Answers must be a plain number (e.g. "12", "1.5") or a LaTeX fraction
  written as \\frac{a}{b} (e.g. "\\frac{3}{4}") — never a bare "a/b" slash
  expression, and never a worked solution or justification.
- Most questions have exactly one correct answer, so "open_ended" is false
  for them. Only when the prompt itself calls for a question with no single
  correct answer (e.g. an opinion or reflection prompt like "How do you feel
  about fractions?") should you set "open_ended": true and "answer": "" —
  never guess an arbitrary "correct" answer for a genuinely open-ended
  question just to fill the field.
- Vary question difficulty and phrasing; avoid near-duplicate questions.
- Match the count, topic, and difficulty level requested in the prompt exactly.
"""


def generate_questions(
    client: anthropic.Anthropic,
    prompt: str,
    num_questions: int,
    model: str = MODEL,
    on_step: OnStep = _print_step,
) -> list[Question]:
    on_step(f"Generating {num_questions} questions...")
    user_msg = f"{prompt}\n\nGenerate exactly {num_questions} questions."
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    on_step("Received response from Claude", raw)
    data = _parse_json_array(raw)

    questions = [
        Question(id=str(q["id"]), text=q["text"], answer=q["answer"], open_ended=bool(q.get("open_ended", False)))
        for q in data
    ]
    if len(questions) != num_questions:
        on_step(
            f"warning: requested {num_questions} questions, got {len(questions)}"
        )
    return questions


TITLE_SYSTEM_PROMPT = """Generate a short, descriptive title (5-8 words) for
a math worksheet based on the given prompt. Return ONLY the title text -
no quotes, no markdown, no trailing punctuation, no commentary. It is
inserted into a LaTeX document verbatim (not escaped), so avoid characters
with special meaning in LaTeX (%, &, #, _, {, }, ~, ^, \\) unless you intend
them as LaTeX markup."""


def generate_title(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = MODEL,
    on_step: OnStep = _print_step,
) -> str:
    on_step("Generating title...")
    resp = client.messages.create(
        model=model,
        max_tokens=30,
        system=TITLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    title = raw.strip().strip('"')
    on_step(f"Title: {title}")
    return title


HEADER_SYSTEM_PROMPT = """Generate a short paragraph (1-3 sentences) of
instructions or advice for students working on this worksheet, based on the
given prompt (e.g. reminders about formatting or notation, such as not using
a forward slash for fractions). Return ONLY the instructions text - no
quotes, no markdown, no commentary.

It is inserted into a LaTeX document verbatim (not escaped), so:
- Write real LaTeX math notation, delimited with $...$, when referring to a
  formula or notation example (e.g. "write fractions as $\\frac{a}{b}$, not
  a/b"), the same way worksheet questions do.
- Escape any literal %, &, #, _, {, }, ~, ^, or \\ that isn't meant as LaTeX
  markup (e.g. write \\% for a literal percent sign)."""


def generate_header(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = MODEL,
    on_step: OnStep = _print_step,
) -> str:
    on_step("Generating header...")
    resp = client.messages.create(
        model=model,
        max_tokens=120,
        system=HEADER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    header = raw.strip().strip('"')
    on_step(f"Header: {header}")
    return header


_VALID_JSON_ESCAPE_RE = re.compile(r'\\u[0-9a-fA-F]{4}|\\["\\/]')


def _escape_stray_backslashes(text: str) -> str:
    """Doubles backslashes that don't form a pre-existing JSON escape.

    The LLM is asked to emit LaTeX (full of backslashes) inside JSON
    strings; it sometimes forgets to double them (e.g. "\\times" instead
    of "\\\\times"), which json.loads rejects with "Invalid \\escape".
    Only \\\\, \\", \\/ and \\uXXXX are treated as intentional escapes here
    (not \\b \\f \\n \\r \\t) since this pipeline only ever produces LaTeX,
    which has no legitimate use for a literal tab/newline/etc. and commonly
    starts commands with those letters (\\times, \\frac, \\right, ...).
    """
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            match = _VALID_JSON_ESCAPE_RE.match(text, i)
            if match:
                out.append(match.group())
                i = match.end()
            else:
                out.append("\\\\")
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _parse_json_array(raw: str) -> list[dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_escape_stray_backslashes(cleaned))


# --------------------------------------------------------------------------
# Step 2: rendering into \Question{}{} calls
# --------------------------------------------------------------------------

_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters. Run this on all LLM-generated text
    before it goes anywhere near the .tex source."""
    # Backslash must go first so we don't double-escape characters we insert.
    out = []
    for ch in text:
        out.append(_LATEX_SPECIAL_CHARS.get(ch, ch))
    return "".join(out)


_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_CTRL_WORD_RE = re.compile(r"\\[a-zA-Z]+")

# Layout tuning constants (rough estimates, see issue #13 for context).
_ANSWER_BOX_WIDTH_IN = 1.5
_PAGE_TEXT_WIDTH_IN = 6.5
_CHAR_WIDTH_IN = 0.09
_CELL_SPACING_IN = 0.3
_MAX_WIDTH_VARIANCE_RATIO = 1.6
_MIN_QUESTIONS_FOR_GRID = 6


def estimate_question_width(text: str) -> int:
    """Rough character-count estimate of a question's rendered width.
    Strips LaTeX math delimiters and control words (since they don't
    render as visible characters), collapsing \\frac{a}{b} to the width
    of its widest part. Good enough to bucket questions as "short and
    uniform" vs. "too long/variable" for grid layout; not a substitute
    for actually measuring glyphs."""
    s = text.strip().replace("$", "")
    s = _FRAC_RE.sub(lambda m: "X" * max(len(m.group(1)), len(m.group(2))), s)
    s = _CTRL_WORD_RE.sub("", s)
    s = s.replace("{", "").replace("}", "")
    return len(s)


def choose_grid_columns(questions: list[Question]) -> int:
    """Returns how many columns a tabular grid of `questions` should
    use. Falls back to 1 (single vertical column) when there aren't
    enough questions to bother with a grid, or when question widths
    are too uneven for uniform-width columns to look reasonable."""
    if len(questions) < _MIN_QUESTIONS_FOR_GRID:
        return 1

    widths = [estimate_question_width(q.text) for q in questions]
    if max(widths) / max(min(widths), 1) > _MAX_WIDTH_VARIANCE_RATIO:
        return 1

    col_width_in = _ANSWER_BOX_WIDTH_IN + max(widths) * _CHAR_WIDTH_IN + _CELL_SPACING_IN
    return max(1, int(_PAGE_TEXT_WIDTH_IN // col_width_in))


def render_questions(questions: list[Question]) -> str:
    cols = choose_grid_columns(questions)
    if cols <= 1:
        return _render_questions_vertical(questions)
    return _render_questions_grid(questions, cols)


def _render_questions_vertical(questions: list[Question]) -> str:
    # A bare "\n"-joined list of \Question{}{} calls is a single LaTeX
    # paragraph (blank lines, not single newlines, separate paragraphs),
    # so questions ran together instead of stacking one per line. An
    # enumerate environment forces each onto its own line.
    items = "\n".join(
        f"    \\item \\Question{{{escape_latex(q.id)}}}{{{q.text}}}" for q in questions
    )
    return f"\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}"


def _render_questions_grid(questions: list[Question], cols: int) -> str:
    col_width_in = _PAGE_TEXT_WIDTH_IN / cols
    col_spec = f"p{{{col_width_in:.3f}in}}" * cols

    rows = []
    for i in range(0, len(questions), cols):
        row_questions = questions[i : i + cols]
        cells = [f"\\Question{{{escape_latex(q.id)}}}{{{q.text}}}" for q in row_questions]
        cells += [""] * (cols - len(cells))
        rows.append(" & ".join(cells) + r" \\[0.6em]")

    return (
        f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
        + "\n".join(rows)
        + "\n\\end{tabular}"
    )


def fill_template(
    template_path: Path,
    questions_tex: str,
    worksheet_id: Optional[str] = None,
    title: Optional[str] = None,
    header: Optional[str] = None,
) -> str:
    template = template_path.read_text()
    if QUESTIONS_MARKER not in template:
        raise ValueError(
            f"Template does not contain the {QUESTIONS_MARKER} marker. "
            f"Add a line containing exactly {QUESTIONS_MARKER} where "
            f"questions should be inserted."
        )
    filled = template.replace(QUESTIONS_MARKER, questions_tex)

    # Embed the worksheet id (issue #11). Replace the marker with a
    # \WorksheetSetup line when an id is given, otherwise strip the marker so
    # templates lacking an id still compile. Templates without the marker are
    # left untouched for backward compatibility.
    id_setup = (
        rf"\WorksheetSetup{{worksheet id={worksheet_id}}}" if worksheet_id else ""
    )
    filled = filled.replace(WORKSHEET_ID_MARKER, id_setup)

    # Embed the title/instructions block (issue #42). Passed through
    # unescaped, like question text -- the header commonly needs inline math
    # (e.g. "write fractions as $\frac{a}{b}$"), so it's treated as
    # hand-written LaTeX rather than escaped plain text. Empty when neither
    # is given, so templates lacking the marker (or worksheets with no
    # title/header) still compile.
    title_block = ""
    if title or header:
        title_block = r"\WorksheetTitleBlock{%s}{%s}" % (title or "", header or "")
    filled = filled.replace(TITLE_BLOCK_MARKER, title_block)
    return filled


# --------------------------------------------------------------------------
# Step 3: compile, with an LLM repair loop on failure
# --------------------------------------------------------------------------

def compile_tex(tex_path: Path) -> tuple[bool, str]:
    """Compile with latexmk. Returns (success, log_tail)."""
    result = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tex_path.parent}",
            str(tex_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_texinputs_env(),
    )
    success = result.returncode == 0 and (tex_path.with_suffix(".pdf")).exists()
    log = result.stdout + result.stderr
    return success, log[-4000:]  # tail is usually where the error is


REPAIR_SYSTEM_PROMPT = """You are debugging a LaTeX compile failure.
You will be given the full .tex source and the compiler log tail.
Return ONLY the corrected, complete .tex source — no commentary,
no markdown fences. Make the minimal change needed to fix the error(s);
do not alter unrelated content, especially the \\Question{...}{...}
lines' meaning."""


def repair_tex(
    client: anthropic.Anthropic, tex_source: str, log_tail: str, model: str = MODEL
) -> str:
    user_msg = (
        f"Compile log (tail):\n```\n{log_tail}\n```\n\n"
        f"Current .tex source:\n```\n{tex_source}\n```"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=REPAIR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    return re.sub(r"^```(latex|tex)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

class CompileError(Exception):
    """Raised when LaTeX compilation fails after exhausting repair attempts."""

    def __init__(self, log_tail: str):
        self.log_tail = log_tail
        super().__init__(log_tail)


def generate_worksheet(
    client: anthropic.Anthropic,
    template_path: Path,
    prompt: str,
    out: Path,
    num_questions: int,
    max_repairs: int,
    bucket: str = None,
    db_path: Path = Path("worksheets.sqlite3"),
    title: Optional[str] = None,
    header: Optional[str] = None,
    model: str = MODEL,
    on_step: OnStep = _print_step,
) -> tuple[Path, list, "storage.WorksheetRecord"]:
    """Runs the full prompt -> questions -> LaTeX -> compile (with repair
    retries) -> optional S3/DB storage pipeline. Raises CompileError if
    compilation still fails after `max_repairs` retries. Returns
    (tex_path, questions, record), where `record` is None if `bucket` is
    not given.

    `title`/`header` are generated from the prompt when not given, same as
    `generate_title` always did -- but now unconditionally (not just when
    `bucket` is set), since both are rendered into the worksheet body itself
    (issue #42), not only used for storage/filenames.
    """
    questions = generate_questions(client, prompt, num_questions, model=model, on_step=on_step)
    if title is None:
        title = generate_title(client, prompt, model=model, on_step=on_step)
    if header is None:
        header = generate_header(client, prompt, model=model, on_step=on_step)
    document = WorksheetDocument(title=title, header=header, questions=questions)
    return build_worksheet(
        document,
        template_path,
        out,
        max_repairs,
        client=client,
        bucket=bucket,
        db_path=db_path,
        prompt=prompt,
        model=model,
        on_step=on_step,
    )


def build_worksheet(
    document: "WorksheetDocument",
    template_path: Path,
    out: Path,
    max_repairs: int,
    client: Optional[anthropic.Anthropic] = None,
    bucket: str = None,
    db_path: Path = Path("worksheets.sqlite3"),
    prompt: str = "",
    model: str = MODEL,
    on_step: OnStep = _print_step,
) -> tuple[Path, list, "storage.WorksheetRecord"]:
    """Runs the questions -> LaTeX -> compile (with repair retries) ->
    optional S3/DB storage tail of the pipeline on a ready `WorksheetDocument`.

    Shared by the AI path (`generate_worksheet`) and the manual path
    (`create_worksheet_from_questions`). The compile/repair loop only asks the
    model to repair when `client` is given and `max_repairs > 0`; with
    `max_repairs=0` it compiles exactly once and raises `CompileError` on
    failure without touching `client`. Returns (tex_path, questions, record),
    where `record` is None if `bucket` is not given.
    """
    questions = document.questions
    public_id = generate_worksheet_id()
    questions_tex = render_questions(questions)
    tex_source = fill_template(
        template_path,
        questions_tex,
        worksheet_id=public_id,
        title=document.title,
        header=document.header,
    )

    out = Path(out)
    out_dir = out.parent if out.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{out.name}.tex"
    tex_path.write_text(tex_source)

    # Render the QR code the LaTeX embeds. The filename must match the
    # \includegraphics{qr_<id>} in gbworksheet.sty, and it lives next to the
    # .tex so every compile (student/cv/answer-key) can find it.
    render_qr_png(public_id, out_dir / f"qr_{public_id}.png")

    for attempt in range(max_repairs + 1):
        on_step(f"Compiling (attempt {attempt + 1})...")
        success, log_tail = compile_tex(tex_path)
        if success:
            on_step(f"Done: {tex_path.with_suffix('.pdf')}")
            record = None
            if bucket:
                on_step(f"Uploading to s3://{bucket} and recording in {db_path}...")
                record = storage.store_worksheet(
                    tex_path=tex_path,
                    questions=questions,
                    prompt=prompt,
                    model=model,
                    bucket=bucket,
                    db_path=db_path,
                    title=document.title,
                    header=document.header,
                    public_id=public_id,
                )
                on_step(f"Stored worksheet id={record.id}")
                on_step(
                    "PDF URLs:",
                    f"student: {record.student_pdf_s3url}\ncv:      {record.cv_pdf_s3url}\nanswers: {record.answers_pdf_s3url}",
                )
            return tex_path, questions, record
        if attempt == max_repairs:
            raise CompileError(log_tail)
        on_step("Compile failed, asking model to repair...", log_tail)
        tex_source = repair_tex(client, tex_source, log_tail, model=model)
        tex_path.write_text(tex_source)


# --------------------------------------------------------------------------
# Manual entry: build a worksheet from a user-supplied questions JSON
# --------------------------------------------------------------------------

def parse_questions_json(raw: str) -> "list[Question]":
    """Parses a user-supplied JSON array of questions into `Question`s.

    Accepts the same shape `generate_questions` produces —
    `[{"id": ..., "text": ..., "answer": ...}]` — and reuses `_parse_json_array`
    so pasted JSON gets the same markdown-fence stripping and stray-LaTeX-
    backslash tolerance. Raises `ValueError` with a human-readable message if
    the payload isn't a non-empty array of objects each carrying id/text/answer.
    """
    try:
        data = _parse_json_array(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Questions must be valid JSON: {e}") from e

    if not isinstance(data, list) or not data:
        raise ValueError("Questions JSON must be a non-empty array of objects.")

    questions = []
    for i, q in enumerate(data):
        if not isinstance(q, dict) or not {"id", "text", "answer"} <= q.keys():
            raise ValueError(
                f"Question at index {i} must be an object with "
                f'"id", "text", and "answer" keys.'
            )
        questions.append(
            Question(id=str(q["id"]), text=q["text"], answer=q["answer"], open_ended=bool(q.get("open_ended", False)))
        )
    return questions


def create_worksheet_from_questions(
    questions_json: str,
    template_path: Path,
    out: Path,
    title: str,
    header: str = "",
    bucket: str = None,
    db_path: Path = Path("worksheets.sqlite3"),
    on_step: OnStep = _print_step,
) -> tuple[Path, list, "storage.WorksheetRecord"]:
    """Manual worksheet-creation entry point (issue #12): builds a worksheet
    from a user-supplied questions JSON string instead of generating questions
    with the LLM. Compiles exactly once (no AI repair loop) and stores the
    result like the AI path. Raises `ValueError` on bad JSON and `CompileError`
    if the LaTeX doesn't compile. The stored `model` is the literal "manual".
    """
    questions = parse_questions_json(questions_json)
    document = WorksheetDocument(title=title, header=header, questions=questions)
    return build_worksheet(
        document,
        template_path,
        out,
        max_repairs=0,
        client=None,
        bucket=bucket,
        db_path=db_path,
        prompt="",
        model="manual",
        on_step=on_step,
    )


def main():
    from dotenv import load_dotenv
    import os

    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate a math worksheet PDF from a prompt.")
    parser.add_argument("--template", required=True, type=Path, help="Path to .tex template")
    parser.add_argument("--prompt", required=True, help="Worksheet description")
    parser.add_argument("--out", default="worksheet", help="Output basename (no extension)")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument(
        "--model",
        default=MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Claude model used for generation (default: {MODEL})",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Worksheet title, rendered on the page and used as a filename "
        "prefix and DB record. Auto-generated from the prompt if not given.",
    )
    parser.add_argument(
        "--header",
        default=None,
        help="Worksheet instructions/header paragraph, rendered on the page "
        "below the title. Auto-generated from the prompt if not given. "
        "Inserted into the LaTeX source as-is (like question text), so use "
        "$...$ for inline math and escape any literal LaTeX special "
        "characters.",
    )
    parser.add_argument("--save-json", action="store_true", help="Also save the raw question JSON")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket to upload PDFs to. Defaults to the S3_BUCKET env var. "
        "If unset, the worksheet is compiled but not stored.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("worksheets.sqlite3"),
        help="SQLite database file to record worksheet metadata in",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])  # picks up ANTHROPIC_API_KEY from env

    try:
        tex_path, questions, record = generate_worksheet(
            client,
            args.template,
            args.prompt,
            Path(args.out),
            args.num_questions,
            args.max_repairs,
            bucket=args.bucket,
            db_path=args.db_path,
            title=args.title,
            header=args.header,
            model=args.model,
        )
    except CompileError as e:
        print("Failed after max repair attempts. Log tail:\n" + e.log_tail, file=sys.stderr)
        sys.exit(1)

    if args.save_json:
        Path(f"{args.out}.json").write_text(
            json.dumps([asdict(q) for q in questions], indent=2)
        )

    if not record:
        print("No --bucket/S3_BUCKET set; skipping upload and DB storage.", file=sys.stderr)


if __name__ == "__main__":
    main()