#!/usr/bin/env python3
"""
Worksheet generator agent.

Pipeline:
  prompt -> LLM generates structured question JSON
         -> render each question via \\Question{id}{text}, LaTeX-escaping the id
         -> insert into template at %%QUESTIONS%% marker
         -> compile with latexmk
         -> on LaTeX compile error, feed log back to LLM to repair, retry

Requires: ANTHROPIC_API_KEY env var, `anthropic` python package, a LaTeX
distribution with `latexmk` on PATH.

Usage:
  python worksheetbot.py --template worksheet_template.tex \
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

import anthropic

import storage

MODEL = "claude-sonnet-4-6"
QUESTIONS_MARKER = "%%QUESTIONS%%"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    text: str
    answer: str


# --------------------------------------------------------------------------
# Step 1: generate structured question data from the prompt
# --------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """You are a arithmetic and algebra worksheet author.
Given a topic/prompt, produce exam-quality questions as JSON only — no prose,
no markdown fences, no commentary.

Output schema (a JSON array):
[
  {"id": "1", "text": "<question text, real LaTeX>", "answer": "<final answer>"}
]

Rules:
- Write question text as real LaTeX math notation, delimited with $...$
  (e.g. "$x^2 + 3x - 4 = 0$", "$\\frac{1}{4}+\\frac{1}{2}=$"). It is inserted
  into the worksheet .tex source verbatim, so it must be valid LaTeX.
- Answers must be a plain number (e.g. "12", "1.5") or a LaTeX fraction
  written as \\frac{a}{b} (e.g. "\\frac{3}{4}") — never a bare "a/b" slash
  expression, and never a worked solution or justification.
- Vary question difficulty and phrasing; avoid near-duplicate questions.
- Match the count, topic, and difficulty level requested in the prompt exactly.
"""


def generate_questions(client: anthropic.Anthropic, prompt: str, num_questions: int) -> list[Question]:
    user_msg = f"{prompt}\n\nGenerate exactly {num_questions} questions."
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    data = _parse_json_array(raw)

    questions = [Question(id=str(q["id"]), text=q["text"], answer=q["answer"]) for q in data]
    if len(questions) != num_questions:
        print(
            f"warning: requested {num_questions} questions, got {len(questions)}",
            file=sys.stderr,
        )
    return questions


def _parse_json_array(raw: str) -> list[dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


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


def render_questions(questions: list[Question]) -> str:
    lines = []
    for q in questions:
        qid = escape_latex(q.id)
        lines.append(f"\\Question{{{qid}}}{{{q.text}}}")
    return "\n".join(lines)


def fill_template(template_path: Path, questions_tex: str) -> str:
    template = template_path.read_text()
    if QUESTIONS_MARKER not in template:
        raise ValueError(
            f"Template does not contain the {QUESTIONS_MARKER} marker. "
            f"Add a line containing exactly {QUESTIONS_MARKER} where "
            f"questions should be inserted."
        )
    return template.replace(QUESTIONS_MARKER, questions_tex)


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


def repair_tex(client: anthropic.Anthropic, tex_source: str, log_tail: str) -> str:
    user_msg = (
        f"Compile log (tail):\n```\n{log_tail}\n```\n\n"
        f"Current .tex source:\n```\n{tex_source}\n```"
    )
    resp = client.messages.create(
        model=MODEL,
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
) -> tuple[Path, list, "storage.WorksheetRecord"]:
    """Runs the full prompt -> questions -> LaTeX -> compile (with repair
    retries) -> optional S3/DB storage pipeline. Raises CompileError if
    compilation still fails after `max_repairs` retries. Returns
    (tex_path, questions, record), where `record` is None if `bucket` is
    not given.
    """
    print(f"Generating {num_questions} questions...", file=sys.stderr)
    questions = generate_questions(client, prompt, num_questions)

    questions_tex = render_questions(questions)
    tex_source = fill_template(template_path, questions_tex)

    out = Path(out)
    out_dir = out.parent if out.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{out.name}.tex"
    tex_path.write_text(tex_source)

    for attempt in range(max_repairs + 1):
        print(f"Compiling (attempt {attempt + 1})...", file=sys.stderr)
        success, log_tail = compile_tex(tex_path)
        if success:
            print(f"Done: {tex_path.with_suffix('.pdf')}", file=sys.stderr)
            record = None
            if bucket:
                print(f"Uploading to s3://{bucket} and recording in {db_path}...", file=sys.stderr)
                record = storage.store_worksheet(
                    tex_path=tex_path,
                    questions=questions,
                    prompt=prompt,
                    model=MODEL,
                    bucket=bucket,
                    db_path=db_path,
                )
                print(f"Stored worksheet id={record.id}", file=sys.stderr)
                print(f"  student: {record.student_pdf_s3url}", file=sys.stderr)
                print(f"  cv:      {record.cv_pdf_s3url}", file=sys.stderr)
                print(f"  answers: {record.answers_pdf_s3url}", file=sys.stderr)
            return tex_path, questions, record
        if attempt == max_repairs:
            raise CompileError(log_tail)
        print("Compile failed, asking model to repair...", file=sys.stderr)
        tex_source = repair_tex(client, tex_source, log_tail)
        tex_path.write_text(tex_source)


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