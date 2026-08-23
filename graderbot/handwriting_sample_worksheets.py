"""Generates "copy worksheets" for issue #81's real-handwriting-data
harvesting: an ordinary graderbot worksheet (same QR code, box layout, and
storage as any other -- built via `worksheetbot.build_worksheet`, the exact
function every other worksheet-creation path in the app already uses)
except each question already prints its own answer, and the student is only
asked to copy it into the box beside it.

Because the "correct answer" is known by construction -- it's literally what
was printed -- every filled-in box scanned back in is a trustworthy (crop,
text) label for training/evaluating `response_scorer.CnnResponseScorer`,
with no OCR guess to confirm/correct the way `scripts/label_handwriting.py`'s
Mathpix-seeded review pass needs. `handwriting_harvest.py` is what turns a
scan of the printed result back into `HANDWRITING_LABEL` rows.

Target strings are drawn from `response_candidates.SAMPLE_ANSWER_POOL`, the
same domain distribution `training/dataset.py` trains the synthetic side of
the model on, so the real and synthetic halves of training data cover the
same answer shapes.
"""

import random
from pathlib import Path
from typing import List, Optional

from graderbot.response_candidates import SAMPLE_ANSWER_POOL
from graderbot.worksheetbot import (
    OnStep,
    Question,
    WorksheetDocument,
    _print_step,
    build_worksheet,
)

DEFAULT_TITLE = "Handwriting Practice"
_HEADER = (
    "Copy each number exactly as it is printed into the empty box beside it. "
    "Write neatly, one number per box."
)
# \Question{id}{text} (questions.sty) renders text immediately to the left
# of a blank answer box -- \Large/\bfseries makes the printed target easy to
# read and copy without the box itself needing to be any bigger than a
# normal answer box (the geometry harvested crops train on should match
# real answer boxes, not be a special larger size).
_QUESTION_TEXT_TEMPLATE = r"\textbf{{\Large ${target}$}}"


def generate_copy_questions(count: int, rng: Optional[random.Random] = None) -> List[Question]:
    """Builds `count` copy-practice questions, each printing one target
    string (from `SAMPLE_ANSWER_POOL`, shuffled and cycled so a `count`
    larger than the pool still produces a full set, just with repeats) as
    both `text` (what's printed for the student to copy) and `answer` (the
    ground-truth label harvested crops are stored with) -- the two are
    identical by construction, which is the entire point of this worksheet
    (issue #81)."""
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = rng if rng is not None else random.Random()
    pool = list(SAMPLE_ANSWER_POOL)
    rng.shuffle(pool)
    targets = [pool[i % len(pool)] for i in range(count)]
    return [
        Question(
            id=f"hw{i + 1}",
            text=_QUESTION_TEXT_TEMPLATE.format(target=target),
            answer=target,
            open_ended=False,
        )
        for i, target in enumerate(targets)
    ]


def build_handwriting_sample_worksheet(
    count: int,
    template_path: Path,
    out: Path,
    bucket: str,
    db_path: Path,
    title: str = DEFAULT_TITLE,
    rng: Optional[random.Random] = None,
    on_step: OnStep = _print_step,
):
    """Builds and stores a copy-practice worksheet with `count` questions
    (see `generate_copy_questions`) through the same
    `worksheetbot.build_worksheet` pipeline every other worksheet uses --
    same QR code, box layout, and `WORKSHEET` row, so it can be scanned
    back in with the ordinary registration/QR machinery. Compiles exactly
    once (`max_repairs=0`, no LLM involved -- there's nothing to repair
    since the LaTeX shape is fixed and machine-generated). Returns
    `(tex_path, questions, record)`, same shape as `build_worksheet`.
    """
    questions = generate_copy_questions(count, rng)
    document = WorksheetDocument(title=title, header=_HEADER, questions=questions)
    return build_worksheet(
        document,
        template_path,
        out,
        max_repairs=0,
        client=None,
        bucket=bucket,
        db_path=db_path,
        prompt="",
        model="handwriting_sample",
        on_step=on_step,
    )
