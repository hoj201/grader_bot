"""Grade scanned student worksheets straight from the SQLite database.

Each worksheet carries its public id as an on-page QR code (issue #11). Given a
pile of scans, this module decodes each scan's id, looks the worksheet up in the
database, and grades it against the exact stored answer key and answer-box
locations -- no separately-supplied answer key, and no re-rendering of the
worksheet. A single call can mix scans from different worksheets: they are
grouped by decoded id and each group graded against its own worksheet.

The heavy CV/OCR work is reused from `graderbot` (`load_image_rgb`,
`read_worksheet_id`, `grade_hw`); the answer key and box locations come from
`storage` (`get_worksheet_by_public_id`, `deserialize_boxes`). Which student
wrote a page is decided by a `name_reader.NameReader` -- Tesseract by default,
or the trained handwriting classifier (issue #58).
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from graderbot.grading import grade_hw
from graderbot.imaging import load_scan_pages
from graderbot.models import Box, QuestionResult
from graderbot.name_reader import NameGuess, NameReader, OcrNameReader
from graderbot.registration import read_worksheet_id, rectify_to_canonical
from graderbot.markup import render_marked_page
from graderbot.storage import deserialize_boxes, get_worksheet_by_public_id, images_to_pdf, init_db

_NAME_BOX_ID = "name"

# Per-student grading: question id -> QuestionResult (answer/response/correct).
StudentResults = Dict[str, QuestionResult]

# Progress callback, mirroring graderbot.worksheetbot.OnStep: a short message and
# an optional multi-line detail string. Used to stream grading progress into the
# Streamlit UI (or the console) so a failed QR/rectify can be pinned to a page.
OnStep = Callable[[str, Optional[str]], None]


def _print_step(msg: str, detail: Optional[str] = None) -> None:
    """Default `on_step`: prints to stderr, optionally with detail."""
    print(msg, file=sys.stderr)
    if detail:
        print(detail, file=sys.stderr)


@dataclass
class NamePrediction:
    """How one page's student name was identified (issue #58). Kept per page --
    unlike `results_by_worksheet`, which is keyed by name and so collapses two
    pages that resolved to the same student -- so the Grade tab can show every
    page's read and its confidence."""

    page: str
    worksheet_id: str
    name: str
    confidence: float
    source: str


@dataclass
class ScanBatchResult:
    """Outcome of grading a pile of scans.

    - `results_by_worksheet`: decoded worksheet id -> {student name -> per-question
      results}, where the per-question results are {question id -> QuestionResult}.
    - `unreadable`: scan paths whose QR id could not be decoded.
    - `unknown_worksheets`: decoded id -> scan paths, for ids with no matching
      (or incompletely stored) database row.
    - `name_predictions`: one entry per graded page, in scan order, recording
      the name read off it and how confident the reader was.
    """

    results_by_worksheet: Dict[str, Dict[str, StudentResults]] = field(default_factory=dict)
    unreadable: List[str] = field(default_factory=list)
    unknown_worksheets: Dict[str, List[str]] = field(default_factory=dict)
    name_predictions: List[NamePrediction] = field(default_factory=list)


@dataclass
class _GradedScan:
    """A single graded page, carrying everything a marked-up render needs: the
    rectified canonical-frame image, the question boxes, and the results."""

    worksheet_id: str
    name: str
    image: np.ndarray
    question_boxes: Dict[str, Box]
    results: StudentResults
    confidence: float = 0.0
    name_source: str = ""


def _grade_batch(
    hws: List[Union[str, Path]],
    roster: List[str],
    conn: Connection,
    on_step: OnStep = _print_step,
    name_reader: Optional[NameReader] = None,
) -> Tuple[ScanBatchResult, List[_GradedScan]]:
    """Shared core of `grade_scans`/`mark_scan`: rectifies every scan page to the
    canonical frame, groups pages by decoded worksheet id, and grades each group
    against its stored answer key. Grading runs on the rectified image so a
    marked-up render lands its marks in the same aligned frame. Returns the
    summary result plus, in scan order, the per-page graded material.

    `name_reader` identifies the student on each page; it defaults to
    `OcrNameReader(roster)`, the Tesseract path grading has always used. Pass a
    `ClassifierNameReader` to use the trained handwriting classifier instead
    (issue #58).

    `on_step(msg, detail)` receives per-page progress, separating the two ways a
    page becomes "unreadable" -- rectification (ArUco markers not found) versus
    QR-code decode -- so a failure can be pinned to the exact page and stage."""
    name_reader = name_reader if name_reader is not None else OcrNameReader(roster)
    # Group rectified pages by decoded worksheet id.
    grouped: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    result = ScanBatchResult()
    for hw in hws:
        pages = load_scan_pages(str(hw))
        on_step(f"Loaded {hw}: {len(pages)} page(s).")
        for page_index, page in enumerate(pages):
            label = str(hw) if len(pages) == 1 else f"{hw} (page {page_index + 1})"
            try:
                rectified = rectify_to_canonical(page)
            except ValueError:
                result.unreadable.append(label)
                on_step(
                    f"{label}: could not rectify to canonical frame "
                    "(ArUco registration markers not found)."
                )
                continue
            worksheet_id = read_worksheet_id(rectified)
            if not worksheet_id:
                result.unreadable.append(label)
                on_step(
                    f"{label}: rectified OK, but the worksheet QR code "
                    "could not be decoded."
                )
                continue
            on_step(f"{label}: decoded worksheet id={worksheet_id}.")
            grouped[worksheet_id].append((label, rectified))

    graded_scans: List[_GradedScan] = []
    for worksheet_id, items in grouped.items():
        record = get_worksheet_by_public_id(conn, worksheet_id)
        if record is None or not record.boxes_json or not record.questions_json:
            result.unknown_worksheets[worksheet_id] = [label for label, _ in items]
            on_step(
                f"Worksheet id={worksheet_id} not found (or incompletely stored) "
                f"in the database; skipping {len(items)} page(s)."
            )
            continue

        questions_data = json.loads(record.questions_json)
        answer_key = {q["id"]: q["answer"] for q in questions_data}
        open_ended_key = {q["id"]: q.get("open_ended", False) for q in questions_data}
        boxes = deserialize_boxes(record.boxes_json)
        name_box = boxes.get(_NAME_BOX_ID)
        question_boxes = {
            qid: box for qid, box in boxes.items() if qid != _NAME_BOX_ID
        }

        # Read every page's name in one batch: a remote embedder then embeds the
        # whole group in a single API call instead of one per page.
        if name_box is not None:
            guesses = name_reader.read_many([image for _, image in items], name_box)
        else:
            guesses = [NameGuess("", 0.0, "none") for _ in items]

        student_results: Dict[str, StudentResults] = {}
        for (label, image), guess in zip(items, guesses):
            results = grade_hw(answer_key, question_boxes, image, open_ended_key)
            student_results[guess.name] = results
            graded = [r for r in results.values() if not r.open_ended]
            n_correct = sum(1 for r in graded if r.correct)
            on_step(
                f"{label}: graded {guess.name or '(no name)'} "
                f"[{guess.source} {guess.confidence:.0%}] -- "
                f"{n_correct}/{len(graded)} correct."
            )
            result.name_predictions.append(
                NamePrediction(
                    page=label,
                    worksheet_id=worksheet_id,
                    name=guess.name,
                    confidence=guess.confidence,
                    source=guess.source,
                )
            )
            graded_scans.append(
                _GradedScan(
                    worksheet_id,
                    guess.name,
                    image,
                    question_boxes,
                    results,
                    guess.confidence,
                    guess.source,
                )
            )
        result.results_by_worksheet[worksheet_id] = student_results

    return result, graded_scans


def results_by_student(result: ScanBatchResult) -> Dict[str, Dict[str, StudentResults]]:
    """Transposes a `ScanBatchResult` into the issue-#23 display shape:
    {student name -> {worksheet id -> {question id -> QuestionResult}}}."""
    by_student: Dict[str, Dict[str, StudentResults]] = defaultdict(dict)
    for worksheet_id, students in result.results_by_worksheet.items():
        for name, question_results in students.items():
            by_student[name][worksheet_id] = question_results
    return dict(by_student)


def grade_scans(
    hws: List[Union[str, Path]],
    roster: List[str],
    db_path: Union[str, Path],
    on_step: OnStep = _print_step,
    name_reader: Optional[NameReader] = None,
) -> ScanBatchResult:
    """Grades each scan in `hws` against the worksheet its QR code identifies,
    fetched from the database at `db_path`. Student names are resolved against
    `roster` by OCR unless `name_reader` overrides that (see `_grade_batch`).
    `on_step` streams per-page progress. See `ScanBatchResult` for the return
    shape."""
    conn = init_db(Path(db_path))
    try:
        result, _ = _grade_batch(hws, roster, conn, on_step=on_step, name_reader=name_reader)
        return result
    finally:
        conn.close()


def mark_scan(
    hws: List[Union[str, Path]],
    roster: List[str],
    db_path: Union[str, Path],
    out_path: Union[str, Path],
    on_step: OnStep = _print_step,
    name_reader: Optional[NameReader] = None,
) -> ScanBatchResult:
    """Grades `hws` exactly like `grade_scans` and, in addition, writes a single
    combined marked-up PDF to `out_path` -- one page per successfully graded
    scan, each showing the correct answers beside the wrong ones. `on_step`
    streams per-page progress (see `_grade_batch`). Returns the same
    `ScanBatchResult` (scans that were unreadable or whose worksheet is
    unknown contribute no page). No PDF is written if nothing graded."""
    conn = init_db(Path(db_path))
    try:
        result, graded = _grade_batch(hws, roster, conn, on_step=on_step, name_reader=name_reader)
        if graded:
            on_step(f"Rendering marked-up PDF ({len(graded)} page(s))...")
            # Build the marked-up pages one scan at a time and drop each
            # scan's rectified image as soon as its marked-up copy exists,
            # instead of holding a full rectified image *and* a full
            # marked-up image for every page in the batch at once. A large
            # multi-page scan otherwise doubles its peak image memory here,
            # which is tight on the 1GB fly.io VM this runs on.
            marked_bgr = []
            for scan in graded:
                marked_bgr.append(
                    cv2.cvtColor(
                        render_marked_page(scan.image, scan.results, scan.question_boxes),
                        cv2.COLOR_RGB2BGR,
                    )
                )
                scan.image = None
            images_to_pdf(marked_bgr, Path(out_path))
            on_step(f"Wrote marked-up PDF to {out_path}.")
        return result
    finally:
        conn.close()
