"""Grade scanned student worksheets straight from the SQLite database.

Each worksheet carries its public id as an on-page QR code (issue #11). Given a
pile of scans, this module decodes each scan's id, looks the worksheet up in the
database, and grades it against the exact stored answer key and answer-box
locations -- no separately-supplied answer key, and no re-rendering of the
worksheet. A single call can mix scans from different worksheets: they are
grouped by decoded id and each group graded against its own worksheet.

The heavy CV/OCR work is reused from `graderbot` (`load_image_rgb`,
`read_worksheet_id`, `extract_name`, `grade_hw`); the answer key and box
locations come from `storage` (`get_worksheet_by_public_id`,
`deserialize_boxes`).
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from graderbot.grading import grade_hw
from graderbot.imaging import load_scan_pages
from graderbot.models import Box, QuestionResult
from graderbot.ocr import extract_name
from graderbot.registration import read_worksheet_id, rectify_to_canonical
from graderbot.markup import render_marked_page
from graderbot.storage import deserialize_boxes, get_worksheet_by_public_id, images_to_pdf, init_db

_NAME_BOX_ID = "name"

# Per-student grading: question id -> QuestionResult (answer/response/correct).
StudentResults = Dict[str, QuestionResult]


@dataclass
class ScanBatchResult:
    """Outcome of grading a pile of scans.

    - `results_by_worksheet`: decoded worksheet id -> {student name -> per-question
      results}, where the per-question results are {question id -> QuestionResult}.
    - `unreadable`: scan paths whose QR id could not be decoded.
    - `unknown_worksheets`: decoded id -> scan paths, for ids with no matching
      (or incompletely stored) database row.
    """

    results_by_worksheet: Dict[str, Dict[str, StudentResults]] = field(default_factory=dict)
    unreadable: List[str] = field(default_factory=list)
    unknown_worksheets: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class _GradedScan:
    """A single graded page, carrying everything a marked-up render needs: the
    rectified canonical-frame image, the question boxes, and the results."""

    worksheet_id: str
    name: str
    image: np.ndarray
    question_boxes: Dict[str, Box]
    results: StudentResults


def _grade_batch(
    hws: List[Union[str, Path]],
    roster: List[str],
    conn: Connection,
) -> Tuple[ScanBatchResult, List[_GradedScan]]:
    """Shared core of `grade_scans`/`mark_scan`: rectifies every scan page to the
    canonical frame, groups pages by decoded worksheet id, and grades each group
    against its stored answer key. Grading runs on the rectified image so a
    marked-up render lands its marks in the same aligned frame. Returns the
    summary result plus, in scan order, the per-page graded material."""
    # Group rectified pages by decoded worksheet id.
    grouped: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    result = ScanBatchResult()
    for hw in hws:
        pages = load_scan_pages(str(hw))
        for page_index, page in enumerate(pages):
            label = str(hw) if len(pages) == 1 else f"{hw} (page {page_index + 1})"
            try:
                rectified = rectify_to_canonical(page)
            except ValueError:
                result.unreadable.append(label)
                continue
            worksheet_id = read_worksheet_id(rectified)
            if not worksheet_id:
                result.unreadable.append(label)
                continue
            grouped[worksheet_id].append((label, rectified))

    graded_scans: List[_GradedScan] = []
    for worksheet_id, items in grouped.items():
        record = get_worksheet_by_public_id(conn, worksheet_id)
        if record is None or not record.boxes_json or not record.questions_json:
            result.unknown_worksheets[worksheet_id] = [label for label, _ in items]
            continue

        answer_key = {q["id"]: q["answer"] for q in json.loads(record.questions_json)}
        boxes = deserialize_boxes(record.boxes_json)
        name_box = boxes.get(_NAME_BOX_ID)
        question_boxes = {
            qid: box for qid, box in boxes.items() if qid != _NAME_BOX_ID
        }

        student_results: Dict[str, StudentResults] = {}
        for _label, image in items:
            name = extract_name(image, name_box, roster) if name_box is not None else ""
            results = grade_hw(answer_key, question_boxes, image)
            student_results[name] = results
            graded_scans.append(
                _GradedScan(worksheet_id, name, image, question_boxes, results)
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
) -> ScanBatchResult:
    """Grades each scan in `hws` against the worksheet its QR code identifies,
    fetched from the database at `db_path`. Student names are resolved against
    `roster`. See `ScanBatchResult` for the return shape."""
    conn = init_db(Path(db_path))
    try:
        result, _ = _grade_batch(hws, roster, conn)
        return result
    finally:
        conn.close()


def mark_scan(
    hws: List[Union[str, Path]],
    roster: List[str],
    db_path: Union[str, Path],
    out_path: Union[str, Path],
) -> ScanBatchResult:
    """Grades `hws` exactly like `grade_scans` and, in addition, writes a single
    combined marked-up PDF to `out_path` -- one page per successfully graded
    scan, each stamped with a score header and the correct answers beside the
    wrong ones. Returns the same `ScanBatchResult` (scans that were unreadable or
    whose worksheet is unknown contribute no page). No PDF is written if nothing
    graded."""
    conn = init_db(Path(db_path))
    try:
        result, graded = _grade_batch(hws, roster, conn)
        if graded:
            marked_bgr = [
                cv2.cvtColor(
                    render_marked_page(scan.image, scan.results, scan.question_boxes),
                    cv2.COLOR_RGB2BGR,
                )
                for scan in graded
            ]
            images_to_pdf(marked_bgr, Path(out_path))
        return result
    finally:
        conn.close()
