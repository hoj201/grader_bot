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
from typing import Dict, List, Union

from graderbot import QuestionResult, extract_name, grade_hw, load_image_rgb, read_worksheet_id
from storage import deserialize_boxes, get_worksheet_by_public_id, init_db

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
        # Group scans by decoded worksheet id, loading each image once.
        grouped: Dict[str, List[tuple]] = defaultdict(list)
        result = ScanBatchResult()
        for hw in hws:
            image = load_image_rgb(str(hw))
            worksheet_id = read_worksheet_id(image)
            if not worksheet_id:
                result.unreadable.append(str(hw))
                continue
            grouped[worksheet_id].append((str(hw), image))

        for worksheet_id, items in grouped.items():
            record = get_worksheet_by_public_id(conn, worksheet_id)
            if record is None or not record.boxes_json or not record.questions_json:
                result.unknown_worksheets[worksheet_id] = [hw for hw, _ in items]
                continue

            answer_key = {q["id"]: q["answer"] for q in json.loads(record.questions_json)}
            boxes = deserialize_boxes(record.boxes_json)
            name_box = boxes.get(_NAME_BOX_ID)
            question_boxes = {
                qid: box for qid, box in boxes.items() if qid != _NAME_BOX_ID
            }

            student_results: Dict[str, StudentResults] = {}
            for _hw, image in items:
                name = extract_name(image, name_box, roster) if name_box is not None else ""
                student_results[name] = grade_hw(answer_key, question_boxes, image)
            result.results_by_worksheet[worksheet_id] = student_results

        return result
    finally:
        conn.close()
