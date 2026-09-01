"""Uploads generated worksheet PDFs to S3 and records them in SQLite.

See the `WORKSHEET` table created by `init_db` for the persisted schema.
The SQLite file is expected to be replicated to S3 by litestream
(`litestream.yml`), which is why `init_db` enables WAL journal mode.
"""

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection, connect
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import cv2
import fitz
import numpy as np

from graderbot.models import Box
from graderbot.worksheet_boxes import extract_answer_boxes
from graderbot.worksheet_synth import WORKSHEET_STY_PATH, fill_worksheet, latexmk_worksheet

if TYPE_CHECKING:
    from graderbot.worksheetbot import Question


@dataclass
class WorksheetRecord:
    prompt: str
    tex_source: str
    questions_json: str
    model: str
    num_questions: int
    title: Optional[str] = None
    header: Optional[str] = None
    public_id: Optional[str] = None
    boxes_json: Optional[str] = None
    student_pdf_s3url: Optional[str] = None
    cv_pdf_s3url: Optional[str] = None
    answers_pdf_s3url: Optional[str] = None
    sty_hash: Optional[str] = None
    created_at: Optional[str] = None
    id: Optional[int] = None


@dataclass
class ClassroomRecord:
    """A group of students whose name-learning worksheets are ingested and
    managed together (issue #43)."""
    label: str
    id: Optional[int] = None


@dataclass
class StudentRecord:
    """One student enrolled in a `ClassroomRecord` (issue #43)."""
    classroom_id: int
    first_name: str
    last_name: str
    nickname: Optional[str] = None
    id: Optional[int] = None


@dataclass
class CsvImportResult:
    """Result of a bulk `import_students_csv` call: the students that were
    added (or already existed) and the rows that were skipped, with a
    human-readable reason for each."""
    added: List[StudentRecord]
    skipped: List[str]


@dataclass
class NameImageRecord:
    """One handwriting sample cropped from a scanned name-collection sheet
    (issue #2/#43): tied to a `StudentRecord`, the crop lives in S3."""
    student_id: int
    box_id: str
    image_s3url: str
    image_sha256: str
    created_at: Optional[str] = None
    id: Optional[int] = None


@dataclass
class NameEmbeddingRecord:
    """A vector embedding of one `NameImageRecord` (issue #46), stored as its
    own S3 object. One row per name image (1:1), matching the KNN
    classifier's need for one training vector per handwriting sample."""
    student_id: int
    name_image_id: int
    embedding_s3url: str
    created_at: Optional[str] = None
    id: Optional[int] = None


@dataclass
class PendingNameLabelRecord:
    """One low/no-confidence name-box crop captured during grading (issue
    #92), waiting for a human to say which student actually wrote it. Lets
    the "Label names" tab draw a random unresolved crop, show the reader's
    own guess (`predicted_name`/`confidence`/`source`) as a hint, and turn a
    confirmed answer straight into a `NameImageRecord` -- the same training
    data `ingest_name_sheets` produces, just sourced from real graded scans
    instead of dedicated name-collection sheets. A row is deleted once
    resolved (assigned or discarded), so this table only ever holds the
    current queue, not a permanent history."""
    classroom_id: int
    image_s3url: str
    image_sha256: str
    box_id: str = "name"
    predicted_name: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    id: Optional[int] = None


@dataclass
class HandwritingLabelRecord:
    """One human-confirmed (crop, text) pair for training/evaluating
    `response_scorer.CnnResponseScorer` (issue #81) -- the "dedicated
    labeling pass" from that issue's Data pipeline section, distinct from
    the bulk *synthetic* training data `answer_glyph_synth` generates.

    `source_mathpix_call_id` optionally links back to the `MATHPIX_CALL` row
    this crop was seeded from -- labeling can start from what Mathpix
    already guessed and just have a human confirm/correct it, rather than
    transcribe every crop from scratch. It's `None` for a crop added some
    other way.

    `verified` is True once a human has actually looked at `text` and
    confirmed it's right; a row inserted straight from an unreviewed
    Mathpix guess starts False and is excluded by
    `list_handwriting_labels(verified_only=True)` -- an unverified label is
    exactly the kind of noisy ground truth this whole feature exists to
    stop relying on."""
    image_s3url: str
    image_sha256: str
    text: str
    verified: bool = False
    source_mathpix_call_id: Optional[int] = None
    created_at: Optional[str] = None
    id: Optional[int] = None


# --------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------

def init_db(db_path: Path) -> Connection:
    conn = connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS WORKSHEET (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            tex_source TEXT,
            questions_json TEXT,
            model TEXT,
            num_questions INTEGER,
            title TEXT,
            header TEXT,
            public_id TEXT,
            boxes_json TEXT,
            student_pdf_s3url TEXT,
            cv_pdf_s3url TEXT,
            answers_pdf_s3url TEXT,
            sty_hash TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS STY_VERSION (
            hash TEXT PRIMARY KEY,
            content TEXT,
            created_at TEXT
        )
        """
    )
    # One row per Mathpix OCR call (issue #1), so we can compile a labelled
    # dataset for a future in-house OCR model. The image itself lives in S3
    # (image_s3url); image_sha256 is the content hash used both as the S3 key
    # and to dedupe/join identical crops.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS MATHPIX_CALL (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_s3url TEXT,
            image_sha256 TEXT,
            response_json TEXT,
            response_text TEXT,
            created_at TEXT
        )
        """
    )
    # A class of students whose name-learning worksheets are managed together
    # (issue #43).
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS CLASSROOM (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS STUDENT (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL REFERENCES CLASSROOM(id),
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            nickname TEXT,
            created_at TEXT,
            UNIQUE(classroom_id, first_name, last_name)
        )
        """
    )
    # One row per handwriting sample cropped from a scanned name-collection
    # sheet (issue #2/#43), used to train a per-student name classifier. The
    # crop itself lives in S3 (image_s3url); image_sha256 is the content hash
    # used both as the S3 key and to dedupe identical crops on re-ingest.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS NAME_IMAGES (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL REFERENCES STUDENT(id),
            box_id TEXT,
            image_s3url TEXT,
            image_sha256 TEXT,
            created_at TEXT
        )
        """
    )
    # A vector embedding of one NAME_IMAGES row (issue #46), stored as its own
    # S3 object; name_image_id is UNIQUE so "already embedded" is a join, not a
    # separate dedup index.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS NAME_EMBEDDINGS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL REFERENCES STUDENT(id),
            name_image_id INTEGER NOT NULL UNIQUE REFERENCES NAME_IMAGES(id),
            embedding_s3url TEXT,
            created_at TEXT
        )
        """
    )
    # A low/no-confidence name-box crop captured during grading (issue #92),
    # queued for a human to assign to the right student in the "Label names"
    # tab. Deleted once resolved (assigned or discarded) -- this table only
    # ever holds the current queue.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS PENDING_NAME_LABEL (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL REFERENCES CLASSROOM(id),
            box_id TEXT,
            image_s3url TEXT,
            image_sha256 TEXT,
            predicted_name TEXT,
            confidence REAL,
            source TEXT,
            created_at TEXT
        )
        """
    )
    # One row per human-confirmed (crop, text) pair for the response-scorer
    # model (issue #81) -- see HandwritingLabelRecord. source_mathpix_call_id
    # is nullable (not every label starts from a Mathpix guess); verified
    # defaults to 0 (unreviewed) so a bulk import never silently counts as
    # ground truth.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS HANDWRITING_LABEL (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_s3url TEXT,
            image_sha256 TEXT,
            text TEXT,
            verified INTEGER NOT NULL DEFAULT 0,
            source_mathpix_call_id INTEGER REFERENCES MATHPIX_CALL(id),
            created_at TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    if "git_sha" in columns:
        conn.execute("ALTER TABLE WORKSHEET DROP COLUMN git_sha")
    if "sty_hash" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN sty_hash TEXT")
    if "title" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN title TEXT")
    if "header" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN header TEXT")
    if "public_id" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN public_id TEXT")
    if "boxes_json" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN boxes_json TEXT")
    conn.commit()
    return conn


def insert_worksheet(conn: Connection, record: WorksheetRecord) -> int:
    cursor = conn.execute(
        """
        INSERT INTO WORKSHEET
            (prompt, tex_source, questions_json, model, num_questions, title,
             header, public_id, boxes_json, student_pdf_s3url, cv_pdf_s3url,
             answers_pdf_s3url, sty_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.prompt,
            record.tex_source,
            record.questions_json,
            record.model,
            record.num_questions,
            record.title,
            record.header,
            record.public_id,
            record.boxes_json,
            record.student_pdf_s3url,
            record.cv_pdf_s3url,
            record.answers_pdf_s3url,
            record.sty_hash,
            record.created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_mathpix_call(
    conn: Connection,
    image_s3url: str,
    image_sha256: str,
    response_json: str,
    response_text: str,
    created_at: str,
) -> int:
    """Records a single Mathpix OCR call in the MATHPIX_CALL table (issue #1)
    and returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO MATHPIX_CALL
            (image_s3url, image_sha256, response_json, response_text, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (image_s3url, image_sha256, response_json, response_text, created_at),
    )
    conn.commit()
    return cursor.lastrowid


def get_or_create_classroom(conn: Connection, label: str) -> ClassroomRecord:
    """Looks up a CLASSROOM by its (unique) label, creating it if it doesn't
    exist yet -- this is what gives the Roster tab's "new class or append to
    an existing class" (issue #43) behaviour for free."""
    conn.execute(
        "INSERT OR IGNORE INTO CLASSROOM (label, created_at) VALUES (?, ?)",
        (label, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id, label FROM CLASSROOM WHERE label = ?", (label,)
    ).fetchone()
    return ClassroomRecord(id=row[0], label=row[1])


def list_classrooms(conn: Connection) -> List[ClassroomRecord]:
    rows = conn.execute("SELECT id, label FROM CLASSROOM ORDER BY label").fetchall()
    return [ClassroomRecord(id=row[0], label=row[1]) for row in rows]


def get_or_create_student(
    conn: Connection,
    classroom_id: int,
    first_name: str,
    last_name: str,
    nickname: Optional[str] = None,
) -> StudentRecord:
    """Looks up a STUDENT by (classroom_id, first_name, last_name), creating
    it if it doesn't exist yet."""
    conn.execute(
        """
        INSERT OR IGNORE INTO STUDENT
            (classroom_id, first_name, last_name, nickname, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (classroom_id, first_name, last_name, nickname, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT id, classroom_id, first_name, last_name, nickname
        FROM STUDENT WHERE classroom_id = ? AND first_name = ? AND last_name = ?
        """,
        (classroom_id, first_name, last_name),
    ).fetchone()
    return StudentRecord(
        id=row[0], classroom_id=row[1], first_name=row[2], last_name=row[3], nickname=row[4]
    )


_CSV_REQUIRED_COLUMNS = ("first_name", "last_name")


def import_students_csv(conn: Connection, classroom_id: int, csv_text: str) -> CsvImportResult:
    """Bulk-adds students to a classroom from a roster CSV.

    Expected format: a header row containing `first_name` and `last_name`
    columns, plus an optional `nickname` column. Column order doesn't
    matter, matching is case-insensitive, and extra columns are ignored.
    One student per remaining row, e.g.::

        first_name,last_name,nickname
        Anna,Smith,
        Zeke,Jones,Z

    Rows missing a first or last name are skipped (reported in
    `CsvImportResult.skipped`), not fatal. Adding a name that already
    exists in the classroom is a no-op (`get_or_create_student` is
    idempotent), so re-importing the same CSV is safe.
    """
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV is empty.")

    normalized = [h.strip().lower() for h in header]
    missing = [col for col in _CSV_REQUIRED_COLUMNS if col not in normalized]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            "Expected a header row with 'first_name', 'last_name', "
            "and optionally 'nickname'."
        )

    first_idx = normalized.index("first_name")
    last_idx = normalized.index("last_name")
    nick_idx = normalized.index("nickname") if "nickname" in normalized else None

    def cell(row: List[str], idx: Optional[int]) -> str:
        if idx is None or idx >= len(row):
            return ""
        return row[idx].strip()

    added: List[StudentRecord] = []
    skipped: List[str] = []
    for row_num, row in enumerate(reader, start=2):  # header occupies row 1
        if not row or all(not field.strip() for field in row):
            continue  # blank line

        first_name = cell(row, first_idx)
        last_name = cell(row, last_idx)
        nickname = cell(row, nick_idx) if nick_idx is not None else ""

        if not first_name or not last_name:
            skipped.append(f"row {row_num}: missing first or last name")
            continue

        student = get_or_create_student(
            conn, classroom_id, first_name, last_name, nickname or None
        )
        added.append(student)

    return CsvImportResult(added=added, skipped=skipped)


def list_students(conn: Connection, classroom_id: int) -> List[StudentRecord]:
    rows = conn.execute(
        """
        SELECT id, classroom_id, first_name, last_name, nickname
        FROM STUDENT WHERE classroom_id = ? ORDER BY first_name, last_name
        """,
        (classroom_id,),
    ).fetchall()
    return [
        StudentRecord(id=row[0], classroom_id=row[1], first_name=row[2], last_name=row[3], nickname=row[4])
        for row in rows
    ]


def transfer_student(conn: Connection, student_id: int, new_classroom_id: int) -> StudentRecord:
    """Moves a student to a different classroom (issue #72).

    Unlike delete-and-re-add, this preserves the student's `NAME_IMAGES`/
    `NAME_EMBEDDINGS` handwriting samples -- those tables are keyed by
    `student_id`, not `classroom_id`, so there is nothing to re-collect.
    (The per-classroom trained classifier is a separate saved artifact and
    still needs retraining for both classrooms afterwards, same as any other
    roster change -- see the README.)

    Raises `ValueError` if the student or the target classroom doesn't
    exist, or if the target classroom already has a student with the same
    (first_name, last_name) -- transferring into that name collision is
    left for the caller to resolve by hand rather than silently merging.
    Transferring a student to the classroom they're already in is a no-op.
    """
    row = conn.execute(
        "SELECT id, classroom_id, first_name, last_name, nickname FROM STUDENT WHERE id = ?",
        (student_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No student with id {student_id}.")
    student = StudentRecord(
        id=row[0], classroom_id=row[1], first_name=row[2], last_name=row[3], nickname=row[4]
    )

    if student.classroom_id == new_classroom_id:
        return student

    classroom_row = conn.execute(
        "SELECT id FROM CLASSROOM WHERE id = ?", (new_classroom_id,)
    ).fetchone()
    if classroom_row is None:
        raise ValueError(f"No classroom with id {new_classroom_id}.")

    collision = conn.execute(
        """
        SELECT id FROM STUDENT
        WHERE classroom_id = ? AND first_name = ? AND last_name = ?
        """,
        (new_classroom_id, student.first_name, student.last_name),
    ).fetchone()
    if collision is not None:
        raise ValueError(
            f"{student.first_name} {student.last_name} already exists in the "
            "target classroom."
        )

    conn.execute(
        "UPDATE STUDENT SET classroom_id = ? WHERE id = ?",
        (new_classroom_id, student_id),
    )
    conn.commit()
    return StudentRecord(
        id=student.id,
        classroom_id=new_classroom_id,
        first_name=student.first_name,
        last_name=student.last_name,
        nickname=student.nickname,
    )


def insert_name_image(conn: Connection, record: NameImageRecord) -> int:
    """Records a single handwriting sample in the NAME_IMAGES table
    (issue #2/#43) and returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO NAME_IMAGES
            (student_id, box_id, image_s3url, image_sha256, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            record.student_id,
            record.box_id,
            record.image_s3url,
            record.image_sha256,
            record.created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def name_image_exists(conn: Connection, image_sha256: str) -> bool:
    """Returns True if a NAME_IMAGES row already stores the crop with this
    content hash, so ingest can skip re-uploading duplicate samples."""
    row = conn.execute(
        "SELECT 1 FROM NAME_IMAGES WHERE image_sha256 = ? LIMIT 1",
        (image_sha256,),
    ).fetchone()
    return row is not None


_NAME_IMAGE_COLUMNS = (
    "id", "student_id", "box_id", "image_s3url", "image_sha256", "created_at",
)


def _row_to_name_image(row) -> NameImageRecord:
    return NameImageRecord(
        id=row[0],
        student_id=row[1],
        box_id=row[2],
        image_s3url=row[3],
        image_sha256=row[4],
        created_at=row[5],
    )


def list_name_images(conn: Connection, student_id: Optional[int] = None) -> List[NameImageRecord]:
    if student_id is None:
        rows = conn.execute(
            f"SELECT {', '.join(_NAME_IMAGE_COLUMNS)} FROM NAME_IMAGES ORDER BY created_at"
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {', '.join(_NAME_IMAGE_COLUMNS)} FROM NAME_IMAGES "
            "WHERE student_id = ? ORDER BY created_at",
            (student_id,),
        ).fetchall()
    return [_row_to_name_image(row) for row in rows]


def list_unembedded_name_images(conn: Connection) -> List[NameImageRecord]:
    """NAME_IMAGES rows with no matching NAME_EMBEDDINGS row yet, so
    `vectorize_samples` can embed only what's new (idempotent re-runs)."""
    rows = conn.execute(
        f"""
        SELECT {', '.join(f'ni.{c}' for c in _NAME_IMAGE_COLUMNS)}
        FROM NAME_IMAGES ni
        LEFT JOIN NAME_EMBEDDINGS ne ON ne.name_image_id = ni.id
        WHERE ne.id IS NULL
        ORDER BY ni.created_at
        """
    ).fetchall()
    return [_row_to_name_image(row) for row in rows]


def insert_name_embedding(conn: Connection, record: NameEmbeddingRecord) -> int:
    """Records a single embedding in the NAME_EMBEDDINGS table (issue #43/#46)
    and returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO NAME_EMBEDDINGS
            (student_id, name_image_id, embedding_s3url, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (record.student_id, record.name_image_id, record.embedding_s3url, record.created_at),
    )
    conn.commit()
    return cursor.lastrowid


def insert_pending_name_label(conn: Connection, record: PendingNameLabelRecord) -> int:
    """Queues one low/no-confidence name-box crop from grading (issue #92)
    and returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO PENDING_NAME_LABEL
            (classroom_id, box_id, image_s3url, image_sha256, predicted_name,
             confidence, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.classroom_id,
            record.box_id,
            record.image_s3url,
            record.image_sha256,
            record.predicted_name,
            record.confidence,
            record.source,
            record.created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def pending_name_label_exists(conn: Connection, image_sha256: str) -> bool:
    """True if this crop is already queued (same content hash) -- lets
    capture skip re-queuing a crop across repeated grading runs of the same
    scan, mirroring `name_image_exists`."""
    row = conn.execute(
        "SELECT 1 FROM PENDING_NAME_LABEL WHERE image_sha256 = ? LIMIT 1",
        (image_sha256,),
    ).fetchone()
    return row is not None


_PENDING_NAME_LABEL_COLUMNS = (
    "id", "classroom_id", "box_id", "image_s3url", "image_sha256",
    "predicted_name", "confidence", "source", "created_at",
)


def _row_to_pending_name_label(row) -> PendingNameLabelRecord:
    return PendingNameLabelRecord(
        id=row[0],
        classroom_id=row[1],
        box_id=row[2],
        image_s3url=row[3],
        image_sha256=row[4],
        predicted_name=row[5],
        confidence=row[6],
        source=row[7],
        created_at=row[8],
    )


def count_pending_name_labels(conn: Connection, classroom_id: int) -> int:
    """How many crops are still queued for this classroom -- shown in the
    "Label names" tab so a teacher can see whether the queue is worth
    working through."""
    row = conn.execute(
        "SELECT COUNT(*) FROM PENDING_NAME_LABEL WHERE classroom_id = ?",
        (classroom_id,),
    ).fetchone()
    return row[0]


def random_pending_name_label(
    conn: Connection, classroom_id: int
) -> Optional[PendingNameLabelRecord]:
    """One pending crop for this classroom, chosen uniformly at random
    (issue #92: "for the moment, you can just request the label tasks
    uniformly at random"), or `None` if the queue is empty."""
    row = conn.execute(
        f"""
        SELECT {', '.join(_PENDING_NAME_LABEL_COLUMNS)} FROM PENDING_NAME_LABEL
        WHERE classroom_id = ? ORDER BY RANDOM() LIMIT 1
        """,
        (classroom_id,),
    ).fetchone()
    return _row_to_pending_name_label(row) if row is not None else None


def delete_pending_name_label(conn: Connection, pending_id: int) -> None:
    """Removes a resolved (assigned or discarded) crop from the queue."""
    conn.execute("DELETE FROM PENDING_NAME_LABEL WHERE id = ?", (pending_id,))
    conn.commit()


def insert_handwriting_label(conn: Connection, record: HandwritingLabelRecord) -> int:
    """Records a single labeled crop in the HANDWRITING_LABEL table
    (issue #81) and returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO HANDWRITING_LABEL
            (image_s3url, image_sha256, text, verified, source_mathpix_call_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.image_s3url,
            record.image_sha256,
            record.text,
            int(record.verified),
            record.source_mathpix_call_id,
            record.created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def handwriting_label_exists(conn: Connection, image_sha256: str) -> bool:
    """True if a HANDWRITING_LABEL row already stores the crop with this
    content hash, so a labeling pass can skip re-uploading duplicate crops
    (same convention as `name_image_exists`)."""
    row = conn.execute(
        "SELECT 1 FROM HANDWRITING_LABEL WHERE image_sha256 = ? LIMIT 1",
        (image_sha256,),
    ).fetchone()
    return row is not None


_HANDWRITING_LABEL_COLUMNS = (
    "id", "image_s3url", "image_sha256", "text", "verified", "source_mathpix_call_id", "created_at",
)


def _row_to_handwriting_label(row) -> HandwritingLabelRecord:
    return HandwritingLabelRecord(
        id=row[0],
        image_s3url=row[1],
        image_sha256=row[2],
        text=row[3],
        verified=bool(row[4]),
        source_mathpix_call_id=row[5],
        created_at=row[6],
    )


def list_handwriting_labels(conn: Connection, verified_only: bool = False) -> List[HandwritingLabelRecord]:
    """All labeled crops, or only the human-confirmed ones when
    `verified_only` is True -- `training/eval.py`'s real-data comparison
    (issue #81) should always pass `verified_only=True`; an unreviewed
    label is exactly the kind of noisy ground truth this feature exists to
    stop relying on."""
    query = f"SELECT {', '.join(_HANDWRITING_LABEL_COLUMNS)} FROM HANDWRITING_LABEL"
    if verified_only:
        query += " WHERE verified = 1"
    query += " ORDER BY created_at"
    rows = conn.execute(query).fetchall()
    return [_row_to_handwriting_label(row) for row in rows]


def unlabeled_mathpix_calls(conn: Connection, limit: int = 100) -> List[Tuple[int, str, str, str]]:
    """MATHPIX_CALL rows with no matching HANDWRITING_LABEL row yet, most
    recent first: `(id, image_s3url, image_sha256, response_text)` --
    exactly what `scripts/label_handwriting.py` needs to show a human the
    image and Mathpix's own guess as a starting point to confirm/correct."""
    rows = conn.execute(
        """
        SELECT mc.id, mc.image_s3url, mc.image_sha256, mc.response_text
        FROM MATHPIX_CALL mc
        LEFT JOIN HANDWRITING_LABEL hl ON hl.source_mathpix_call_id = mc.id
        WHERE hl.id IS NULL
        ORDER BY mc.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return list(rows)


def delete_student(conn: Connection, student_id: int, s3_client=None) -> None:
    """Deletes a student's S3 blobs (every NAME_IMAGES/NAME_EMBEDDINGS object)
    and DB rows. S3 deletes happen first and strictly: if any raises, the
    error propagates and the DB rows are left intact, mirroring
    `delete_worksheet`."""
    image_urls = [
        row[0]
        for row in conn.execute(
            "SELECT image_s3url FROM NAME_IMAGES WHERE student_id = ? AND image_s3url IS NOT NULL",
            (student_id,),
        ).fetchall()
    ]
    embedding_urls = [
        row[0]
        for row in conn.execute(
            "SELECT embedding_s3url FROM NAME_EMBEDDINGS WHERE student_id = ? AND embedding_s3url IS NOT NULL",
            (student_id,),
        ).fetchall()
    ]
    for url in image_urls + embedding_urls:
        img_bucket, key = parse_s3_url(url)
        delete_from_s3(img_bucket, key, s3_client=s3_client)

    conn.execute("DELETE FROM NAME_EMBEDDINGS WHERE student_id = ?", (student_id,))
    conn.execute("DELETE FROM NAME_IMAGES WHERE student_id = ?", (student_id,))
    conn.execute("DELETE FROM STUDENT WHERE id = ?", (student_id,))
    conn.commit()


# Column order shared by every WorksheetRecord SELECT, kept in sync with
# `_row_to_record` below.
_RECORD_COLUMNS = (
    "id", "prompt", "tex_source", "questions_json", "model", "num_questions",
    "title", "header", "public_id", "boxes_json", "student_pdf_s3url",
    "cv_pdf_s3url", "answers_pdf_s3url", "sty_hash", "created_at",
)


def _row_to_record(row) -> WorksheetRecord:
    return WorksheetRecord(
        id=row[0],
        prompt=row[1],
        tex_source=row[2],
        questions_json=row[3],
        model=row[4],
        num_questions=row[5],
        title=row[6],
        header=row[7],
        public_id=row[8],
        boxes_json=row[9],
        student_pdf_s3url=row[10],
        cv_pdf_s3url=row[11],
        answers_pdf_s3url=row[12],
        sty_hash=row[13],
        created_at=row[14],
    )


def list_worksheets(conn: Connection) -> List[WorksheetRecord]:
    rows = conn.execute(
        f"SELECT {', '.join(_RECORD_COLUMNS)} FROM WORKSHEET ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_record(row) for row in rows]


def get_worksheet_by_public_id(conn: Connection, public_id: str) -> Optional[WorksheetRecord]:
    """Looks up a single worksheet by its embedded public id (the value
    encoded in the on-page QR code). Returns None if no row matches."""
    row = conn.execute(
        f"SELECT {', '.join(_RECORD_COLUMNS)} FROM WORKSHEET WHERE public_id = ?",
        (public_id,),
    ).fetchone()
    return _row_to_record(row) if row is not None else None


def serialize_boxes(boxes: Dict[str, Box]) -> str:
    """Serializes a {box id: Box} mapping (as produced by
    `graderbot.extract_answer_boxes`) to JSON for the WORKSHEET.boxes_json
    column, so the grader can recover answer-box locations from a scan
    without re-rendering the worksheet."""
    return json.dumps({box_id: asdict(box) for box_id, box in boxes.items()})


def deserialize_boxes(boxes_json: str) -> Dict[str, Box]:
    """Inverse of `serialize_boxes`: rebuilds the {box id: Box} mapping from
    stored JSON."""
    return {box_id: Box(**fields) for box_id, fields in json.loads(boxes_json).items()}


# --------------------------------------------------------------------------
# .sty versioning
# --------------------------------------------------------------------------

def compute_sty_hash(sty_path: Path = WORKSHEET_STY_PATH) -> str:
    """Returns the sha256 hex digest of the .sty file's contents, used to
    tie a compiled worksheet to the exact layout it was built with."""
    return hashlib.sha256(Path(sty_path).read_bytes()).hexdigest()


def record_sty_version(conn: Connection, sty_path: Path = WORKSHEET_STY_PATH) -> str:
    """Ensures a STY_VERSION row exists for the current contents of
    `sty_path`, inserting one only the first time this hash is seen, and
    returns its hash."""
    sty_hash = compute_sty_hash(sty_path)
    conn.execute(
        "INSERT OR IGNORE INTO STY_VERSION (hash, content, created_at) VALUES (?, ?, ?)",
        (sty_hash, Path(sty_path).read_text(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return sty_hash


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def slugify_title(title: str) -> str:
    """Converts a worksheet title into a filesystem/URL-safe slug for use
    as a PDF filename prefix (e.g. "Linear Equations!" -> "Linear_Equations")."""
    slug = _SLUG_RE.sub("_", title).strip("_")
    return slug or "worksheet"


def _default_s3_client():
    """boto3 only reads AWS_DEFAULT_REGION automatically, not AWS_REGION
    (which is what this repo's .env and README use) - so an unset region
    silently falls back to us-east-1. If the bucket lives elsewhere,
    presigned URLs get signed for the wrong region's endpoint and S3
    rejects them. Pass AWS_REGION through explicitly to avoid that."""
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION"))


def upload_to_s3(local_path: Path, bucket: str, key: str, s3_client=None) -> str:
    client = s3_client if s3_client is not None else _default_s3_client()
    client.upload_file(str(local_path), bucket, key)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def parse_s3_url(url: str) -> Tuple[str, str]:
    """Splits a `https://{bucket}.s3.amazonaws.com/{key}` URL, as produced
    by `upload_to_s3`, back into `(bucket, key)`."""
    parsed = urlparse(url)
    bucket = parsed.netloc.split(".s3.amazonaws.com")[0]
    key = parsed.path.lstrip("/")
    return bucket, key


def generate_presigned_url(bucket: str, key: str, s3_client=None, expires_in: int = 3600) -> str:
    client = s3_client if s3_client is not None else _default_s3_client()
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
    )


def delete_from_s3(bucket: str, key: str, s3_client=None) -> None:
    client = s3_client if s3_client is not None else _default_s3_client()
    client.delete_object(Bucket=bucket, Key=key)


# --------------------------------------------------------------------------
# PDF generation
# --------------------------------------------------------------------------

def images_to_pdf(
    images: Iterable[np.ndarray], out_path: Path, jpeg_quality: int = 85
) -> Path:
    """Writes one or more BGR images as a single (multi-page) PDF, one page per
    image, in order.

    `images` may be any iterable, including a generator -- each image is
    JPEG-encoded and inserted into the output document as it's produced, so
    at most one raw (uncompressed) page is ever resident at a time.

    The encoding has to be JPEG specifically: PyMuPDF's `insert_image` can
    only embed a JPEG stream as-is (as a PDF `DCTDecode` image). Anything
    else -- including PNG -- it fully decodes back to a raw bitmap first and
    embeds *that*, uncompressed, regardless of how small the source encoding
    was. A 1275x1650 canonical page, PNG-encoded to ~50KB, still landed in
    the PDF at ~6.3MB (its exact raw RGB size) -- so a big multi-page batch
    blew memory even after switching from Pillow's `append_images` (which
    needed every page's full-resolution array resident at once) to this
    one-page-at-a-time approach, just at a bigger multiple. JPEG at
    `jpeg_quality` instead lands in the PDF at roughly its encoded size
    (tens of KB for a typical worksheet page), which is what actually keeps
    a many-page batch off the 1GB fly.io VM's memory ceiling."""
    doc = fitz.open()
    try:
        wrote_a_page = False
        for img in images:
            height, width = img.shape[:2]
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise ValueError("failed to JPEG-encode an image for images_to_pdf")
            page = doc.new_page(width=width, height=height)
            page.insert_image(page.rect, stream=buf.tobytes())
            wrote_a_page = True
        if not wrote_a_page:
            raise ValueError("images_to_pdf requires at least one image")
        doc.save(out_path)
    finally:
        doc.close()
    return out_path


def image_to_pdf(image: np.ndarray, out_path: Path) -> Path:
    return images_to_pdf([image], out_path)


def generate_answer_key_pdf(tex_fn: str, answers: Dict[str, str], out_path: Path) -> Path:
    images = fill_worksheet(tex_fn, answers)
    return images_to_pdf(images, out_path)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def store_worksheet(
    tex_path: Path,
    questions: "list[Question]",
    prompt: str,
    model: str,
    bucket: str,
    db_path: Path,
    title: Optional[str] = None,
    header: Optional[str] = None,
    public_id: Optional[str] = None,
    s3_client=None,
) -> WorksheetRecord:
    tex_path = Path(tex_path)
    stem = tex_path.stem
    filename_prefix = slugify_title(title) if title else stem

    # Namespace S3 keys by the unique public_id so two worksheets that share a
    # title (or the default tex stem) can't collide and overwrite each other's
    # PDFs (issue #33). Fall back to a fresh random id if ever called without a
    # public_id. The title slug stays in the filename for readable downloads.
    key_prefix = public_id or uuid4().hex[:8]

    student_pdf = latexmk_worksheet(str(tex_path), cv_mode=False)
    cv_pdf = latexmk_worksheet(str(tex_path), cv_mode=True)

    # Persist answer-box locations so the grader can map a scan's boxes from
    # the DB alone, without re-rendering the cv worksheet at grade time.
    boxes_json = serialize_boxes(extract_answer_boxes(cv_pdf))

    # Open-ended questions (issue #65) have nothing to reveal, so they're left
    # off the answer key -- fill_worksheet skips any box id absent from
    # `answers`, so those boxes simply stay blank on the rendered PDF.
    answers = {q.id: q.answer for q in questions if not q.open_ended}
    answers_pdf_path = tex_path.with_name(f"{stem}_answers.pdf")
    answers_pdf = generate_answer_key_pdf(str(tex_path), answers, answers_pdf_path)

    student_url = upload_to_s3(
        Path(student_pdf), bucket, f"{key_prefix}/{filename_prefix}_student.pdf", s3_client=s3_client
    )
    cv_url = upload_to_s3(
        Path(cv_pdf), bucket, f"{key_prefix}/{filename_prefix}_cv.pdf", s3_client=s3_client
    )
    answers_url = upload_to_s3(
        Path(answers_pdf), bucket, f"{key_prefix}/{filename_prefix}_answers.pdf", s3_client=s3_client
    )

    conn = init_db(db_path)
    sty_hash = record_sty_version(conn)

    record = WorksheetRecord(
        prompt=prompt,
        tex_source=tex_path.read_text(),
        questions_json=json.dumps([asdict(q) for q in questions]),
        model=model,
        num_questions=len(questions),
        title=title,
        header=header,
        public_id=public_id,
        boxes_json=boxes_json,
        student_pdf_s3url=student_url,
        cv_pdf_s3url=cv_url,
        answers_pdf_s3url=answers_url,
        sty_hash=sty_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    record.id = insert_worksheet(conn, record)
    conn.close()

    return record


def delete_worksheet(conn: Connection, record: WorksheetRecord, s3_client=None) -> None:
    """Deletes a worksheet's S3 blobs (student/cv/answer-key PDFs) and its
    WORKSHEET row. S3 deletes happen first and strictly: if any raises, the
    error propagates and the SQLite row is left intact so nothing is left in a
    half-deleted state."""
    for url in (record.student_pdf_s3url, record.cv_pdf_s3url, record.answers_pdf_s3url):
        if url:
            bucket, key = parse_s3_url(url)
            delete_from_s3(bucket, key, s3_client=s3_client)

    conn.execute("DELETE FROM WORKSHEET WHERE id = ?", (record.id,))
    conn.commit()
