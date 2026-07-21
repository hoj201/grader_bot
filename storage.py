"""Uploads generated worksheet PDFs to S3 and records them in SQLite.

See the `WORKSHEET` table created by `init_db` for the persisted schema.
The SQLite file is expected to be replicated to S3 by litestream
(`litestream.yml`), which is why `init_db` enables WAL journal mode.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection, connect
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
from PIL import Image

from graderbot import Box, extract_answer_boxes
from worksheet_synth import WORKSHEET_STY_PATH, fill_worksheet, latexmk_worksheet

if TYPE_CHECKING:
    from worksheetbot import Question


@dataclass
class WorksheetRecord:
    prompt: str
    tex_source: str
    questions_json: str
    model: str
    num_questions: int
    title: Optional[str] = None
    public_id: Optional[str] = None
    boxes_json: Optional[str] = None
    student_pdf_s3url: Optional[str] = None
    cv_pdf_s3url: Optional[str] = None
    answers_pdf_s3url: Optional[str] = None
    sty_hash: Optional[str] = None
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
    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    if "git_sha" in columns:
        conn.execute("ALTER TABLE WORKSHEET DROP COLUMN git_sha")
    if "sty_hash" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN sty_hash TEXT")
    if "title" not in columns:
        conn.execute("ALTER TABLE WORKSHEET ADD COLUMN title TEXT")
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
             public_id, boxes_json, student_pdf_s3url, cv_pdf_s3url,
             answers_pdf_s3url, sty_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.prompt,
            record.tex_source,
            record.questions_json,
            record.model,
            record.num_questions,
            record.title,
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


# Column order shared by every WorksheetRecord SELECT, kept in sync with
# `_row_to_record` below.
_RECORD_COLUMNS = (
    "id", "prompt", "tex_source", "questions_json", "model", "num_questions",
    "title", "public_id", "boxes_json", "student_pdf_s3url", "cv_pdf_s3url",
    "answers_pdf_s3url", "sty_hash", "created_at",
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
        public_id=row[7],
        boxes_json=row[8],
        student_pdf_s3url=row[9],
        cv_pdf_s3url=row[10],
        answers_pdf_s3url=row[11],
        sty_hash=row[12],
        created_at=row[13],
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


# --------------------------------------------------------------------------
# PDF generation
# --------------------------------------------------------------------------

def image_to_pdf(image: np.ndarray, out_path: Path) -> Path:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_image).save(out_path, "PDF")
    return out_path


def generate_answer_key_pdf(tex_fn: str, answers: Dict[str, str], out_path: Path) -> Path:
    image = fill_worksheet(tex_fn, answers)
    return image_to_pdf(image, out_path)


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
    public_id: Optional[str] = None,
    s3_client=None,
) -> WorksheetRecord:
    tex_path = Path(tex_path)
    stem = tex_path.stem
    filename_prefix = slugify_title(title) if title else stem

    student_pdf = latexmk_worksheet(str(tex_path), cv_mode=False)
    cv_pdf = latexmk_worksheet(str(tex_path), cv_mode=True)

    # Persist answer-box locations so the grader can map a scan's boxes from
    # the DB alone, without re-rendering the cv worksheet at grade time.
    boxes_json = serialize_boxes(extract_answer_boxes(cv_pdf))

    answers = {q.id: q.answer for q in questions}
    answers_pdf_path = tex_path.with_name(f"{stem}_answers.pdf")
    answers_pdf = generate_answer_key_pdf(str(tex_path), answers, answers_pdf_path)

    student_url = upload_to_s3(
        Path(student_pdf), bucket, f"{stem}/{filename_prefix}_student.pdf", s3_client=s3_client
    )
    cv_url = upload_to_s3(
        Path(cv_pdf), bucket, f"{stem}/{filename_prefix}_cv.pdf", s3_client=s3_client
    )
    answers_url = upload_to_s3(
        Path(answers_pdf), bucket, f"{stem}/{filename_prefix}_answers.pdf", s3_client=s3_client
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
