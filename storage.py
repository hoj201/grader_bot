"""Uploads generated worksheet PDFs to S3 and records them in SQLite.

See the `WORKSHEET` table created by `init_db` for the persisted schema.
The SQLite file is expected to be replicated to S3 by litestream
(`litestream.yml`), which is why `init_db` enables WAL journal mode.
"""

import json
import os
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

from worksheet_synth import fill_worksheet, latexmk_worksheet

if TYPE_CHECKING:
    from worksheetbot import Question


@dataclass
class WorksheetRecord:
    prompt: str
    tex_source: str
    questions_json: str
    model: str
    num_questions: int
    student_pdf_s3url: Optional[str] = None
    cv_pdf_s3url: Optional[str] = None
    answers_pdf_s3url: Optional[str] = None
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
            student_pdf_s3url TEXT,
            cv_pdf_s3url TEXT,
            answers_pdf_s3url TEXT,
            created_at TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    if "git_sha" in columns:
        conn.execute("ALTER TABLE WORKSHEET DROP COLUMN git_sha")
    conn.commit()
    return conn


def insert_worksheet(conn: Connection, record: WorksheetRecord) -> int:
    cursor = conn.execute(
        """
        INSERT INTO WORKSHEET
            (prompt, tex_source, questions_json, model, num_questions,
             student_pdf_s3url, cv_pdf_s3url, answers_pdf_s3url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.prompt,
            record.tex_source,
            record.questions_json,
            record.model,
            record.num_questions,
            record.student_pdf_s3url,
            record.cv_pdf_s3url,
            record.answers_pdf_s3url,
            record.created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def list_worksheets(conn: Connection) -> List[WorksheetRecord]:
    rows = conn.execute(
        """
        SELECT id, prompt, tex_source, questions_json, model, num_questions,
               student_pdf_s3url, cv_pdf_s3url, answers_pdf_s3url, created_at
        FROM WORKSHEET
        ORDER BY created_at DESC
        """
    ).fetchall()
    return [
        WorksheetRecord(
            id=row[0],
            prompt=row[1],
            tex_source=row[2],
            questions_json=row[3],
            model=row[4],
            num_questions=row[5],
            student_pdf_s3url=row[6],
            cv_pdf_s3url=row[7],
            answers_pdf_s3url=row[8],
            created_at=row[9],
        )
        for row in rows
    ]


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------

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
    s3_client=None,
) -> WorksheetRecord:
    tex_path = Path(tex_path)
    stem = tex_path.stem

    student_pdf = latexmk_worksheet(str(tex_path), cv_mode=False)
    cv_pdf = latexmk_worksheet(str(tex_path), cv_mode=True)

    answers = {q.id: q.answer for q in questions}
    answers_pdf_path = tex_path.with_name(f"{stem}_answers.pdf")
    answers_pdf = generate_answer_key_pdf(str(tex_path), answers, answers_pdf_path)

    student_url = upload_to_s3(Path(student_pdf), bucket, f"{stem}/student.pdf", s3_client=s3_client)
    cv_url = upload_to_s3(Path(cv_pdf), bucket, f"{stem}/cv.pdf", s3_client=s3_client)
    answers_url = upload_to_s3(Path(answers_pdf), bucket, f"{stem}/answers.pdf", s3_client=s3_client)

    record = WorksheetRecord(
        prompt=prompt,
        tex_source=tex_path.read_text(),
        questions_json=json.dumps([asdict(q) for q in questions]),
        model=model,
        num_questions=len(questions),
        student_pdf_s3url=student_url,
        cv_pdf_s3url=cv_url,
        answers_pdf_s3url=answers_url,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    conn = init_db(db_path)
    record.id = insert_worksheet(conn, record)
    conn.close()

    return record
