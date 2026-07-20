"""Streamlit frontend for browsing and creating GraderBot worksheets.

Run with:
  streamlit run app.py

Requires ANTHROPIC_API_KEY, S3_BUCKET, and AWS credentials in the
environment (see README.md).
"""

import os
from pathlib import Path
from uuid import uuid4

import anthropic
import streamlit as st
from dotenv import load_dotenv

import storage
from worksheetbot import CompileError, generate_worksheet

load_dotenv()

TEMPLATE_PATH = Path(os.environ.get("WORKSHEET_TEMPLATE", "worksheet_template.tex"))
DB_PATH = Path(os.environ.get("WORKSHEETS_DB_PATH", "worksheets.sqlite3"))
BUCKET = os.environ.get("S3_BUCKET")


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def render_gallery() -> None:
    conn = storage.init_db(DB_PATH)
    records = storage.list_worksheets(conn)
    conn.close()

    if not records:
        st.info("No worksheets yet. Create one in the Create tab.")
        return

    for record in records:
        with st.container(border=True):
            st.markdown(f"**{record.prompt}**")
            sty_version = record.sty_hash[:8] if record.sty_hash else "unknown"
            st.caption(
                f"id={record.id} · {record.num_questions} questions · "
                f"{record.model} · sty={sty_version} · {record.created_at}"
            )
            cols = st.columns(3)
            for col, label, url in zip(
                cols,
                ("Student", "CV", "Answer key"),
                (record.student_pdf_s3url, record.cv_pdf_s3url, record.answers_pdf_s3url),
            ):
                with col:
                    if url:
                        bucket, key = storage.parse_s3_url(url)
                        presigned = storage.generate_presigned_url(bucket, key)
                        st.link_button(label, presigned, use_container_width=True)
                    else:
                        st.button(label, disabled=True, use_container_width=True)


def render_create() -> None:
    prompt = st.text_area("Worksheet prompt", placeholder="10 question algebra worksheet on solving linear equations, grade 9")
    num_questions = st.number_input("Number of questions", min_value=1, max_value=50, value=10)
    submitted = st.button("Generate worksheet", type="primary", disabled=not prompt.strip())

    if not submitted:
        return

    out = Path("generated") / uuid4().hex[:8] / "worksheet"
    client = get_client()

    with st.spinner("Generating questions, compiling LaTeX, and uploading..."):
        try:
            _, _, record = generate_worksheet(
                client,
                TEMPLATE_PATH,
                prompt,
                out,
                num_questions=int(num_questions),
                max_repairs=3,
                bucket=BUCKET,
                db_path=DB_PATH,
            )
        except CompileError as e:
            st.error(f"LaTeX compilation failed after repair attempts:\n\n{e.log_tail}")
            return

    st.success(f"Created worksheet id={record.id}")
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="GraderBot", layout="wide")
    st.title("GraderBot")

    if not BUCKET:
        st.error("S3_BUCKET is not set. Configure it in .env before using this app.")
        st.stop()

    gallery_tab, create_tab = st.tabs(["Gallery", "Create"])
    with gallery_tab:
        render_gallery()
    with create_tab:
        render_create()


main()
