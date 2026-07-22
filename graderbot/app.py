"""Streamlit frontend for browsing and creating GraderBot worksheets.

Run with:
  streamlit run graderbot/app.py

Requires ANTHROPIC_API_KEY, S3_BUCKET, and AWS credentials in the
environment (see README.md).
"""

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import anthropic
import streamlit as st
from dotenv import load_dotenv

from graderbot import storage
from graderbot.scan_grader import mark_scan, results_by_student
from graderbot.worksheetbot import (
    AVAILABLE_MODELS,
    CompileError,
    create_worksheet_from_questions,
    generate_worksheet,
)

load_dotenv()

_TEX_DIR = Path(__file__).resolve().parent.parent / "tex"
TEMPLATE_PATH = Path(
    os.environ.get("WORKSHEET_TEMPLATE", str(_TEX_DIR / "worksheet_template.tex"))
)
DB_PATH = Path(os.environ.get("WORKSHEETS_DB_PATH", "worksheets.sqlite3"))
BUCKET = os.environ.get("S3_BUCKET")


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _delete_worksheet(record) -> None:
    conn = storage.init_db(DB_PATH)
    try:
        storage.delete_worksheet(conn, record)
    finally:
        conn.close()


def render_gallery() -> None:
    conn = storage.init_db(DB_PATH)
    records = storage.list_worksheets(conn)
    conn.close()

    if not records:
        st.info("No worksheets yet. Create one in the Create tab.")
        return

    for record in records:
        with st.container(border=True):
            st.markdown(f"**{record.title or record.prompt}**")
            if record.title:
                st.caption(record.prompt)
            sty_version = record.sty_hash[:8] if record.sty_hash else "unknown"
            st.caption(
                f"id={record.id} · {record.num_questions} questions · "
                f"{record.model} · sty={sty_version} · {record.created_at}"
            )
            cols = st.columns(4)
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

            confirm_key = f"confirm_delete_{record.id}"
            with cols[3]:
                if st.session_state.get(confirm_key):
                    st.warning(f"Delete '{record.title or record.prompt}'? This cannot be undone.")
                    yes, no = st.columns(2)
                    if yes.button("Confirm delete", key=f"do_delete_{record.id}",
                                  type="primary", use_container_width=True):
                        _delete_worksheet(record)
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                    if no.button("Cancel", key=f"cancel_delete_{record.id}",
                                 use_container_width=True):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                else:
                    if st.button("Delete", key=f"ask_delete_{record.id}",
                                 use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()

            if record.tex_source:
                with st.expander("View LaTeX source"):
                    st.code(record.tex_source, language="latex")

            if record.questions_json:
                with st.expander("View questions JSON"):
                    st.code(record.questions_json, language="json")


def render_create() -> None:
    _render_create_ai()
    st.divider()
    _render_create_from_json()


def _render_create_ai() -> None:
    prompt = st.text_area("Worksheet prompt", placeholder="10 question algebra worksheet on solving linear equations, grade 9")
    title = st.text_input("Title (optional)", placeholder="Auto-generated from prompt if left blank")
    num_questions = st.number_input("Number of questions", min_value=1, max_value=50, value=10)
    model = st.selectbox("Claude model", AVAILABLE_MODELS, index=0)
    submitted = st.button("Generate worksheet", type="primary", disabled=not prompt.strip())

    if not submitted:
        return

    out = Path("generated") / uuid4().hex[:8] / "worksheet"
    client = get_client()

    with st.status("Generating worksheet...", expanded=True) as status:
        def on_step(msg: str, detail: str | None = None) -> None:
            status.update(label=msg)
            st.write(msg)
            if detail:
                st.code(detail, language=None)

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
                title=title.strip() or None,
                model=model,
                on_step=on_step,
            )
        except CompileError as e:
            status.update(label="Compilation failed", state="error")
            st.error(f"LaTeX compilation failed after repair attempts:\n\n{e.log_tail}")
            return

        status.update(label="Done", state="complete")

    st.success(f"Created worksheet id={record.id}")
    st.rerun()


def _render_create_from_json() -> None:
    st.subheader("Or enter questions as JSON")
    st.caption(
        "Provide the questions yourself instead of having Claude write them. "
        "The LaTeX is compiled once (no auto-repair); fix the JSON if it fails."
    )
    questions_json = st.text_area(
        "Questions JSON",
        placeholder='[{"id": "1", "text": "$2+2=$", "answer": "4"}, '
        '{"id": "2", "text": "$\\\\frac{1}{2}+\\\\frac{1}{4}=$", "answer": "\\\\frac{3}{4}"}]',
        key="manual_questions_json",
    )
    title = st.text_input("Title", key="manual_title", placeholder="Required")
    submitted = st.button(
        "Create from JSON",
        type="primary",
        disabled=not (questions_json.strip() and title.strip()),
        key="manual_submit",
    )

    if not submitted:
        return

    out = Path("generated") / uuid4().hex[:8] / "worksheet"

    with st.status("Creating worksheet...", expanded=True) as status:
        def on_step(msg: str, detail: str | None = None) -> None:
            status.update(label=msg)
            st.write(msg)
            if detail:
                st.code(detail, language=None)

        try:
            _, _, record = create_worksheet_from_questions(
                questions_json,
                TEMPLATE_PATH,
                out,
                title=title.strip(),
                bucket=BUCKET,
                db_path=DB_PATH,
                on_step=on_step,
            )
        except ValueError as e:
            status.update(label="Invalid questions JSON", state="error")
            st.error(str(e))
            return
        except CompileError as e:
            status.update(label="Compilation failed", state="error")
            st.error(f"LaTeX compilation failed:\n\n{e.log_tail}")
            return

        status.update(label="Done", state="complete")

    st.success(f"Created worksheet id={record.id}")
    st.rerun()


def _display_results(result) -> dict:
    """Builds the issue-#23 JSON: {student -> {worksheet id -> {question id ->
    {answer, response, correct}}}}."""
    return {
        name: {
            worksheet_id: {qid: asdict(res) for qid, res in question_results.items()}
            for worksheet_id, question_results in worksheets.items()
        }
        for name, worksheets in results_by_student(result).items()
    }


def render_grade() -> None:
    st.write(
        "Upload a PDF of scanned student work. Each page is matched to its "
        "worksheet by its QR code, graded against the stored answer key, and "
        "returned as a marked-up PDF plus per-student results."
    )
    uploaded = st.file_uploader("Student work (PDF)", type=["pdf"])
    roster_text = st.text_area(
        "Roster (one student name per line, optional)",
        placeholder="Alice Smith\nBob Jones",
        help="Handwritten names are fuzzy-matched to this list.",
    )
    submitted = st.button("Grade", type="primary", disabled=uploaded is None)

    if not submitted or uploaded is None:
        return

    roster = [line.strip() for line in roster_text.splitlines() if line.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        scan_path = Path(tmp) / "scan.pdf"
        scan_path.write_bytes(uploaded.getvalue())
        marked_path = Path(tmp) / "marked.pdf"

        with st.spinner("Grading..."):
            result = mark_scan([scan_path], roster, DB_PATH, marked_path)

        graded = _display_results(result)
        if not graded:
            st.warning("No pages could be graded from this PDF.")
        else:
            st.success(f"Graded {len(graded)} student(s).")
            st.json(graded)
            st.download_button(
                "Download marked-up PDF",
                data=marked_path.read_bytes(),
                file_name="marked.pdf",
                mime="application/pdf",
            )

        if result.unreadable:
            st.warning("Could not read a worksheet QR code on: " + ", ".join(result.unreadable))
        for worksheet_id, scans in result.unknown_worksheets.items():
            st.warning(f"Worksheet id '{worksheet_id}' is not in the database ({len(scans)} scan(s)).")


def main() -> None:
    st.set_page_config(page_title="GraderBot", layout="wide")
    st.title("GraderBot")

    if not BUCKET:
        st.error("S3_BUCKET is not set. Configure it in .env before using this app.")
        st.stop()

    gallery_tab, create_tab, grade_tab = st.tabs(["Gallery", "Create", "Grade"])
    with gallery_tab:
        render_gallery()
    with create_tab:
        render_create()
    with grade_tab:
        render_grade()


main()
