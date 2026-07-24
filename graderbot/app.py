"""Streamlit frontend for browsing and creating GraderBot worksheets.

Run with:
  streamlit run graderbot/app.py

Requires ANTHROPIC_API_KEY, S3_BUCKET, and AWS credentials in the
environment (see README.md).
"""

import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import anthropic
import streamlit as st
from dotenv import load_dotenv

from graderbot import storage
from graderbot.name_dataset import ingest_name_sheets
from graderbot.name_worksheets import generate_name_worksheets
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


_NEW_CLASSROOM_OPTION = "<create new class>"


def _select_classroom(key: str, allow_create: bool) -> "storage.ClassroomRecord | None":
    """Renders a classroom selectbox (optionally with a "create new" option)
    and returns the selected/created `ClassroomRecord`, or `None` if there are
    no classrooms yet and none was created (issue #43)."""
    conn = storage.init_db(DB_PATH)
    try:
        classrooms = storage.list_classrooms(conn)
    finally:
        conn.close()

    options = [c.label for c in classrooms]
    if allow_create:
        options = options + [_NEW_CLASSROOM_OPTION]
    if not options:
        st.info("No classes yet. Create one in the Roster tab.")
        return None

    choice = st.selectbox("Classroom", options, key=key)
    if choice == _NEW_CLASSROOM_OPTION:
        new_label = st.text_input("New classroom name", key=f"{key}_new_label")
        if not new_label.strip():
            return None
        conn = storage.init_db(DB_PATH)
        try:
            return storage.get_or_create_classroom(conn, new_label.strip())
        finally:
            conn.close()

    return next(c for c in classrooms if c.label == choice)


def render_roster() -> None:
    st.write(
        "Create or select a class, upload a scan of filled-in name-learning "
        "worksheets to add students to it, then view or remove students."
    )
    classroom = _select_classroom("roster_classroom", allow_create=True)
    if classroom is None:
        return

    st.subheader("Upload name-learning worksheets")
    uploaded = st.file_uploader(
        "Scanned PDF or image", type=["pdf", "jpg", "jpeg", "png"], key="roster_upload"
    )
    submitted = st.button("Ingest", type="primary", disabled=uploaded is None)

    if submitted and uploaded is not None:
        with tempfile.TemporaryDirectory() as tmp:
            scan_suffix = Path(uploaded.name).suffix or ".pdf"
            scan_path = Path(tmp) / f"scan{scan_suffix}"
            scan_path.write_bytes(uploaded.getvalue())

            with st.status("Ingesting worksheets...", expanded=True) as status:
                result = ingest_name_sheets(
                    str(scan_path), DB_PATH, classroom.id, bucket=BUCKET
                )
                for reason in result.skipped:
                    st.write(f"Skipped {reason}")
                status.update(label="Done", state="complete")

        st.success(f"Ingested {len(result.records)} handwriting sample(s).")
        if result.skipped:
            st.warning(
                f"{len(result.skipped)} page(s) were skipped:\n"
                + "\n".join(f"- {reason}" for reason in result.skipped)
            )
        st.rerun()

    st.divider()
    st.subheader(f"Roster: {classroom.label}")
    conn = storage.init_db(DB_PATH)
    try:
        students = storage.list_students(conn, classroom.id)
    finally:
        conn.close()

    if not students:
        st.info("No students in this class yet.")
        return

    for student in students:
        with st.container(border=True):
            cols = st.columns([4, 1])
            label = f"{student.first_name} {student.last_name}".strip()
            if student.nickname:
                label += f" ({student.nickname})"
            cols[0].markdown(label)

            confirm_key = f"confirm_delete_student_{student.id}"
            with cols[1]:
                if st.session_state.get(confirm_key):
                    st.warning(f"Delete {label}?")
                    yes, no = st.columns(2)
                    if yes.button("Confirm", key=f"do_delete_student_{student.id}",
                                  type="primary", use_container_width=True):
                        conn = storage.init_db(DB_PATH)
                        try:
                            storage.delete_student(conn, student.id)
                        finally:
                            conn.close()
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                    if no.button("Cancel", key=f"cancel_delete_student_{student.id}",
                                 use_container_width=True):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                else:
                    if st.button("Delete", key=f"ask_delete_student_{student.id}",
                                 use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()


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
    header = st.text_area(
        "Header / instructions (optional)",
        placeholder="Auto-generated from prompt if left blank",
    )
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
                header=header.strip() or None,
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
    header = st.text_area(
        "Header / instructions (optional)",
        key="manual_header",
        help="Inserted into the LaTeX source as-is (like the questions above), "
        "so use $...$ for inline math (e.g. $\\frac{a}{b}$) and escape any "
        "literal %, &, #, _, {, }, ~, ^, or \\ that isn't LaTeX markup.",
    )
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
                header=header.strip(),
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
        "Upload a PDF, JPEG, or PNG of scanned student work. Each page is "
        "matched to its worksheet by its QR code, graded against the stored "
        "answer key, and returned as a marked-up PDF plus per-student results."
    )
    uploaded = st.file_uploader(
        "Student work (PDF, JPEG, or PNG)", type=["pdf", "jpg", "jpeg", "png"]
    )
    classroom = _select_classroom("grade_classroom", allow_create=False)
    roster = []
    if classroom is not None:
        conn = storage.init_db(DB_PATH)
        try:
            roster = [
                f"{s.first_name} {s.last_name}".strip()
                for s in storage.list_students(conn, classroom.id)
            ]
        finally:
            conn.close()
    submitted = st.button("Grade", type="primary", disabled=uploaded is None)

    if not submitted or uploaded is None:
        return

    with tempfile.TemporaryDirectory() as tmp:
        scan_suffix = Path(uploaded.name).suffix or ".pdf"
        scan_path = Path(tmp) / f"scan{scan_suffix}"
        scan_path.write_bytes(uploaded.getvalue())
        marked_path = Path(tmp) / "marked.pdf"

        with st.status("Grading...", expanded=True) as status:
            def on_step(msg: str, detail: str | None = None) -> None:
                status.update(label=msg)
                st.write(msg)
                if detail:
                    st.code(detail, language=None)

            result = mark_scan(
                [scan_path], roster, DB_PATH, marked_path, on_step=on_step
            )
            status.update(label="Grading complete", state="complete")

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


def render_name_sheets() -> None:
    st.write(
        "Select a class to generate a printable PDF of name-collection "
        "worksheets — one page per student, with their name printed at the "
        "top to copy into the practice grid."
    )
    classroom = _select_classroom("names_classroom", allow_create=False)
    names = []
    if classroom is not None:
        conn = storage.init_db(DB_PATH)
        try:
            names = [
                f"{s.first_name} {s.last_name}".strip()
                for s in storage.list_students(conn, classroom.id)
            ]
        finally:
            conn.close()
    submitted = st.button(
        "Generate name worksheets", type="primary", disabled=not names
    )

    if not submitted:
        return

    with st.status(f"Generating {len(names)} worksheet(s)...", expanded=True) as status:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "name_worksheets.pdf"
                generate_name_worksheets(names, out_path)
                pdf_bytes = out_path.read_bytes()
        except subprocess.CalledProcessError as e:
            status.update(label="Compilation failed", state="error")
            st.error(f"LaTeX compilation failed:\n\n{e}")
            return
        status.update(label="Done", state="complete")

    st.success(f"Generated {len(names)} name worksheet(s).")
    st.download_button(
        "Download name worksheets PDF",
        data=pdf_bytes,
        file_name="name_worksheets.pdf",
        mime="application/pdf",
    )


def main() -> None:
    st.set_page_config(page_title="GraderBot", layout="wide")
    st.title("GraderBot")

    if not BUCKET:
        st.error("S3_BUCKET is not set. Configure it in .env before using this app.")
        st.stop()

    gallery_tab, create_tab, grade_tab, names_tab, roster_tab = st.tabs(
        ["Gallery", "Create", "Grade", "Name sheets", "Roster"]
    )
    with gallery_tab:
        render_gallery()
    with create_tab:
        render_create()
    with grade_tab:
        render_grade()
    with names_tab:
        render_name_sheets()
    with roster_tab:
        render_roster()


main()
