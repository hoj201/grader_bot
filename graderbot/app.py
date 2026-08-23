"""Streamlit frontend for browsing and creating GraderBot worksheets.

Run with:
  streamlit run graderbot/app.py

Requires ANTHROPIC_API_KEY, S3_BUCKET, and AWS credentials in the
environment (see README.md).
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import anthropic
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from graderbot import embedding, name_classifier, storage
from graderbot.answer_reader import EASYOCR_DEFAULT_ALLOWLIST, EasyOcrAnswerReader
from graderbot.embedding_viz import build_scatter_df
from graderbot.name_dataset import ingest_name_sheets
from graderbot.name_reader import ClassifierNameReader
from graderbot.name_worksheets import generate_name_worksheets
from graderbot.scan_grader import mark_scan, results_by_student
from graderbot.worksheetbot import (
    AVAILABLE_MODELS,
    CompileError,
    build_worksheet,
    create_worksheet_from_questions,
    generate_worksheet_document,
)

load_dotenv()

# Route warnings.warn(...) (e.g. the ingest skip reasons from name_dataset.py)
# through logging too, and make sure app.py's own log lines actually reach
# fly.io's log capture (issue #52 -- previously app.py logged nothing).
logging.captureWarnings(True)
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Set the level on the "graderbot" parent so other graderbot.* module loggers
# (e.g. name_dataset's per-page ingest progress) inherit it too, not just app.py.
logging.getLogger("graderbot").setLevel(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("graderbot.app")

_TEX_DIR = Path(__file__).resolve().parent.parent / "tex"
TEMPLATE_PATH = Path(
    os.environ.get("WORKSHEET_TEMPLATE", str(_TEX_DIR / "worksheet_template.tex"))
)
DB_PATH = Path(os.environ.get("WORKSHEETS_DB_PATH", "worksheets.sqlite3"))
BUCKET = os.environ.get("S3_BUCKET")
# Public base URL this app is reachable at -- used to build the permanent
# `?dl=<public_id>` download links shown in the gallery (see
# build_permanent_download_url / _resolve_download_redirect below). Set this
# explicitly in deployment (fly.toml) rather than relying on the default here
# tracking production, e.g. if a custom domain is ever added.
BASE_URL = os.environ.get("BASE_URL", "https://grader-bot.fly.dev").rstrip("/")

# Below this, a name read off a page is worth eyeballing. With the default KNN
# (k=3) a classifier confidence is one of 0/⅓/⅔/1, so this flags anything short
# of a 2-of-3 majority; for OCR it's the difflib similarity to the roster match.
_LOW_CONFIDENCE = 0.5

_CLASSIFIER_NAME_SOURCE = "Handwriting classifier"
_OCR_NAME_SOURCE = "OCR (Tesseract)"

_MATHPIX_ANSWER_SOURCE = "Mathpix"
_EASYOCR_ANSWER_SOURCE = "EasyOCR"


def _embedder_dim() -> "int | None":
    """Dimension the configured embedder produces, or `None` if that can't be
    determined (no VOYAGE_API_KEY, an unrecognized NAME_EMBEDDER). `None` means
    stored vectors go unfiltered, so a mixed-embedder collection will still
    raise rather than being silently reported as empty."""
    try:
        return embedding.default_embedder().dim
    except Exception:  # noqa: BLE001 - any failure to resolve means "unknown"
        logger.warning("could not determine embedder dimension", exc_info=True)
        return None


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
        "Create or select a class, then add students by typing them in, "
        "importing a CSV, or uploading a scan of filled-in name-learning "
        "worksheets. View, transfer, or remove students below."
    )
    classroom = _select_classroom("roster_classroom", allow_create=True)
    if classroom is None:
        return

    # Ingest below ends with st.rerun() (to refresh the file uploader and
    # roster list), which discards any st.success/warning/error called
    # earlier in that same run. Stash them in session_state instead and
    # flush them here, on the run right after the rerun.
    flash = st.session_state.pop("roster_flash", None)
    if flash:
        for kind, message in flash:
            getattr(st, kind)(message)

    st.subheader("Add a student")
    with st.form("add_student_form", clear_on_submit=True):
        cols = st.columns(3)
        first_name = cols[0].text_input("First name", key="manual_student_first_name")
        last_name = cols[1].text_input("Last name", key="manual_student_last_name")
        nickname = cols[2].text_input("Nickname (optional)", key="manual_student_nickname")
        add_submitted = st.form_submit_button("Add student", type="primary")

    if add_submitted:
        if not first_name.strip() or not last_name.strip():
            st.error("First and last name are required.")
        else:
            conn = storage.init_db(DB_PATH)
            try:
                student = storage.get_or_create_student(
                    conn, classroom.id, first_name.strip(), last_name.strip(),
                    nickname.strip() or None,
                )
            finally:
                conn.close()
            logger.info(
                "added student id=%s classroom=%s (manual)", student.id, classroom.id
            )
            st.success(f"Added {student.first_name} {student.last_name}.")

    st.divider()
    st.subheader("Import students from CSV")
    st.caption(
        "The CSV needs a header row with `first_name` and `last_name` "
        "columns, plus an optional `nickname` column (column order and "
        "case don't matter). Example:"
    )
    st.code("first_name,last_name,nickname\nAnna,Smith,\nZeke,Jones,Z", language="text")
    csv_uploaded = st.file_uploader("Roster CSV", type=["csv"], key="roster_csv_upload")
    csv_submitted = st.button(
        "Import CSV", type="primary", disabled=csv_uploaded is None, key="import_csv_button"
    )
    if csv_submitted and csv_uploaded is not None:
        csv_text = csv_uploaded.getvalue().decode("utf-8-sig")
        conn = storage.init_db(DB_PATH)
        try:
            try:
                result = storage.import_students_csv(conn, classroom.id, csv_text)
            except ValueError as exc:
                result = None
                st.error(str(exc))
        finally:
            conn.close()
        if result is not None:
            logger.info(
                "imported students csv classroom=%s added=%d skipped=%d",
                classroom.id, len(result.added), len(result.skipped),
            )
            st.success(f"Added {len(result.added)} student(s) from CSV.")
            if result.skipped:
                st.warning(
                    f"{len(result.skipped)} row(s) were skipped:\n"
                    + "\n".join(f"- {reason}" for reason in result.skipped)
                )

    st.divider()
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
                def on_step(msg: str, detail: str | None = None) -> None:
                    status.update(label=msg)
                    st.write(msg)

                result = ingest_name_sheets(
                    str(scan_path), DB_PATH, classroom.id, bucket=BUCKET, on_step=on_step
                )
                status.update(label="Embedding handwriting samples...")
                st.write("Embedding handwriting samples...")
                try:
                    num_vectorized = embedding.vectorize_samples(DB_PATH, bucket=BUCKET)
                except Exception as exc:
                    num_vectorized = None
                    vectorize_error = str(exc)
                    status.update(label="Embedding failed", state="error")
                else:
                    vectorize_error = None
                    status.update(label="Done", state="complete")

        logger.info(
            "ingest_name_sheets classroom=%s ingested=%d skipped=%d vectorized=%s",
            classroom.id, len(result.records), len(result.skipped),
            num_vectorized if vectorize_error is None else "failed",
        )
        flash: list[tuple[str, str]] = [
            ("success", f"Ingested {len(result.records)} handwriting sample(s).")
        ]
        if result.skipped:
            flash.append((
                "warning",
                f"{len(result.skipped)} page(s) were skipped:\n"
                + "\n".join(f"- {reason}" for reason in result.skipped),
            ))
        if vectorize_error is not None:
            flash.append((
                "error",
                "Handwriting samples were saved, but embedding failed: "
                f"{vectorize_error}\nStudents were added without name "
                "embeddings; re-upload the same scan to retry once the "
                "problem is fixed (already-embedded samples are skipped, "
                "so this is safe to repeat).",
            ))
        st.session_state["roster_flash"] = flash
        st.rerun()

    st.divider()
    st.subheader(f"Roster: {classroom.label}")
    conn = storage.init_db(DB_PATH)
    try:
        students = storage.list_students(conn, classroom.id)
        other_classrooms = [c for c in storage.list_classrooms(conn) if c.id != classroom.id]
    finally:
        conn.close()

    if not students:
        st.info("No students in this class yet.")
        return

    for student in students:
        with st.container(border=True):
            cols = st.columns([3, 3, 1])
            label = f"{student.first_name} {student.last_name}".strip()
            if student.nickname:
                label += f" ({student.nickname})"
            cols[0].markdown(label)

            # Transfer (issue #72): moves the STUDENT row to another
            # classroom while leaving NAME_IMAGES/NAME_EMBEDDINGS (the
            # handwriting "signature") untouched, unlike delete-and-re-add.
            transfer_confirm_key = f"confirm_transfer_student_{student.id}"
            transfer_target_key = f"transfer_target_{student.id}"
            with cols[1]:
                if not other_classrooms:
                    st.caption("No other classes to transfer to.")
                elif st.session_state.get(transfer_confirm_key):
                    target_label = st.session_state[transfer_target_key]
                    st.warning(f"Transfer to {target_label}?")
                    yes, no = st.columns(2)
                    if yes.button("Confirm", key=f"do_transfer_student_{student.id}",
                                  type="primary", use_container_width=True):
                        target = next(c for c in other_classrooms if c.label == target_label)
                        conn = storage.init_db(DB_PATH)
                        try:
                            transfer_error = None
                            try:
                                storage.transfer_student(conn, student.id, target.id)
                            except ValueError as exc:
                                transfer_error = str(exc)
                        finally:
                            conn.close()
                        if transfer_error:
                            st.error(transfer_error)
                        else:
                            logger.info(
                                "transferred student id=%s from classroom=%s to classroom=%s",
                                student.id, classroom.id, target.id,
                            )
                            st.session_state.pop(transfer_confirm_key, None)
                            st.session_state.pop(transfer_target_key, None)
                            st.rerun()
                    if no.button("Cancel", key=f"cancel_transfer_student_{student.id}",
                                 use_container_width=True):
                        st.session_state.pop(transfer_confirm_key, None)
                        st.session_state.pop(transfer_target_key, None)
                        st.rerun()
                else:
                    select_col, button_col = st.columns([2, 1])
                    target_label = select_col.selectbox(
                        "Transfer to", [c.label for c in other_classrooms],
                        key=f"transfer_select_{student.id}", label_visibility="collapsed",
                    )
                    if button_col.button("Transfer", key=f"ask_transfer_student_{student.id}",
                                          use_container_width=True):
                        st.session_state[transfer_confirm_key] = True
                        st.session_state[transfer_target_key] = target_label
                        st.rerun()

            confirm_key = f"confirm_delete_student_{student.id}"
            with cols[2]:
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
                        logger.info(
                            "deleted student id=%s classroom=%s", student.id, classroom.id
                        )
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


def render_visualize() -> None:
    st.write(
        "3D t-SNE projection of each student's handwriting-sample embeddings, "
        "for debugging the name classifier."
    )
    classroom = _select_classroom("visualize_classroom", allow_create=False)
    if classroom is None:
        return

    conn = storage.init_db(DB_PATH)
    try:
        students = storage.list_students(conn, classroom.id)
    finally:
        conn.close()

    if not students:
        st.info("No students in this class yet.")
        return

    vectors, student_ids, name_image_ids, discarded = embedding.load_training_vectors(
        DB_PATH, bucket=BUCKET, classroom_id=classroom.id, dim=_embedder_dim()
    )
    if discarded:
        st.warning(
            f"Ignored {discarded} embedding(s) produced by a different embedder. "
            "Delete and re-ingest those students to re-vectorize them with the "
            "current one."
        )
    df = build_scatter_df(vectors, student_ids, name_image_ids, students)

    if df.empty:
        st.info("No handwriting-sample embeddings yet for this class.")
        return

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="student_name",
        hover_data={
            "name_image_id": True,
            "student_name": True,
            "x": False,
            "y": False,
            "z": False,
        },
        title=f"Handwriting embeddings: {classroom.label}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Classifier accuracy")
    st.write(
        "Leave-one-out cross-validation of the name classifier: each "
        "handwriting sample is held out and predicted from the rest, "
        "scored one student at a time (issue #55)."
    )
    if st.button("Evaluate classifier", key="visualize_evaluate_classifier"):
        with st.spinner("Running leave-one-out cross-validation..."):
            accuracy, insufficient, confusion = name_classifier.loo_cross_validate(
                vectors, student_ids
            )
        name_by_id = {s.id: f"{s.first_name} {s.last_name}".strip() for s in students}
        if accuracy:
            rows = [
                {"Student": name_by_id.get(student_id, str(student_id)), "LOO accuracy": acc}
                for student_id, acc in accuracy.items()
            ]
            rows.sort(key=lambda row: row["Student"])
            for row in rows:
                row["LOO accuracy"] = f"{row['LOO accuracy']:.0%}"
            st.table(rows)

            st.caption("Confusion matrix (rows: actual student, columns: predicted)")
            predicted_ids = {pred for preds in confusion.values() for pred in preds}
            column_ids = sorted(
                set(accuracy) | predicted_ids, key=lambda sid: name_by_id.get(sid, str(sid))
            )
            matrix_rows = []
            for true_id in sorted(accuracy, key=lambda sid: name_by_id.get(sid, str(sid))):
                row = {"Actual": name_by_id.get(true_id, str(true_id))}
                for pred_id in column_ids:
                    label = name_by_id.get(pred_id, str(pred_id))
                    row[label] = confusion.get(true_id, {}).get(pred_id, 0)
                matrix_rows.append(row)
            st.dataframe(matrix_rows, use_container_width=True)
        else:
            st.info("No students have enough samples yet for cross-validation.")
        if insufficient:
            names = ", ".join(name_by_id.get(sid, str(sid)) for sid in insufficient)
            st.caption(f"Insufficient data (need ≥2 samples): {names}")

    st.divider()
    st.subheader("Train classifier")
    st.write(
        "Fit a classifier on this class's handwriting samples and save it, so "
        "the Grade tab can identify students by their handwriting instead of "
        "by OCR (issue #58). Retrain after ingesting new name sheets — the "
        "saved model does not update on its own."
    )
    if st.button("Train classifier", key="visualize_train_classifier"):
        with st.spinner("Training..."):
            try:
                report = name_classifier.train_classroom_classifier(
                    DB_PATH, BUCKET, classroom.id
                )
            except ValueError as e:
                st.error(str(e))
                return
        logger.info(
            "trained name classifier classroom=%s samples=%d students=%d",
            classroom.id, report.n_samples, report.n_students,
        )
        st.success(
            f"Trained on {report.n_samples} sample(s) from {report.n_students} "
            f"student(s) and saved to {report.s3_url}"
        )
        if report.discarded_wrong_dim:
            st.warning(
                f"Skipped {report.discarded_wrong_dim} embedding(s) from a "
                "different embedder."
            )
        if report.students_with_no_samples:
            st.warning(
                "No handwriting samples, so these students can never be "
                "recognized: " + ", ".join(report.students_with_no_samples)
            )
        if report.students_with_one_sample:
            st.warning(
                "Only one sample each, so recognition will be unreliable for: "
                + ", ".join(report.students_with_one_sample)
            )


def build_permanent_download_url(base_url: str, public_id: str) -> str:
    """Copy-pasteable, non-expiring link for a worksheet's student PDF.

    Unlike the presigned S3 URLs shown elsewhere in the gallery (which expire
    after an hour), this URL never changes. Visiting it re-enters this app,
    which mints a fresh presigned S3 URL server-side and redirects -- see
    _resolve_download_redirect and its use in main().
    """
    return f"{base_url}/?dl={public_id}"


def _resolve_download_redirect(conn, public_id: str) -> tuple[str | None, str | None]:
    """Resolves a `dl` query-param public_id to a fresh presigned URL for
    that worksheet's student PDF.

    Returns (presigned_url, error_message) -- exactly one is non-None.
    Deliberately scoped to the student-facing PDF only; CV/answer-key PDFs
    aren't exposed through this permanent-link route.
    """
    record = storage.get_worksheet_by_public_id(conn, public_id)
    if record is None:
        return None, f"No worksheet found for id '{public_id}'."
    if not record.student_pdf_s3url:
        return None, "This worksheet has no student PDF available."
    bucket, key = storage.parse_s3_url(record.student_pdf_s3url)
    return storage.generate_presigned_url(bucket, key), None


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
            with cols[0]:
                if record.student_pdf_s3url:
                    bucket, key = storage.parse_s3_url(record.student_pdf_s3url)
                    presigned = storage.generate_presigned_url(bucket, key)
                    st.link_button("Student", presigned, use_container_width=True)
                    if record.public_id:
                        st.caption("Permanent link:")
                        st.code(
                            build_permanent_download_url(BASE_URL, record.public_id),
                            language=None,
                        )
                else:
                    st.button("Student", disabled=True, use_container_width=True)

            # CV and answer-key PDFs stay presigned-only (1hr expiry) -- the
            # permanent-link feature is deliberately scoped to the student PDF.
            for col, label, url in zip(
                cols[1:],
                ("CV", "Answer key"),
                (record.cv_pdf_s3url, record.answers_pdf_s3url),
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
                        logger.info("deleted worksheet id=%s", record.id)
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
    # Both the AI-generation accept flow and the manual-JSON flow end with
    # st.rerun() (to clear the just-submitted preview/form state), which
    # discards any st.success called earlier in that same run (issue #64).
    # Stash it in session_state instead and flush it here, on the run right
    # after the rerun -- same pattern as roster_flash above.
    flash = st.session_state.pop("create_flash", None)
    if flash:
        for kind, message in flash:
            getattr(st, kind)(message)

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
    submitted = st.button("Generate questions", type="primary", disabled=not prompt.strip())

    if submitted:
        client = get_client()
        with st.status("Generating questions...", expanded=True) as status:
            def on_step(msg: str, detail: str | None = None) -> None:
                status.update(label=msg)
                st.write(msg)
                if detail:
                    st.code(detail, language=None)

            document = generate_worksheet_document(
                client,
                prompt,
                num_questions=int(num_questions),
                title=title.strip() or None,
                header=header.strip() or None,
                model=model,
                on_step=on_step,
            )
            status.update(label="Done", state="complete")

        # issue #68: hold the generated document for review instead of
        # compiling immediately -- survives the rerun that follows every
        # widget interaction, since a plain local variable would not.
        st.session_state["ai_preview"] = {"document": document, "prompt": prompt, "model": model}
        st.rerun()

    _render_ai_preview()


def _render_ai_preview() -> None:
    """The accept/reject step from issue #68: shows the questions a prior
    'Generate questions' click produced (stashed in session_state, since a
    widget click always triggers a rerun) so the user can review the JSON
    before any LaTeX is compiled."""
    preview = st.session_state.get("ai_preview")
    if not preview:
        return

    document = preview["document"]
    st.subheader("Review generated questions")
    st.caption(f"Title: {document.title}")
    if document.header:
        st.caption(f"Header: {document.header}")
    questions_json = json.dumps([asdict(q) for q in document.questions], indent=2)
    st.code(questions_json, language="json")

    accept_col, reject_col = st.columns(2)
    accept = accept_col.button("Accept and compile", type="primary", key="ai_preview_accept")
    reject = reject_col.button("Reject", key="ai_preview_reject")

    if accept:
        out = Path("generated") / uuid4().hex[:8] / "worksheet"
        client = get_client()
        with st.status("Compiling worksheet...", expanded=True) as status:
            def on_step(msg: str, detail: str | None = None) -> None:
                status.update(label=msg)
                st.write(msg)
                if detail:
                    st.code(detail, language=None)

            try:
                _, _, record = build_worksheet(
                    document,
                    TEMPLATE_PATH,
                    out,
                    max_repairs=3,
                    client=client,
                    bucket=BUCKET,
                    db_path=DB_PATH,
                    prompt=preview["prompt"],
                    model=preview["model"],
                    on_step=on_step,
                )
            except CompileError as e:
                logger.error("worksheet compile failed (AI): %s", e.log_tail)
                status.update(label="Compilation failed", state="error")
                st.error(f"LaTeX compilation failed after repair attempts:\n\n{e.log_tail}")
                return

            status.update(label="Done", state="complete")

        logger.info(
            "created worksheet id=%s model=%s num_questions=%s",
            record.id,
            preview["model"],
            len(document.questions),
        )
        st.session_state.pop("ai_preview", None)
        st.session_state["create_flash"] = [("success", f"Created worksheet id={record.id}")]
        st.rerun()

    if reject:
        # Hand off to the manual JSON form below (issue #68) by pre-filling
        # its widgets via their session_state keys, so the user can tweak
        # the generated JSON before submitting through the existing manual
        # (no-AI-repair) compile path.
        st.session_state["manual_questions_json"] = questions_json
        st.session_state["manual_title"] = document.title
        st.session_state["manual_header"] = document.header
        st.session_state.pop("ai_preview", None)
        st.info("Copied into the JSON form below for editing.")
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
        help='Add "open_ended": true (and leave "answer": "") to a question '
        "with no single correct answer, e.g. an opinion/reflection prompt -- "
        "it's shown to students normally but never marked right or wrong.",
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
            logger.error("invalid questions JSON: %s", e)
            status.update(label="Invalid questions JSON", state="error")
            st.error(str(e))
            return
        except CompileError as e:
            logger.error("worksheet compile failed (JSON): %s", e.log_tail)
            status.update(label="Compilation failed", state="error")
            st.error(f"LaTeX compilation failed:\n\n{e.log_tail}")
            return

        status.update(label="Done", state="complete")

    logger.info("created worksheet id=%s (from JSON)", record.id)
    st.session_state["create_flash"] = [("success", f"Created worksheet id={record.id}")]
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


def _classifier_exists(classroom_id: int) -> bool:
    """Whether a trained handwriting classifier is saved for this class."""
    if not BUCKET:
        return False
    try:
        storage._default_s3_client().head_object(
            Bucket=BUCKET, Key=name_classifier.classifier_key(classroom_id)
        )
        return True
    except Exception:  # noqa: BLE001 - a 404 and a credentials failure both mean "can't use it"
        return False


def _display_name_predictions(result) -> None:
    """Show how each page's student was identified, so a doubtful read can be
    caught by eye (issue #58). Listed per page rather than per student because
    two pages that resolve to the same name collapse in the results JSON."""
    if not result.name_predictions:
        return
    st.caption("Student names read from each page")
    st.dataframe(
        [
            {
                "Page": p.page,
                "Worksheet": p.worksheet_id,
                "Student": p.name or "(no name)",
                "Confidence": f"{p.confidence:.0%}",
                "Read by": p.source,
            }
            for p in result.name_predictions
        ],
        use_container_width=True,
    )
    doubtful = [p for p in result.name_predictions if p.confidence < _LOW_CONFIDENCE]
    if doubtful:
        pages = ", ".join(f"{p.page} → {p.name or '(no name)'}" for p in doubtful)
        st.warning(f"Low-confidence name(s), worth checking by hand: {pages}")


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

    has_model = classroom is not None and _classifier_exists(classroom.id)
    options = [_CLASSIFIER_NAME_SOURCE, _OCR_NAME_SOURCE]
    name_source = st.selectbox(
        "Read student names with",
        options,
        index=0 if has_model else 1,
        key="grade_name_source",
    )
    if not has_model:
        st.caption(
            "No handwriting classifier has been trained for this class yet — "
            "train one on the Visualize tab to use it here."
        )

    answer_options = [_MATHPIX_ANSWER_SOURCE, _EASYOCR_ANSWER_SOURCE]
    answer_source = st.selectbox(
        "Read answers with",
        answer_options,
        index=0,
        key="grade_answer_source",
    )
    easyocr_extra_chars = ""
    if answer_source == _EASYOCR_ANSWER_SOURCE:
        easyocr_extra_chars = st.text_input(
            f"Extra characters to allow (appended to '{EASYOCR_DEFAULT_ALLOWLIST}')",
            key="grade_easyocr_extra_chars",
            help="EasyOCR only recognizes characters in this allowlist. Widen it "
            "for a worksheet that needs more (e.g. 'xy' for algebra).",
        )
        st.caption(
            "EasyOCR can't read fractions — use Mathpix for worksheets with "
            "fraction answers. Requires the easyocr_service sidecar "
            "(`docker compose up -d easyocr`) and EASYOCR_SERVICE_URL set."
        )

    submitted = st.button("Grade", type="primary", disabled=uploaded is None)

    if not submitted or uploaded is None:
        return

    name_reader = None
    if name_source == _CLASSIFIER_NAME_SOURCE:
        if classroom is None:
            st.error("Select a class to grade with the handwriting classifier.")
            return
        try:
            name_reader = ClassifierNameReader.from_classroom(DB_PATH, classroom.id, BUCKET)
        except ValueError as e:
            st.error(str(e))
            return
        if name_reader is None:
            st.error(
                "No handwriting classifier is saved for this class. Train one on "
                "the Visualize tab, or switch to OCR above."
            )
            return

    answer_reader = None
    if answer_source == _EASYOCR_ANSWER_SOURCE:
        allowlist = EASYOCR_DEFAULT_ALLOWLIST + easyocr_extra_chars
        try:
            answer_reader = EasyOcrAnswerReader(allowlist=allowlist)
        except EnvironmentError as e:
            st.error(str(e))
            return

    with tempfile.TemporaryDirectory() as tmp:
        scan_suffix = Path(uploaded.name).suffix or ".pdf"
        scan_path = Path(tmp) / f"scan{scan_suffix}"
        scan_path.write_bytes(uploaded.getvalue())
        marked_path = Path(tmp) / "marked.pdf"
        logger.info("grading scan filename=%s", uploaded.name)

        with st.status("Grading...", expanded=True) as status:
            def on_step(msg: str, detail: str | None = None) -> None:
                status.update(label=msg)
                st.write(msg)
                if detail:
                    st.code(detail, language=None)

            result = mark_scan(
                [scan_path],
                roster,
                DB_PATH,
                marked_path,
                on_step=on_step,
                name_reader=name_reader,
                answer_reader=answer_reader,
            )
            status.update(label="Grading complete", state="complete")

        graded = _display_results(result)
        logger.info(
            "graded students=%d unreadable=%d unknown_worksheets=%d",
            len(graded), len(result.unreadable), len(result.unknown_worksheets),
        )
        _display_name_predictions(result)
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
            logger.error("name worksheet generation failed: %s", e)
            status.update(label="Compilation failed", state="error")
            st.error(f"LaTeX compilation failed:\n\n{e}")
            return
        status.update(label="Done", state="complete")

    logger.info("generated %d name worksheet(s) classroom=%s", len(names), classroom.id)
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

    # Permanent download links (see build_permanent_download_url) point back
    # at this app with `?dl=<public_id>`. Resolve that here, before anything
    # else renders, and redirect to a freshly-minted presigned S3 URL.
    dl_id = st.query_params.get("dl")
    if dl_id:
        conn = storage.init_db(DB_PATH)
        presigned, error = _resolve_download_redirect(conn, dl_id)
        conn.close()
        if error:
            st.error(error)
        else:
            st.markdown(
                f'<meta http-equiv="refresh" content="0;url={presigned}">',
                unsafe_allow_html=True,
            )
            st.write("Redirecting to your download...")
            st.link_button("Click here if the download doesn't start automatically", presigned)
        st.stop()

    if not BUCKET:
        st.error("S3_BUCKET is not set. Configure it in .env before using this app.")
        st.stop()

    gallery_tab, create_tab, grade_tab, names_tab, roster_tab, visualize_tab = st.tabs(
        ["Gallery", "Create", "Grade", "Name sheets", "Roster", "Visualize"]
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
    with visualize_tab:
        render_visualize()


if __name__ == "__main__":
    main()
