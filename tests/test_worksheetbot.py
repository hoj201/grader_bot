import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from graderbot.worksheetbot import (
    AVAILABLE_MODELS,
    MODEL,
    CompileError,
    Question,
    choose_grid_columns,
    create_worksheet_from_questions,
    escape_latex,
    estimate_question_width,
    generate_questions,
    generate_title,
    generate_worksheet,
    parse_questions_json,
    render_questions,
    repair_tex,
)


def _fake_client(raw_text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=raw_text)]
    )
    return client


def test_generate_questions_parses_plain_json_array():
    data = [
        {"id": "1", "text": "2 + 2 = ?", "answer": "4"},
        {"id": "2", "text": "3 * 3 = ?", "answer": "9"},
    ]
    client = _fake_client(json.dumps(data))

    questions = generate_questions(client, "arithmetic worksheet", 2)

    assert questions == [
        Question(id="1", text="2 + 2 = ?", answer="4"),
        Question(id="2", text="3 * 3 = ?", answer="9"),
    ]


def test_generate_questions_strips_markdown_code_fences():
    data = [{"id": "1", "text": "x + 1 = 2", "answer": "1"}]
    raw_text = f"```json\n{json.dumps(data)}\n```"
    client = _fake_client(raw_text)

    questions = generate_questions(client, "algebra worksheet", 1)

    assert questions == [Question(id="1", text="x + 1 = 2", answer="1")]


def test_generate_questions_repairs_unescaped_latex_backslashes():
    # The model sometimes forgets to double LaTeX backslashes, e.g. it emits
    # a literal "\div" instead of "\\div". "\d" isn't a valid JSON escape,
    # so json.loads raises "Invalid \escape" and the whole request used to
    # blow up with no recovery.
    raw_text = r'[{"id": "1", "text": "$8 \div 4 = ?$", "answer": "2"}]'
    client = _fake_client(raw_text)

    questions = generate_questions(client, "division worksheet", 1)

    assert questions == [Question(id="1", text=r"$8 \div 4 = ?$", answer="2")]


def test_generate_questions_warns_on_count_mismatch(capsys):
    data = [{"id": "1", "text": "1 + 1 = ?", "answer": "2"}]
    client = _fake_client(json.dumps(data))

    questions = generate_questions(client, "arithmetic worksheet", 3)

    assert len(questions) == 1
    assert "requested 3 questions, got 1" in capsys.readouterr().err


def test_generate_questions_sends_prompt_and_count_to_model():
    data = [{"id": "1", "text": "q", "answer": "a"}]
    client = _fake_client(json.dumps(data))

    generate_questions(client, "fractions worksheet", 5)

    _, kwargs = client.messages.create.call_args
    user_message = kwargs["messages"][0]["content"]
    assert "fractions worksheet" in user_message
    assert "exactly 5 questions" in user_message


def test_default_model_is_haiku():
    # Issue #22: the default worksheet-generation model is Haiku.
    assert MODEL == "claude-haiku-4-5"
    assert AVAILABLE_MODELS[0] == MODEL


def test_generate_questions_defaults_to_default_model():
    client = _fake_client(json.dumps([{"id": "1", "text": "q", "answer": "a"}]))

    generate_questions(client, "arithmetic worksheet", 1)

    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == MODEL


def test_generate_questions_uses_explicit_model():
    client = _fake_client(json.dumps([{"id": "1", "text": "q", "answer": "a"}]))

    generate_questions(client, "arithmetic worksheet", 1, model="claude-opus-4-8")

    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-opus-4-8"


def test_generate_title_uses_explicit_model():
    client = _fake_client("Some Title")

    generate_title(client, "algebra worksheet", model="claude-opus-4-8")

    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-opus-4-8"


def test_repair_tex_uses_explicit_model():
    client = _fake_client("FIXED SOURCE")

    repair_tex(client, "BROKEN", "log tail", model="claude-sonnet-4-6")

    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_generate_worksheet_threads_model_through_generation_and_storage(tmp_path):
    client = _client_with_responses(
        json.dumps([{"id": "1", "text": "1+1=?", "answer": "2"}]), "Auto Title"
    )
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"
    db_path = tmp_path / "worksheets.sqlite3"
    fake_record = SimpleNamespace(
        id=1,
        student_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/student.pdf",
        cv_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/cv.pdf",
        answers_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/answers.pdf",
    )

    with patch("graderbot.worksheetbot.compile_tex", return_value=(True, "")), patch(
        "graderbot.worksheetbot.storage.store_worksheet", return_value=fake_record
    ) as mock_store:
        generate_worksheet(
            client,
            template_path,
            "arithmetic",
            out,
            num_questions=1,
            max_repairs=3,
            bucket="my-bucket",
            db_path=db_path,
            model="claude-opus-4-8",
        )

    # Every generation call and the stored record use the chosen model.
    for _, kwargs in client.messages.create.call_args_list:
        assert kwargs["model"] == "claude-opus-4-8"
    assert mock_store.call_args.kwargs["model"] == "claude-opus-4-8"


def test_generate_title_returns_stripped_text():
    client = _fake_client("  Linear Equations Practice  ")

    title = generate_title(client, "algebra worksheet")

    assert title == "Linear Equations Practice"


def test_generate_title_strips_surrounding_quotes():
    client = _fake_client('"Fractions Warmup"')

    title = generate_title(client, "fractions worksheet")

    assert title == "Fractions Warmup"


def test_generate_title_sends_prompt_to_model():
    client = _fake_client("Some Title")

    generate_title(client, "division worksheet")

    _, kwargs = client.messages.create.call_args
    assert kwargs["messages"][0]["content"] == "division worksheet"


def test_escape_latex_escapes_special_characters():
    assert escape_latex(r"50% & $5 #1 {a}_b ~x^y \z") == (
        r"50\% \& \$5 \#1 \{a\}\_b \textasciitilde{}x\textasciicircum{}y "
        r"\textbackslash{}z"
    )


def test_render_questions_uses_question_macro_with_id_and_text():
    questions = [Question(id="add001", text="$7+5=$", answer="12")]

    tex = render_questions(questions)

    assert tex == "\\begin{enumerate}\n    \\item \\Question{add001}{$7+5=$}\n\\end{enumerate}"


def test_render_questions_does_not_escape_question_text():
    questions = [Question(id="frac001", text=r"$\frac{1}{4}+\frac{1}{2}=$", answer=r"\frac{3}{4}")]

    tex = render_questions(questions)

    assert tex == (
        "\\begin{enumerate}\n"
        "    \\item \\Question{frac001}{$\\frac{1}{4}+\\frac{1}{2}=$}\n"
        "\\end{enumerate}"
    )


def test_render_questions_escapes_special_characters_in_id():
    questions = [Question(id="q_1", text="text", answer="1")]

    tex = render_questions(questions)

    assert tex == "\\begin{enumerate}\n    \\item \\Question{q\\_1}{text}\n\\end{enumerate}"


def test_render_questions_stacks_multiple_questions_in_an_enumerate():
    questions = [
        Question(id="1", text="a", answer="x"),
        Question(id="2", text="b", answer="y"),
    ]

    tex = render_questions(questions)

    assert tex == (
        "\\begin{enumerate}\n"
        "    \\item \\Question{1}{a}\n"
        "    \\item \\Question{2}{b}\n"
        "\\end{enumerate}"
    )


def test_estimate_question_width_counts_visible_characters():
    assert estimate_question_width("$7+5=$") == len("7+5=")


def test_estimate_question_width_collapses_frac_to_widest_part():
    # \frac{1}{4} -> numerator "1" (len 1), denominator "4" (len 1) -> "X"
    assert estimate_question_width(r"$\frac{1}{4}+\frac{1}{2}=$") == len("X+X=")
    assert estimate_question_width(r"$\frac{100}{4}=$") == len("XXX=")


def test_estimate_question_width_strips_control_words():
    assert estimate_question_width(r"$\sqrt{9}=$") == len("9=")


def test_choose_grid_columns_falls_back_to_one_below_minimum_count():
    questions = [Question(id=str(i), text="$2+2=$", answer="4") for i in range(3)]

    assert choose_grid_columns(questions) == 1


def test_choose_grid_columns_uses_multiple_columns_for_uniform_short_questions():
    questions = [Question(id=str(i), text="$2+2=$", answer="4") for i in range(10)]

    assert choose_grid_columns(questions) > 1


def test_choose_grid_columns_falls_back_to_one_when_widths_vary_too_much():
    questions = [Question(id=str(i), text="$2+2=$", answer="4") for i in range(9)]
    questions.append(
        Question(
            id="9",
            text="$x^2 - 5x + 6 = 0 \\text{, solve for } x \\text{ using the quadratic formula}$",
            answer="2, 3",
        )
    )

    assert choose_grid_columns(questions) == 1


def test_render_questions_lays_out_uniform_questions_in_a_grid():
    questions = [Question(id=str(i), text="$2+2=$", answer="4") for i in range(10)]

    tex = render_questions(questions)

    assert tex.startswith(r"\begin{tabular}{@{}")
    assert tex.endswith(r"\end{tabular}")
    assert tex.count(r"\Question{") == 10
    assert " & " in tex


def _template(tmp_path: Path) -> Path:
    template_path = tmp_path / "template.tex"
    template_path.write_text("HEADER\n%%QUESTIONS%%\nFOOTER\n")
    return template_path


def _questions_client() -> MagicMock:
    return _fake_client(json.dumps([{"id": "1", "text": "1+1=?", "answer": "2"}]))


def _client_with_responses(*raw_texts: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = [
        SimpleNamespace(content=[SimpleNamespace(type="text", text=text)]) for text in raw_texts
    ]
    return client


def test_generate_worksheet_writes_tex_and_returns_no_record_without_bucket(tmp_path):
    client = _questions_client()
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"

    with patch("graderbot.worksheetbot.compile_tex", return_value=(True, "")):
        tex_path, questions, record = generate_worksheet(
            client, template_path, "arithmetic", out, num_questions=1, max_repairs=3
        )

    assert tex_path == out.with_suffix(".tex")
    assert tex_path.exists()
    assert "%%QUESTIONS%%" not in tex_path.read_text()
    assert questions == [Question(id="1", text="1+1=?", answer="2")]
    assert record is None


def test_generate_worksheet_stores_when_bucket_given(tmp_path):
    client = _client_with_responses(
        json.dumps([{"id": "1", "text": "1+1=?", "answer": "2"}]), "Auto Title"
    )
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"
    db_path = tmp_path / "worksheets.sqlite3"
    fake_record = SimpleNamespace(
        id=1,
        student_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/student.pdf",
        cv_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/cv.pdf",
        answers_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/answers.pdf",
    )

    with patch("graderbot.worksheetbot.compile_tex", return_value=(True, "")), patch(
        "graderbot.worksheetbot.storage.store_worksheet", return_value=fake_record
    ) as mock_store:
        tex_path, questions, record = generate_worksheet(
            client,
            template_path,
            "arithmetic",
            out,
            num_questions=1,
            max_repairs=3,
            bucket="my-bucket",
            db_path=db_path,
        )

    mock_store.assert_called_once_with(
        tex_path=tex_path,
        questions=questions,
        prompt="arithmetic",
        model=MODEL,
        bucket="my-bucket",
        db_path=db_path,
        title="Auto Title",
        public_id=ANY,
    )
    assert record is fake_record


def test_generate_worksheet_uses_explicit_title_without_generating_one(tmp_path):
    client = _questions_client()
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"
    db_path = tmp_path / "worksheets.sqlite3"
    fake_record = SimpleNamespace(
        id=1,
        student_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/Given_Title_student.pdf",
        cv_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/Given_Title_cv.pdf",
        answers_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/Given_Title_answers.pdf",
    )

    with patch("graderbot.worksheetbot.compile_tex", return_value=(True, "")), patch(
        "graderbot.worksheetbot.storage.store_worksheet", return_value=fake_record
    ) as mock_store, patch("graderbot.worksheetbot.generate_title") as mock_generate_title:
        generate_worksheet(
            client,
            template_path,
            "arithmetic",
            out,
            num_questions=1,
            max_repairs=3,
            bucket="my-bucket",
            db_path=db_path,
            title="Given Title",
        )

    mock_generate_title.assert_not_called()
    mock_store.assert_called_once_with(
        tex_path=Path(out).with_suffix(".tex"),
        questions=[Question(id="1", text="1+1=?", answer="2")],
        prompt="arithmetic",
        model=MODEL,
        bucket="my-bucket",
        db_path=db_path,
        title="Given Title",
        public_id=ANY,
    )


def test_generate_worksheet_repairs_and_succeeds_on_retry(tmp_path):
    client = _questions_client()
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"

    with patch(
        "graderbot.worksheetbot.compile_tex", side_effect=[(False, "log tail"), (True, "")]
    ), patch("graderbot.worksheetbot.repair_tex", return_value="FIXED SOURCE") as mock_repair:
        tex_path, questions, record = generate_worksheet(
            client, template_path, "arithmetic", out, num_questions=1, max_repairs=3
        )

    mock_repair.assert_called_once()
    assert tex_path.read_text() == "FIXED SOURCE"
    assert record is None


def test_generate_worksheet_raises_compile_error_after_max_repairs(tmp_path):
    client = _questions_client()
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"

    with patch(
        "graderbot.worksheetbot.compile_tex", return_value=(False, "persistent failure")
    ), patch("graderbot.worksheetbot.repair_tex", return_value="STILL BROKEN") as mock_repair:
        with pytest.raises(CompileError) as exc_info:
            generate_worksheet(
                client, template_path, "arithmetic", out, num_questions=1, max_repairs=1
            )

    assert exc_info.value.log_tail == "persistent failure"
    assert mock_repair.call_count == 1


# --------------------------------------------------------------------------
# Manual entry: parse_questions_json / create_worksheet_from_questions
# --------------------------------------------------------------------------

def test_parse_questions_json_parses_plain_array():
    raw = json.dumps(
        [
            {"id": "1", "text": "2 + 2 = ?", "answer": "4"},
            {"id": "2", "text": "3 * 3 = ?", "answer": "9"},
        ]
    )

    assert parse_questions_json(raw) == [
        Question(id="1", text="2 + 2 = ?", answer="4"),
        Question(id="2", text="3 * 3 = ?", answer="9"),
    ]


def test_parse_questions_json_coerces_non_string_id():
    raw = json.dumps([{"id": 7, "text": "q", "answer": "a"}])

    assert parse_questions_json(raw) == [Question(id="7", text="q", answer="a")]


def test_parse_questions_json_tolerates_unescaped_latex_backslashes():
    # A human pasting LaTeX into JSON hits the same single-backslash pitfall as
    # the LLM; _parse_json_array's repair path should recover it.
    raw = r'[{"id": "1", "text": "$8 \div 4 = ?$", "answer": "2"}]'

    assert parse_questions_json(raw) == [Question(id="1", text=r"$8 \div 4 = ?$", answer="2")]


def test_parse_questions_json_strips_markdown_fences():
    raw = "```json\n" + json.dumps([{"id": "1", "text": "q", "answer": "a"}]) + "\n```"

    assert parse_questions_json(raw) == [Question(id="1", text="q", answer="a")]


@pytest.mark.parametrize(
    "raw",
    [
        "[]",  # empty array
        json.dumps({"id": "1", "text": "q", "answer": "a"}),  # object, not array
        json.dumps([{"id": "1", "text": "q"}]),  # missing "answer"
        json.dumps(["not an object"]),  # array item is not an object
    ],
)
def test_parse_questions_json_rejects_invalid_payloads(raw):
    with pytest.raises(ValueError):
        parse_questions_json(raw)


def test_create_worksheet_from_questions_stores_with_manual_model(tmp_path):
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"
    db_path = tmp_path / "worksheets.sqlite3"
    raw = json.dumps([{"id": "1", "text": "$2+2=$", "answer": "4"}])
    fake_record = SimpleNamespace(
        id=1,
        student_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/student.pdf",
        cv_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/cv.pdf",
        answers_pdf_s3url="https://my-bucket.s3.amazonaws.com/worksheet/answers.pdf",
    )

    with patch("graderbot.worksheetbot.compile_tex", return_value=(True, "")), patch(
        "graderbot.worksheetbot.storage.store_worksheet", return_value=fake_record
    ) as mock_store:
        tex_path, questions, record = create_worksheet_from_questions(
            raw, template_path, out, title="My Sheet", bucket="my-bucket", db_path=db_path
        )

    assert tex_path == out.with_suffix(".tex")
    assert tex_path.exists()
    assert "%%QUESTIONS%%" not in tex_path.read_text()
    assert questions == [Question(id="1", text="$2+2=$", answer="4")]
    assert record is fake_record
    mock_store.assert_called_once_with(
        tex_path=tex_path,
        questions=questions,
        prompt="",
        model="manual",
        bucket="my-bucket",
        db_path=db_path,
        title="My Sheet",
        public_id=ANY,
    )


def test_create_worksheet_from_questions_does_not_repair_on_compile_failure(tmp_path):
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"
    raw = json.dumps([{"id": "1", "text": "$2+2=$", "answer": "4"}])

    with patch(
        "graderbot.worksheetbot.compile_tex", return_value=(False, "bad latex")
    ), patch("graderbot.worksheetbot.repair_tex") as mock_repair:
        with pytest.raises(CompileError) as exc_info:
            create_worksheet_from_questions(raw, template_path, out, title="My Sheet")

    assert exc_info.value.log_tail == "bad latex"
    mock_repair.assert_not_called()
