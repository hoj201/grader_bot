import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from worksheetbot import Question, escape_latex, generate_questions, render_questions


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


def test_escape_latex_escapes_special_characters():
    assert escape_latex(r"50% & $5 #1 {a}_b ~x^y \z") == (
        r"50\% \& \$5 \#1 \{a\}\_b \textasciitilde{}x\textasciicircum{}y "
        r"\textbackslash{}z"
    )


def test_render_questions_uses_question_macro_with_id_and_text():
    questions = [Question(id="add001", text="$7+5=$", answer="12")]

    tex = render_questions(questions)

    assert tex == r"\Question{add001}{$7+5=$}"


def test_render_questions_does_not_escape_question_text():
    questions = [Question(id="frac001", text=r"$\frac{1}{4}+\frac{1}{2}=$", answer=r"\frac{3}{4}")]

    tex = render_questions(questions)

    assert tex == r"\Question{frac001}{$\frac{1}{4}+\frac{1}{2}=$}"


def test_render_questions_escapes_special_characters_in_id():
    questions = [Question(id="q_1", text="text", answer="1")]

    tex = render_questions(questions)

    assert tex == r"\Question{q\_1}{text}"


def test_render_questions_joins_multiple_questions_with_newlines():
    questions = [
        Question(id="1", text="a", answer="x"),
        Question(id="2", text="b", answer="y"),
    ]

    tex = render_questions(questions)

    assert tex == "\\Question{1}{a}\n\\Question{2}{b}"
