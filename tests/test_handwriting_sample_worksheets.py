"""Tests for issue #81's "copy worksheet" generator. Compilation itself is
mocked (see `_template`/`compile_tex` patching, mirroring
test_worksheetbot.py's `test_create_worksheet_from_questions_stores_with_manual_model`)
so these don't need a LaTeX toolchain -- the new logic under test is
question generation, not the shared `build_worksheet` plumbing."""

import random
from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from graderbot.handwriting_sample_worksheets import (
    build_handwriting_sample_worksheet,
    generate_copy_questions,
)
from graderbot.response_candidates import SAMPLE_ANSWER_POOL
from graderbot.worksheetbot import Question


def test_generate_copy_questions_returns_the_requested_count():
    questions = generate_copy_questions(5, rng=random.Random(0))
    assert len(questions) == 5


def test_generate_copy_questions_has_unique_ids():
    questions = generate_copy_questions(10, rng=random.Random(0))
    assert len({q.id for q in questions}) == 10


def test_generate_copy_questions_answer_matches_the_printed_text():
    # The whole point (issue #81): the printed target and the stored answer
    # must be the identical string, so a harvested crop's label is trustworthy.
    for q in generate_copy_questions(20, rng=random.Random(0)):
        assert q.answer in q.text
        assert q.open_ended is False


def test_generate_copy_questions_draws_from_the_sample_answer_pool():
    questions = generate_copy_questions(20, rng=random.Random(0))
    assert all(q.answer in SAMPLE_ANSWER_POOL for q in questions)


def test_generate_copy_questions_is_reproducible_given_the_same_rng_seed():
    a = generate_copy_questions(15, rng=random.Random(42))
    b = generate_copy_questions(15, rng=random.Random(42))
    assert [q.answer for q in a] == [q.answer for q in b]


def test_generate_copy_questions_repeats_once_the_pool_is_exhausted():
    count = len(SAMPLE_ANSWER_POOL) + 3
    questions = generate_copy_questions(count, rng=random.Random(0))
    assert len(questions) == count


def test_generate_copy_questions_rejects_a_non_positive_count():
    with pytest.raises(ValueError):
        generate_copy_questions(0)


def _template(tmp_path: Path) -> Path:
    template_path = tmp_path / "template.tex"
    template_path.write_text("HEADER\n%%QUESTIONS%%\nFOOTER\n")
    return template_path


def test_build_handwriting_sample_worksheet_stores_with_the_handwriting_sample_model(tmp_path):
    template_path = _template(tmp_path)
    out = tmp_path / "worksheet"
    db_path = tmp_path / "worksheets.sqlite3"

    with patch("graderbot.worksheetbot.compile_tex", return_value=(True, "")), patch(
        "graderbot.worksheetbot.storage.store_worksheet"
    ) as mock_store:
        tex_path, questions, record = build_handwriting_sample_worksheet(
            count=4,
            template_path=template_path,
            out=out,
            bucket="my-bucket",
            db_path=db_path,
            rng=random.Random(0),
        )

    assert tex_path == out.with_suffix(".tex")
    assert len(questions) == 4
    assert all(isinstance(q, Question) for q in questions)
    mock_store.assert_called_once_with(
        tex_path=tex_path,
        questions=questions,
        prompt="",
        model="handwriting_sample",
        bucket="my-bucket",
        db_path=db_path,
        title="Handwriting Practice",
        header=ANY,
        public_id=ANY,
    )
