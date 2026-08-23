"""Tests for generating a printable roster of name-collection worksheets (#45).

`generate_name_worksheets` turns a class roster (list of names) into a single
PDF with one name-collection worksheet per student, ready to print.
"""

import shutil
from pathlib import Path

import fitz
import pytest

from graderbot.name_worksheets import generate_name_worksheets, parse_roster


def test_parse_roster_splits_and_strips():
    text = "  John Doe \n\nChristina Kim\n   \nMike Meyers\n"
    assert parse_roster(text) == ["John Doe", "Christina Kim", "Mike Meyers"]


def test_generate_empty_roster_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_name_worksheets([], tmp_path / "out.pdf")
    with pytest.raises(ValueError):
        generate_name_worksheets(["  ", "\n"], tmp_path / "out.pdf")


@pytest.mark.slow
def test_one_sheet_per_name_with_printed_name(tmp_path):
    """The merged PDF is one worksheet per roster entry (the #41 template spans
    a fixed number of pages each), and every student's name is printed in the
    document so they know which name to copy."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    one = generate_name_worksheets(["Jane Doe"], tmp_path / "one.pdf")
    with fitz.open(one) as doc:
        pages_per_sheet = doc.page_count
    # Each student's sheet must fit on a single printed page (issue #45).
    assert pages_per_sheet == 1

    names = ["John Doe", "Christina Kim", "Mike Meyers"]
    out = generate_name_worksheets(names, tmp_path / "names.pdf")

    assert out.exists()
    with fitz.open(out) as doc:
        assert doc.page_count == pages_per_sheet * len(names)
        full_text = "".join(page.get_text() for page in doc)
    for name in names:
        assert name in full_text


@pytest.mark.slow
def test_name_with_latex_special_chars(tmp_path):
    """A roster name with LaTeX-special characters is escaped, not compiled
    verbatim, so it renders a sheet instead of crashing latexmk."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    out = generate_name_worksheets(["A & B_C #1"], tmp_path / "special.pdf")
    with fitz.open(out) as doc:
        assert doc.page_count >= 1
