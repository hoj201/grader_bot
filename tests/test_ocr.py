"""Tests for the name-box OCR helpers (issue #58).

These monkeypatch the Tesseract call so they run without the binary installed;
the end-to-end read of a real rendered worksheet lives in test_graderbot.py.
"""

import numpy as np
import pytest

from graderbot import ocr
from graderbot.models import Box
from graderbot.ocr import extract_name, extract_name_scored

FULL_BOX = Box(x_lower_left=0.0, y_lower_left=0.0, width=1.0, height=1.0)
ROSTER = ["Jane Doe", "John Smith", "Alice Johnson"]


@pytest.fixture
def blank_page():
    return np.full((120, 400, 3), 255, np.uint8)


def test_extract_name_scored_returns_match_and_similarity(monkeypatch, blank_page):
    # A typical Tesseract misread of handwriting: a couple of letters wrong.
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jane Doo")

    name, score = extract_name_scored(blank_page, FULL_BOX, ROSTER)

    assert name == "Jane Doe"
    assert 0.0 < score < 1.0


def test_extract_name_scored_gives_a_clean_read_full_confidence(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jane Doe")

    assert extract_name_scored(blank_page, FULL_BOX, ROSTER) == ("Jane Doe", 1.0)


def test_extract_name_scored_returns_empty_when_nothing_is_close(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "zzzzzzzzzz")

    assert extract_name_scored(blank_page, FULL_BOX, ROSTER) == ("", 0.0)


def test_extract_name_scored_ranks_a_cleaner_read_higher(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jane Doo")
    _, close = extract_name_scored(blank_page, FULL_BOX, ROSTER)
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jarie Dae")
    _, far = extract_name_scored(blank_page, FULL_BOX, ROSTER)

    assert close > far


def test_extract_name_returns_just_the_name(monkeypatch, blank_page):
    """The scoreless wrapper still behaves exactly as before."""
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Jhn Smith")

    assert extract_name(blank_page, FULL_BOX, ROSTER) == "John Smith"


def test_extract_name_returns_empty_string_on_no_match(monkeypatch, blank_page):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "zzzzzzzzzz")

    assert extract_name(blank_page, FULL_BOX, ROSTER) == ""
