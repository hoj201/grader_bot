import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from graderbot.core import read_worksheet_id, render_pdf_page_image
from graderbot.worksheet_qr import (
    decode_worksheet_id,
    generate_worksheet_id,
    render_qr_png,
)
from graderbot.worksheet_synth import latexmk_worksheet
from graderbot.worksheetbot import fill_template

TEMPLATE_TEX = Path(__file__).parent.parent / "tex" / "worksheet_template.tex"


def test_generate_worksheet_id_is_short_and_alphanumeric():
    worksheet_id = generate_worksheet_id()
    assert isinstance(worksheet_id, str)
    assert re.fullmatch(r"[A-Za-z0-9_]+", worksheet_id)
    assert 0 < len(worksheet_id) <= 32


def test_generate_worksheet_id_is_unique():
    ids = {generate_worksheet_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_render_qr_png_creates_file(tmp_path):
    out_path = tmp_path / "qr.png"
    returned = render_qr_png("ws_abc123", out_path)
    assert returned == out_path
    assert out_path.exists()
    image = cv2.imread(str(out_path))
    assert image is not None
    assert image.shape[0] > 0 and image.shape[1] > 0


def test_round_trip_encode_decode(tmp_path):
    worksheet_id = generate_worksheet_id()
    out_path = tmp_path / "qr.png"
    render_qr_png(worksheet_id, out_path)
    image = cv2.imread(str(out_path))
    assert decode_worksheet_id(image) == worksheet_id


def test_decode_returns_none_on_blank_image():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert decode_worksheet_id(blank) is None


def test_embedded_qr_is_decodable_after_latex_render(tmp_path):
    """End-to-end: a worksheet compiled with an embedded id has a QR code that
    survives a real LaTeX render and rasterization, and decodes back."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")

    worksheet_id = generate_worksheet_id()
    tex_source = fill_template(TEMPLATE_TEX, "", worksheet_id=worksheet_id)
    tex_path = tmp_path / "worksheet.tex"
    tex_path.write_text(tex_source)
    render_qr_png(worksheet_id, tmp_path / f"qr_{worksheet_id}.png")

    pdf_path = latexmk_worksheet(str(tex_path), cv_mode=True)
    image = render_pdf_page_image(pdf_path)

    assert read_worksheet_id(image) == worksheet_id
