"""A python module for synthesizing semi-realistic images of worksheets
from worksheet.sty with hand-writing scrawled on top.

The main impetus for this module is the creation of unit-tests
for pencilbot.py
"""

import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def latexmk_worksheet(tex_filename: str, cv_mode: bool) -> str:
    r"""This routine basically just runs the shell commands
    latexmk -pdf -usepretex='\def\WSCVMode{0}' filename.tex
    latexmk -c filename.tex

    then returns the path to the output pdf file.
    """
    tex_path = Path(tex_filename).resolve()
    cv_flag = "1" if cv_mode else "0"
    outdir = tex_path.parent / ("build_cv" if cv_mode else "build_blank")
    outdir.mkdir(exist_ok=True)

    subprocess.run(
        [
            "latexmk",
            "-pdf",
            rf"-usepretex=\def\WSCVMode{{{cv_flag}}}",
            f"-outdir={outdir}",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        check=True,
    )
    subprocess.run(
        ["latexmk", "-c", f"-outdir={outdir}", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )

    return str(outdir / (tex_path.stem + ".pdf"))


def write_on_image(
    image: np.ndarray,
    text: str,
    text_location: Tuple[int, int],
    text_size: int,
    font: str,
) -> np.ndarray:
    """Draws `text` onto a copy of `image` (a BGR numpy array, e.g. from
    cv2.imread) at pixel location `text_location`, using the
    handwriting-style font at path `font` rendered at `text_size`
    points. Returns a new numpy array; `image` is left unmodified."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    pil_font = ImageFont.truetype(font, text_size)
    draw.text(text_location, text, font=pil_font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def perspective_skew_image(image):
    """Takes an image and applies a minor linear warping"""
    raise NotImplementedError


def add_image_noise(image):
    """Applies some noise to simulate dust, stains, and mediocre
    lighting on an image"""
    raise NotImplementedError


def fill_worksheet(tex_fn, answers):
    cv_worksheet = latexmk_worksheet(tex_fn, cv_mode=True)
    blank_workshet = latexmk_worksheet(tex_fn, cv_mode=False)
    from pencilbot import extract_answer_boxes
    boxes = extract_answer_boxes(cv_worksheet)
    """At this point we should be able to use the 
    boxes to know where to write our answers.
    `answers` is a dictionary that maps question ids
    to latex-code.
    """
    raise NotImplementedError
