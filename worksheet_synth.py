"""A python module for synthesizing semi-realistic images of worksheets
from worksheet.sty with hand-writing scrawled on top.

The main impetus for this module is the creation of unit-tests
for pencilbot.py
"""

import subprocess
from pathlib import Path


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


def write_on_image(image, text_location, text_size, font):
    """Takes in an image (perhaps as a fitz.Matrix)
    and then writes on the image using a font that simulates
    human hand-writing"""
    raise NotImplementedError


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
