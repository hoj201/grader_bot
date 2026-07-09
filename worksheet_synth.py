"""A python module for synthesizing semi-realistic images of worksheets
from worksheet.sty with hand-writing scrawled on top.

The main impetus for this module is the creation of unit-tests
for pencilbot.py
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DEFAULT_FONT = str(Path(__file__).parent / "fonts" / "HomemadeApple-Regular.ttf")
_DEFAULT_TEXT_SIZE = 36
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")


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


def perspective_skew_image(
    image: np.ndarray,
    max_skew: float = 0.02,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Applies a minor random 4-corner perspective warp to a copy of
    `image`, simulating a photo taken at a slight angle. Each corner is
    nudged by up to `max_skew` (a fraction of image width/height) in a
    random direction. Exposed border areas are filled white. `image` is
    left unmodified."""
    if rng is None:
        rng = np.random.default_rng()

    height, width = image.shape[:2]
    src_corners = np.array(
        [[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32
    )
    jitter = rng.uniform(-max_skew, max_skew, size=(4, 2)) * [width, height]
    dst_corners = (src_corners + jitter).astype(np.float32)

    homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
    return cv2.warpPerspective(
        image,
        homography,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def add_image_noise(image):
    """Applies some noise to simulate dust, stains, and mediocre
    lighting on an image"""
    raise NotImplementedError


def _box_to_pixels(box, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """Converts a `pencilbot.Box` (relative coordinates, origin at
    bottom-left) into pixel (x0, y0, x1, y1) with origin at top-left."""
    x0 = box.x_lower_left * image_width
    x1 = (box.x_lower_left + box.width) * image_width
    y1 = (1 - box.y_lower_left) * image_height
    y0 = y1 - box.height * image_height
    return int(x0), int(y0), int(x1), int(y1)


def _text_metrics(text: str, font: str, text_size: int) -> Tuple[int, int, int, int]:
    """Returns (width, height, left, top) of `text`'s tight glyph
    bounding box, as reported by `ImageFont.getbbox`."""
    pil_font = ImageFont.truetype(font, text_size)
    left, top, right, bottom = pil_font.getbbox(text)
    return right - left, bottom - top, left, top


def _draw_text_centered_in(
    image: np.ndarray,
    text: str,
    region: Tuple[int, int, int, int],
    font: str,
    text_size: int,
) -> np.ndarray:
    """Draws `text` so its tight glyph bounding box is centered within
    pixel `region` = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = region
    width, height, left, top = _text_metrics(text, font, text_size)
    target_x = x0 + max(((x1 - x0) - width) // 2, 0)
    target_y = y0 + max(((y1 - y0) - height) // 2, 0)
    return write_on_image(image, text, (target_x - left, target_y - top), text_size, font)


def _draw_answer(
    image: np.ndarray,
    box_px: Tuple[int, int, int, int],
    answer: str,
    font: str,
    text_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = box_px

    frac_match = _FRAC_RE.fullmatch(answer.strip())
    if frac_match:
        numerator, denominator = frac_match.groups()
        mid_y = y0 + (y1 - y0) // 2

        image = _draw_text_centered_in(image, numerator, (x0, y0, x1, mid_y), font, text_size)
        image = _draw_text_centered_in(image, denominator, (x0, mid_y, x1, y1), font, text_size)

        num_w, _, _, _ = _text_metrics(numerator, font, text_size)
        den_w, _, _, _ = _text_metrics(denominator, font, text_size)
        line_width = max(num_w, den_w)
        line_x0 = x0 + ((x1 - x0) - line_width) // 2
        line_x1 = line_x0 + line_width
        image = image.copy()
        cv2.line(image, (line_x0, mid_y), (line_x1, mid_y), (0, 0, 0), 2)
        return image

    return _draw_text_centered_in(image, answer, (x0, y0, x1, y1), font, text_size)


def fill_worksheet(
    tex_fn: str,
    answers: Dict[str, str],
    font: str = _DEFAULT_FONT,
    text_size: int = _DEFAULT_TEXT_SIZE,
) -> np.ndarray:
    """Renders the blank version of `tex_fn` and writes `answers` (a
    dictionary mapping question ids to short LaTeX snippets, either a
    plain number or a `\\frac{a}{b}`) into their corresponding answer
    boxes, using the CV-mode render of the same worksheet to locate
    those boxes. Returns the composited worksheet as a BGR numpy array.
    """
    from pencilbot import extract_answer_boxes, render_pdf_page_image

    cv_worksheet = latexmk_worksheet(tex_fn, cv_mode=True)
    blank_worksheet = latexmk_worksheet(tex_fn, cv_mode=False)
    boxes = extract_answer_boxes(cv_worksheet)

    rgb_image = render_pdf_page_image(blank_worksheet)
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    for qid, answer in answers.items():
        if qid not in boxes:
            raise KeyError(f"No answer box found for question id {qid!r}")
        box_px = _box_to_pixels(boxes[qid], image.shape[1], image.shape[0])
        image = _draw_answer(image, box_px, answer, font, text_size)

    return image
