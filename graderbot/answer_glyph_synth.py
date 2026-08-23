"""Fast synthetic renderer for single answer-box crops (issue #81), used to
generate bulk training data for `response_scorer.CnnResponseScorer` without
`worksheet_synth.fill_worksheet`'s cost -- that path compiles a full LaTeX
worksheet through latexmk and rasterizes an entire page per sample, far too
slow to call thousands of times for training. This renders directly onto a
small canvas sized like a real cropped answer box, reusing
`worksheet_synth`'s own centering/skew/noise helpers so a synthetic crop is
produced by (mostly) the same pipeline as a real one, rather than a
parallel implementation that could drift from it.

Scope matches `response_candidates`: plain numeric text only (digits, `.`,
`-`); no fraction rendering here (see that module's docstring).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from graderbot.worksheet_synth import (
    _draw_text_centered_in,
    add_image_noise,
    perspective_skew_image,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Only one handwriting-style font ships in the repo today
# (`worksheet_synth._DEFAULT_FONT`); training on a single font risks the
# model learning that font's quirks instead of handwriting in general, so
# this is deliberately a tuple that more fonts can be appended to later
# (issue #81's "Data pipeline" section flags this as a known follow-up, not
# attempted here since sourcing/licensing new handwriting fonts needs the
# user's say-so).
DEFAULT_FONTS: Tuple[str, ...] = (str(_REPO_ROOT / "fonts" / "HomemadeApple-Regular.ttf"),)

# (width, height) px -- roughly the aspect ratio of a real rectified answer
# box crop (`imaging.crop_box_content_aware`'s output), not any particular
# worksheet's exact box size, since the model must generalize across boxes.
DEFAULT_CANVAS_SIZE: Tuple[int, int] = (240, 100)
_DEFAULT_TEXT_SIZE = 48
_TEXT_SIZE_JITTER = 8


def render_answer_glyph(
    text: str,
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
    font: Optional[str] = None,
    text_size: int = _DEFAULT_TEXT_SIZE,
    rng: Optional[np.random.Generator] = None,
    jitter: bool = True,
) -> np.ndarray:
    """Renders `text` (a plain numeric string) centered on a blank white
    canvas sized `canvas_size` = (width, height), in `font` (defaults to
    `DEFAULT_FONTS[0]`). When `jitter` is True (the default), layers the
    same `perspective_skew_image`/`add_image_noise` `worksheet_synth` uses
    to simulate a photographed/scanned page. Returns a BGR numpy array
    (`worksheet_synth.write_on_image`'s convention) the same shape a real
    crop would have if it happened to be exactly `canvas_size`."""
    rng = rng if rng is not None else np.random.default_rng()
    font = font or DEFAULT_FONTS[0]
    width, height = canvas_size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image = _draw_text_centered_in(image, text, (0, 0, width, height), font, text_size)
    if jitter:
        image = perspective_skew_image(image, max_skew=0.03, rng=rng)
        image = add_image_noise(image, noise_level=0.04, rng=rng)
    return image


def generate_training_sample(
    text: str,
    rng: Optional[np.random.Generator] = None,
    fonts: Tuple[str, ...] = DEFAULT_FONTS,
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> np.ndarray:
    """One randomly-jittered rendering of `text` for training:
    font/text-size/skew/noise all vary per call (driven by `rng`) so
    repeated calls for the same `text` don't yield visually-identical
    examples, the way a fixed-parameter `render_answer_glyph` call would."""
    rng = rng if rng is not None else np.random.default_rng()
    font = fonts[int(rng.integers(0, len(fonts)))]
    text_size = int(rng.integers(_DEFAULT_TEXT_SIZE - _TEXT_SIZE_JITTER, _DEFAULT_TEXT_SIZE + _TEXT_SIZE_JITTER + 1))
    return render_answer_glyph(text, canvas_size, font, text_size, rng, jitter=True)
