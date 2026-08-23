import numpy as np

from graderbot.answer_glyph_synth import (
    DEFAULT_CANVAS_SIZE,
    generate_training_sample,
    render_answer_glyph,
)
from graderbot.imaging import ink_fraction


def test_render_answer_glyph_has_the_requested_canvas_shape():
    image = render_answer_glyph("42", jitter=False)
    height, width = DEFAULT_CANVAS_SIZE[1], DEFAULT_CANVAS_SIZE[0]
    assert image.shape == (height, width, 3)


def test_render_answer_glyph_writes_visible_ink():
    image = render_answer_glyph("42", jitter=False)
    assert ink_fraction(image) > 0.0


def test_render_answer_glyph_is_deterministic_given_the_same_rng_seed():
    a = render_answer_glyph("7.5", rng=np.random.default_rng(0))
    b = render_answer_glyph("7.5", rng=np.random.default_rng(0))
    assert np.array_equal(a, b)


def test_render_answer_glyph_jitter_changes_the_image():
    plain = render_answer_glyph("13", jitter=False)
    jittered = render_answer_glyph("13", rng=np.random.default_rng(0), jitter=True)
    assert not np.array_equal(plain, jittered)


def test_generate_training_sample_varies_across_calls():
    a = generate_training_sample("9", rng=np.random.default_rng(1))
    b = generate_training_sample("9", rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_generate_training_sample_is_non_blank():
    image = generate_training_sample("-3", rng=np.random.default_rng(0))
    assert ink_fraction(image) > 0.0
