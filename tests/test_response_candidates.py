import numpy as np

from graderbot.response_candidates import generate_candidates, is_plain_numeric


def test_is_plain_numeric_accepts_integers_decimals_and_negatives():
    assert is_plain_numeric("16")
    assert is_plain_numeric("-16")
    assert is_plain_numeric("3.5")
    assert is_plain_numeric("-0.25")


def test_is_plain_numeric_rejects_fractions_and_garbage():
    assert not is_plain_numeric("\\frac{1}{2}")
    assert not is_plain_numeric("abc")
    assert not is_plain_numeric("")


def test_generate_candidates_always_includes_answer_first():
    candidates = generate_candidates("16")
    assert candidates[0] == "16"
    assert len(candidates) == len(set(candidates))


def test_generate_candidates_passes_through_non_numeric_answers_unchanged():
    assert generate_candidates("\\frac{1}{2}") == ["\\frac{1}{2}"]


def test_generate_candidates_includes_digit_confusion_swaps():
    # "1" is a known confusable for "7" (issue #81's digit-confusion table).
    candidates = generate_candidates("1", max_candidates=10)
    assert "7" in candidates


def test_generate_candidates_includes_sign_flip():
    assert "-16" in generate_candidates("16", max_candidates=10)
    assert "16" in generate_candidates("-16", max_candidates=10)


def test_generate_candidates_includes_decimal_point_variants():
    candidates = generate_candidates("35", max_candidates=10)
    assert "3.5" in candidates


def test_generate_candidates_respects_max_candidates_cap():
    candidates = generate_candidates("123456", max_candidates=4)
    assert len(candidates) == 4
    assert candidates[0] == "123456"


def test_generate_candidates_subsampling_is_reproducible_given_seed():
    a = generate_candidates("123456", rng=np.random.default_rng(0), max_candidates=4)
    b = generate_candidates("123456", rng=np.random.default_rng(0), max_candidates=4)
    assert a == b


def test_generate_candidates_never_duplicates_the_answer_itself():
    candidates = generate_candidates("6", max_candidates=10)
    assert candidates.count("6") == 1
