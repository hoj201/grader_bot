from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Box:
    x_lower_left: float # relative coordinate between 0 and 1
    y_lower_left: float # relative coordinate between 0 and 1
    width: float
    height: float


@dataclass(frozen=True)
class QuestionResult:
    answer: str    # the stored correct answer (LaTeX)
    response: str  # what the student wrote, as OCR'd (LaTeX; "" if blank)
    correct: bool
    blank: bool = False  # True if the box was skipped as unanswered (issue #66); markup draws nothing for it
    open_ended: bool = False  # True if the question has no single correct answer (issue #65); never graded right/wrong, and markup draws nothing for it
    # Debugging aid for OCR misreads (issue #70): Mathpix's self-reported
    # confidence for `response` and its raw pre-repair text (see
    # ocr.OcrResult). Both None when the box was never sent to Mathpix
    # (blank box, issue #66).
    ocr_confidence: Optional[float] = None
    ocr_raw: Optional[str] = None
