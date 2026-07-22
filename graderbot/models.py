from dataclasses import dataclass


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
