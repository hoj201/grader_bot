from typing import Dict
from dataclasses import dataclass

import fitz

@dataclass(frozen=True)
class Box:
    x_lower_left: float # relative coordinate between 0 and 1
    y_lower_left: float # relative coordinate between 0 and 1
    width: float
    height: float


_RED = (1, 0, 0)
_RED_TOLERANCE = 0.15


def _is_red(color) -> bool:
    if color is None or len(color) != 3:
        return False
    return all(abs(c - r) <= _RED_TOLERANCE for c, r in zip(color, _RED))


def extract_answer_boxes(worksheet_filename: str) -> Dict[str, Box]:
    """Takes the filename of a pdf and finds all the red outlined rectangles in the file.
    Inside each rectangle is text corresponding to an id, which is also read.
    The returned object is a dictionary that maps each id to it's corresponding box.
    """
    boxes: Dict[str, Box] = {}

    with fitz.open(worksheet_filename) as doc:
        for page in doc:
            page_width, page_height = page.rect.width, page.rect.height

            for drawing in page.get_drawings():
                if not _is_red(drawing.get("color")):
                    continue

                rect = drawing["rect"]

                words = page.get_text("words", clip=rect)
                if not words:
                    continue
                words.sort(key=lambda w: (w[5], w[6], w[7]))
                box_id = "".join(w[4] for w in words)

                boxes[box_id] = Box(
                    x_lower_left=rect.x0 / page_width,
                    y_lower_left=1 - rect.y1 / page_height,
                    width=rect.width / page_width,
                    height=rect.height / page_height,
                )

    return boxes