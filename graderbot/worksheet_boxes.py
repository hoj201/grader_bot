from typing import Dict, List

import fitz

from graderbot.models import Box

_RED = (1, 0, 0)
_RED_TOLERANCE = 0.15


def _is_red(color) -> bool:
    if color is None or len(color) != 3:
        return False
    return all(abs(c - r) <= _RED_TOLERANCE for c, r in zip(color, _RED))


def extract_answer_boxes_by_page(worksheet_filename: str) -> List[Dict[str, Box]]:
    """Like `extract_answer_boxes`, but keeps the page each box lives on. Returns
    a list with one `{id: Box}` dict per PDF page (in page order); each `Box`'s
    coordinates are relative to its own page. A page with no answer boxes yields
    an empty dict, so the list length always equals the PDF's page count.

    A multi-page worksheet reuses the same relative coordinate space on every
    page, so a flat id->Box mapping (see `extract_answer_boxes`) cannot tell
    which page a box belongs to. Callers that render or fill per page
    (`fill_worksheet`) need that page association to stamp answers onto the
    right page rather than at the same relative spot on page 1.
    """
    pages: List[Dict[str, Box]] = []

    with fitz.open(worksheet_filename) as doc:
        for page in doc:
            page_boxes: Dict[str, Box] = {}
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

                page_boxes[box_id] = Box(
                    x_lower_left=rect.x0 / page_width,
                    y_lower_left=1 - rect.y1 / page_height,
                    width=rect.width / page_width,
                    height=rect.height / page_height,
                )

            pages.append(page_boxes)

    return pages


def extract_answer_boxes(worksheet_filename: str) -> Dict[str, Box]:
    """Takes the filename of a pdf and finds all the red outlined rectangles in the file.
    Inside each rectangle is text corresponding to an id, which is also read.
    The returned object is a dictionary that maps each id to it's corresponding box.

    Boxes from every page are merged into one flat mapping; use
    `extract_answer_boxes_by_page` when the page association matters.
    """
    boxes: Dict[str, Box] = {}
    for page_boxes in extract_answer_boxes_by_page(worksheet_filename):
        boxes.update(page_boxes)
    return boxes
