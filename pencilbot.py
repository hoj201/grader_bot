from typing import Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class Box:
    x_lower_left: float # relative coordinate between 0 and 1
    y_lower_left: float # relative coordinate between 0 and 1
    width: float
    height: float


def extract_answer_boxes(worksheet_filename: str) -> Dict[str, Box]:
    """Takes the filename of a pdf and finds all the red outlined rectangles in the file.
    Inside each rectangle is text corresponding to an id, which is also read.
    The returned object is a dictionary that maps each id to it's corresponding box.
    """
    pass