import cv2
from pathlib import Path

# gbworksheet.sty resolves the markers via TEXINPUTS (which points at <repo>/tex/),
# so write them into <repo>/tex/aruco_images/ regardless of the current directory.
# This script lives at <repo>/scripts/, so the repo root is two levels up.
DIR = Path(__file__).resolve().parent.parent / "tex" / "aruco_images"
DIR.mkdir(parents=True, exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(
cv2.aruco.DICT_5X5_100
)

for marker_id in range(100):
    img = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        1000  # pixels
    )
    cv2.imwrite(str(DIR / f"aruco_{marker_id}.png"), img)

