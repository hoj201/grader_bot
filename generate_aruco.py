import cv2
import os

DIR = "aruco_images"
if DIR not in os.listdir("."):
    os.makedirs(DIR)

dictionary = cv2.aruco.getPredefinedDictionary(
cv2.aruco.DICT_5X5_100
)

for marker_id in range(100):
    img = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        1000  # pixels
    )
    cv2.imwrite(f"{DIR}/aruco_{marker_id}.png", img)

