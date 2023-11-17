import cv2
import numpy as np
import sys


if len(sys.argv) != 3:
    print("Usage: python findpokemon.py <input_image_path> <output_file_name>")
    sys.exit(1)

image_path = sys.argv[1]
output_file_name = sys.argv[2]
image = cv2.imread(image_path)

# keep the right side of the image
height, width = image.shape[:2]
percentage_to_keep = 0.5
width_to_keep = int(width * percentage_to_keep)
right_side = image[:, width - width_to_keep:]
image = right_side

# find shapes on the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find the most complex shape on the image
most_complex_shape = None
max_complexity_contour = None
max_complexity = 0

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    complexity = len(approx)
    if complexity > max_complexity:
        max_complexity = complexity
        most_complex_shape = approx
        max_complexity_contour = contour

#cut the image to include at least one third of its size being the center the most complex shape
x, y, w, h = cv2.boundingRect(max_complexity_contour)
min_size = max(image.shape[0], image.shape[1]) // 3
if w < min_size or h < min_size:
    padding = (min_size - w) // 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, min_size)
    h = min(image.shape[0] - y, min_size)
result = image[y:y + h, x:x + w]

cv2.imwrite(output_file_name, result)