import cv2
import numpy as np
import sys
import os


def find_pokemon(input_image,input_edges,out_edges,out_image):
    image_path = input_image
    output_file_name = out_image
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
    cv2.imwrite(input_edges, edges)

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
    min_size = max(image.shape[0], image.shape[1]) // 2
    if w < min_size or h < min_size:
        padding = (min_size - w) // 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, min_size)
        h = min(image.shape[0] - y, min_size)

    result_edges = edges[y:y + h, x:x + w]
    result = image[y:y + h, x:x + w]
    cv2.imwrite(out_edges, result_edges)
    cv2.imwrite(output_file_name, result)



if len(sys.argv) != 2:
    print("Usage: python findpokemon.py <dataset_path>")
    sys.exit(1)

folder_path = sys.argv[1] #"./ScreenShoots/images"


# Get the list of files and subdirectories in the specified folder
contents = os.listdir(folder_path)

for item in contents:
    image_path = os.path.join(folder_path, item)
    output_file_name = f'./ScreenShoots/output/{item}'
    input_edges = f'./ScreenShoots/input_edges/{item}'
    output_edges = f'./ScreenShoots/output_edges/{item}'
    find_pokemon(image_path,input_edges,output_edges,output_file_name)