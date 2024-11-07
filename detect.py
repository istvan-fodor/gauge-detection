import cv2
import numpy as np

def detect_gauges(image_path, output_folder, min_dist_ratio=0.1, min_radius_ratio=0.05, max_radius_ratio=0.2):
    # Load the image and get dimensions
    image_original = cv2.imread(image_path)

    height, width = image_original.shape[:2]
    
    # Calculate minDist, minRadius, and maxRadius based on the width of the image
    minDist = int(width * min_dist_ratio)
    minRadius = int(width * min_radius_ratio)
    maxRadius = int(width * max_radius_ratio)

    # Preprocess the image
    image = image_original.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (9, 9), 2)

    # Apply Canny edge detection to emphasize edges
    image = cv2.Canny(image, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Apply a binary threshold to emphasize circular shapes
    # f, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    # print(f"F: {f}")

    # cv2.imshow(f"Thresholded {image_path}", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Resolution ratio
        minDist=minDist,  # Minimum distance between circles
        param1=200,  # Upper threshold for Canny edge detection
        param2=70,  # Accumulator threshold for circle detection
        minRadius=minRadius,  # Minimum radius of circle
        maxRadius=maxRadius   # Maximum radius of circle
    )

    if circles is None:
        print("No gauges found.")
        return
    else:
        print(f"{len(circles[0, :])} circles found.")

    # If circles are detected, process each circle
    if len(circles) > 0:
        file_output_folder = f"{output_folder}/{image_path.replace('/', '_')}"
        os.mkdir(file_output_folder)
        circles = np.round(circles[0, :]).astype("int")
        for idx, (x, y, r) in enumerate(circles):
            if filter_by_color(image_original, x,y,r, color_threshold=150):
                process_circle(image_original, image, idx, x, y, r, file_output_folder)

    else:
        print("No gauges found.")

def filter_by_color(image, x,y,r, color_threshold=150):
    # Define a smaller ROI inside the circle to avoid edges
    roi_radius = int(r * 0.75)  # Use 75% of radius for interior check
    x_start, x_end = max(0, x - roi_radius), min(image.shape[1], x + roi_radius)
    y_start, y_end = max(0, y - roi_radius), min(image.shape[0], y + roi_radius)
    roi = image[y_start:y_end, x_start:x_end]
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (roi_radius, roi_radius), roi_radius, 255, -1)
    # Calculate the average color in the ROI
    avg_color = cv2.mean(roi, mask=mask)[:3]  # Ignore the alpha channel if present
    avg_color = np.array(avg_color)

    # Check if the color is close to white (adjust RGB threshold as needed)
    print(f"Average color: {avg_color}")
    if np.all(avg_color >= color_threshold):
        return True
    else:
        return False

def process_circle(image_original, processed_image, idx, x, y, r, file_output_folder):
    roi_original = image_original[y - r:y + r, x - r:x + r].copy()
    roi = processed_image[y - r:y + r, x - r:x + r].copy()
    # contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #             # Calculate circularity
    #     perimeter = cv2.arcLength(contour, True)
    #     area = cv2.contourArea(contour)
    #      # Display the ROI with contours for visual inspection
    #     roi_original = image[y - r:y + r, x - r:x + r].copy()
    #     cv2.drawContours(roi_original, [contour], -1, (0, 255, 0), 2)

    #     cv2.imshow(f"Gauge {idx} Contours", roi_original)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     if area > 0:
    #         circularity = 4 * np.pi * (area / (perimeter ** 2))
    #         print(f"Gauge {idx} circularity: {circularity}")
    #                 # Check if circularity is close to 1 (perfect circle)
    #         #if circularity < 0.85 or circularity > 1.15:  # Tune this range as needed
    #          #   return

            # Draw circle on the image (optional, for visualization)
    x_start, x_end = max(0, x - r), min(image_original.shape[1], x + r)
    y_start, y_end = max(0, y - r), min(image_original.shape[0], y + r)
    gauge_roi = image_original[y_start:y_end, x_start:x_end].copy()
    cv2.circle(gauge_roi, (int(gauge_roi.shape[1]/2), int(gauge_roi.shape[0]/2)), r, (0, 255, 0), 4)
            
    roi_path = f"{file_output_folder}/gauge_{idx}.png"
    cv2.imwrite(roi_path, gauge_roi)
    print(f"Gauge {idx} saved at: {roi_path}")


import glob
import os

os.makedirs("output", exist_ok=True)
files = glob.glob("input/*.jpg")

for file in files:
    detect_gauges(file, "output", min_dist_ratio=0.2, min_radius_ratio=0.10, max_radius_ratio=0.8)
