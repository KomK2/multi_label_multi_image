import cv2
import numpy as np

# Load the image
image_path = '/home/kiran/projects/github/test/all-shapes-and-colors/dataset/train_dataset/img_46.png'
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
detected_shapes = []


# Define color ranges for red, blue, and green in HSV
color_ranges = {
    'red': ((0, 100, 100), (10, 255, 255)),
    'blue': ((100, 150, 0), (140, 255, 255)), 
    'green': ((40, 70, 70), (80, 255, 255))
}

def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    sides = len(approx)
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        # Distinguish between square and rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 < aspect_ratio < 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    else:
        return "Circle"


# Function to find contours and draw bounding boxes
def find_bounding_boxes(mask, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            shape = detect_shape(contour)
            x, y, w, h = cv2.boundingRect(contour)
            detected_shapes.append((shape.lower(), color_name.lower()))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            label = f"{color_name} {shape}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


# Process each color
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, lower, upper)
    find_bounding_boxes(mask, color_name)

print(detected_shapes)

# Save the result
output_path = '/home/kiran/projects/github/test/output_img_0.png'
cv2.imwrite(output_path, image)

# Display the result
cv2.imshow('Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()