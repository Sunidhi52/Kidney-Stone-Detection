import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read the image at path: {image_path}")
        return

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to identify bright regions
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    min_area = 50
    max_area = 1000
    potential_stones = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Draw contours on the original image
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, potential_stones, -1, (0, 255, 0), 2) 
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Analyzed Image'), plt.xticks([]), plt.yticks([])

    if potential_stones:
        plt.suptitle(f"Potential high-intensity regions detected: {len(potential_stones)}", color='red', fontsize=16)
        print(f"Detected {len(potential_stones)} potential high-intensity regions.")
    else:
        plt.suptitle("No significant high-intensity regions detected", color='green', fontsize=16)
        print("No significant high-intensity regions detected.")

    plt.show()

    print("\nIMPORTANT: This is a demonstration only and not a medical diagnostic tool.")
    print("Accurate kidney stone detection requires professional medical imaging and interpretation.")

# Example usage

image_path = input("Enter the path to the image file: ")
analyze_image(image_path)