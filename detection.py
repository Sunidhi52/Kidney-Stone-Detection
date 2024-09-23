import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_kidney_stones(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to read the image at path: {image_path}")
        return
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=30)
    
    # Intensity analysis
    kidney_stones = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Create a mask for the circle
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
            
            # Calculate mean intensity inside the circle
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            
            # Calculate standard deviation of intensity inside the circle
            roi = gray[max(0, i[1]-i[2]):min(gray.shape[0], i[1]+i[2]),
                       max(0, i[0]-i[2]):min(gray.shape[1], i[0]+i[2])]
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask[max(0, i[1]-i[2]):min(gray.shape[0], i[1]+i[2]),
                                                             max(0, i[0]-i[2]):min(gray.shape[1], i[0]+i[2])])
            std_dev = np.std(roi_masked[roi_masked != 0])
            
            # Calculate confidence score based on intensity and std dev
            confidence = (mean_intensity / 255) * (1 - std_dev / 128)
            
            if confidence > 0.6:  # Adjust this threshold as needed
                kidney_stones.append((i[0], i[1], i[2], confidence))
    
    # Draw results
    result = img.copy()
    for stone in kidney_stones:
        cv2.circle(result, (stone[0], stone[1]), stone[2], (0, 255, 0), 2)
        cv2.putText(result, f"{stone[3]:.2f}", (stone[0], stone[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display results
    plt.figure(figsize=(18, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Kidney Stones'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    # Determine if kidney stones are detected
    if len(kidney_stones) > 0:
        print(f"Kidney stones detected: {len(kidney_stones)} potential stones found.")
        for i, stone in enumerate(kidney_stones):
            print(f"Stone {i+1}: Confidence score = {stone[3]:.2f}")
    else:
        print("No kidney stones detected.")

# Example usage
if __name__ == '__main__':
    image_path = input("Enter the path to the JPG image file: ")
    detect_kidney_stones(image_path)