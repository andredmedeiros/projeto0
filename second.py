import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Show results
    titles = ['Original', 'Gray', 'Blurred', 'Edges', 'Threshold']
    images = [img, gray, blurred, edges, thresh]

    plt.figure(figsize=(10, 8))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        if i == 0:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace 'image.jpg' with your image file path
    process_image('image.jpg')