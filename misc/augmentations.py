import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Image Sharpening
    sharpening_kernel = np.array([[0, -1, 0], 
                                  [-1, 5, -1], 
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_kernel)

    # Contrast Adjustment using Histogram Equalization
    contrast_adjusted = cv2.equalizeHist(sharpened)

    # Gaussian Filtering (3x3 kernel)
    smoothed = cv2.GaussianBlur(contrast_adjusted, (3, 3), 0)

    # Plot the images for comparison
    # cv2.imshow('Original', img_display)
    # cv2.imshow('Sharpened', smoothed)

    return smoothed
