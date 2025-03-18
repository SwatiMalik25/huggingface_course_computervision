#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 2: Feature Extraction
================================

This assignment covers traditional feature extraction methods in computer vision.
You'll work with various techniques to extract meaningful features from images,
which is an essential step in many computer vision applications.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from urllib.request import urlretrieve

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

def load_image(image_path):
    """
    Load an image and convert it to grayscale.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: The original image and its grayscale version
    """
    # TODO: Load the image using OpenCV (in BGR format)
    # img = ...
    
    # TODO: Convert the image to RGB format
    # img_rgb = ...
    
    # TODO: Convert the RGB image to grayscale
    # gray = ...
    
    return img_rgb, gray

def detect_edges(gray_image):
    """
    Detect edges in a grayscale image using various edge detection methods.
    
    Args:
        gray_image (numpy.ndarray): The grayscale image
        
    Returns:
        dict: A dictionary containing the results of different edge detection methods
    """
    # TODO: Apply Sobel edge detection
    # sobelx = ...
    # sobely = ...
    # sobel_combined = ...
    
    # TODO: Apply Canny edge detection
    # canny = ...
    
    # TODO: Apply Laplacian edge detection
    # laplacian = ...
    
    # Display the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    # TODO: Display Sobel X result
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display Sobel Y result
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display combined Sobel result
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display Canny result
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display Laplacian result
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/edge_detection.png')
    plt.show()
    
    return {
        'sobel_x': sobelx,
        'sobel_y': sobely,
        'sobel_combined': sobel_combined,
        'canny': canny,
        'laplacian': laplacian
    }

def detect_corners(gray_image, original_image):
    """
    Detect corners in an image using Harris corner detection.
    
    Args:
        gray_image (numpy.ndarray): The grayscale image
        original_image (numpy.ndarray): The original RGB image
        
    Returns:
        numpy.ndarray: Image with detected corners marked
    """
    # TODO: Apply Harris corner detection
    # corners = ...
    
    # TODO: Create a copy of the original image to mark corners
    # image_with_corners = ...
    
    # TODO: Mark the corners on the image
    # ...
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_corners)
    plt.title('Harris Corners')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/corner_detection.png')
    plt.show()
    
    return image_with_corners

def extract_sift_features(gray_image, original_image):
    """
    Extract SIFT features from an image.
    
    Args:
        gray_image (numpy.ndarray): The grayscale image
        original_image (numpy.ndarray): The original RGB image
        
    Returns:
        tuple: Keypoints and descriptors
    """
    # TODO: Initialize the SIFT detector
    # sift = ...
    
    # TODO: Detect keypoints and compute descriptors
    # keypoints, descriptors = ...
    
    # TODO: Draw the keypoints on the original image
    # image_with_keypoints = ...
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_keypoints)
    plt.title('SIFT Features')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/sift_features.png')
    plt.show()
    
    return keypoints, descriptors

def compute_histogram(gray_image):
    """
    Compute and display the histogram of a grayscale image.
    
    Args:
        gray_image (numpy.ndarray): The grayscale image
        
    Returns:
        numpy.ndarray: The histogram values
    """
    # TODO: Compute the histogram
    # hist = ...
    
    # Display the histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    # TODO: Plot the histogram
    # plt.subplot(...)
    # plt.bar(...)
    # plt.title(...)
    # plt.xlabel(...)
    # plt.ylabel(...)
    
    plt.tight_layout()
    plt.savefig('output/histogram.png')
    plt.show()
    
    return hist

def main():
    """
    Main function to run the assignment.
    """
    # Download a sample image if it doesn't exist
    sample_image_path = 'sample_image.jpg'
    if not os.path.exists(sample_image_path):
        print("Downloading a sample image...")
        # Example URL, replace with a stable image source
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks.png"
        urlretrieve(url, sample_image_path)
    
    # Load the image
    img_rgb, gray = load_image(sample_image_path)
    
    # Detect edges
    edges = detect_edges(gray)
    
    # Detect corners
    corners_image = detect_corners(gray, img_rgb)
    
    # Extract SIFT features
    keypoints, descriptors = extract_sift_features(gray, img_rgb)
    
    # Compute histogram
    histogram = compute_histogram(gray)
    
    print("Assignment completed successfully! Check the output directory for saved visualizations.")

if __name__ == "__main__":
    main() 