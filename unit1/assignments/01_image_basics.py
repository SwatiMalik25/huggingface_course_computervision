#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 1: Image Basics
==========================

This assignment covers the fundamentals of working with digital images in Python.
You'll learn how to load, manipulate, and visualize images, as well as work with
different color spaces and basic transformations.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

def load_and_display_image(image_path):
    """
    Load and display an image from the given path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: The loaded image
    """
    # TODO: Load the image using PIL's Image module
    # image = ...
    
    # TODO: Display the image using matplotlib
    # plt.figure(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    # plt.show()
    
    return image

def convert_color_spaces(image):
    """
    Convert an image between different color spaces and display the results.
    
    Args:
        image (PIL.Image): The input image
        
    Returns:
        dict: A dictionary containing the image in different color spaces
    """
    # Convert PIL Image to numpy array for easier manipulation
    img_array = np.array(image)
    
    # TODO: Convert the image from RGB to grayscale
    # gray = ...
    
    # TODO: Convert the image from RGB to HSV
    # hsv = ...
    
    # TODO: Convert the image from RGB to LAB
    # lab = ...
    
    # Display the original and converted images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_array)
    plt.title('Original (RGB)')
    plt.axis('off')
    
    # TODO: Display the grayscale image
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the HSV image (only display the hue channel)
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the LAB image (only display the L channel)
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/color_spaces.png')
    plt.show()
    
    return {
        'rgb': img_array,
        'gray': gray,
        'hsv': hsv, 
        'lab': lab
    }

def image_channels(img_array):
    """
    Split an RGB image into its color channels and display them.
    
    Args:
        img_array (numpy.ndarray): The input RGB image as a numpy array
        
    Returns:
        tuple: The red, green, and blue channel arrays
    """
    # TODO: Split the image into its red, green, and blue channels
    # red_channel = ...
    # green_channel = ...
    # blue_channel = ...
    
    # Display the individual channels
    plt.figure(figsize=(15, 10))
    
    # TODO: Display the original RGB image
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the red channel
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the green channel
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the blue channel
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/image_channels.png')
    plt.show()
    
    return red_channel, green_channel, blue_channel

def basic_transformations(image):
    """
    Apply basic transformations to an image and display the results.
    
    Args:
        image (PIL.Image): The input image
        
    Returns:
        dict: A dictionary containing the transformed images
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # TODO: Resize the image to half its original size
    # resized = ...
    
    # TODO: Rotate the image by 45 degrees
    # rotated = ...
    
    # TODO: Flip the image horizontally
    # flipped_h = ...
    
    # TODO: Flip the image vertically
    # flipped_v = ...
    
    # Display the original and transformed images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_array)
    plt.title('Original')
    plt.axis('off')
    
    # TODO: Display the resized image
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the rotated image
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the horizontally flipped image
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    # TODO: Display the vertically flipped image
    # plt.subplot(...)
    # plt.imshow(...)
    # plt.title(...)
    # plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/transformations.png')
    plt.show()
    
    return {
        'resized': resized,
        'rotated': rotated,
        'flipped_h': flipped_h,
        'flipped_v': flipped_v
    }

def main():
    """
    Main function to run the assignment.
    """
    # Download a sample image if it doesn't exist
    sample_image_path = 'sample_image.jpg'
    if not os.path.exists(sample_image_path):
        print("Downloading a sample image...")
        # TODO: Add code to download a sample image or use a URL
        from urllib.request import urlretrieve
        # Example URL, replace with a stable image source
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks.png"
        urlretrieve(url, sample_image_path)
    
    # Load and display the image
    image = load_and_display_image(sample_image_path)
    
    # Convert between color spaces
    color_spaces = convert_color_spaces(image)
    
    # Split and display the image channels
    r, g, b = image_channels(color_spaces['rgb'])
    
    # Apply basic transformations
    transformations = basic_transformations(image)
    
    print("Assignment completed successfully! Check the output directory for saved visualizations.")

if __name__ == "__main__":
    main() 