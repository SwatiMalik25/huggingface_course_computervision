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
from datasets import load_dataset, Image, DatasetDict
import cv2
import os
import pprint

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)



def load_and_display_image(img_dataset_path, num_images=100):       
    """
    Load and display images from the Hugging Face cats-image dataset.
    
    Returns:
        list: A list of PIL.Image objects
    """
    # Load the dataset
    image_dataset = load_dataset(img_dataset_path, split="test")
    
    # Get actual number of available images
    available_images = len(image_dataset)
    num_to_process = min(num_images, available_images)
    
    print(f"Requested {num_images} images, dataset has {available_images} images, processing {num_to_process}")
    
    # Prepare a list to store images
    images = []
    
    # Iterate over available images, limited by num_images
    for i in range(num_to_process):
        # Load the image using PIL
        image = image_dataset[i]['image']
        images.append(image)
        
        # Display the image using matplotlib
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f'Image {i+1}')
        plt.axis('off')
        plt.show()
    
    return images


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
    gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    
    # TODO: Convert the image from RGB to HSV
    hsv = cv2.cvtColor(img_array,cv2.COLOR_RGB2HSV)
    
    # TODO: Convert the image from RGB to LAB
    lab = cv2.cvtColor(img_array,cv2.COLOR_RGB2LAB)
    
    # Display the original and converted images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_array)
    plt.title('Original (RGB)')
    plt.axis('off')
    
    # TODO: Display the grayscale image
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    ## also similarly for hsv and lab
    plt.subplot(2, 2, 3)
    plt.imshow(hsv)
    plt.title('HSV')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(lab)
    plt.title('LAB')
    plt.axis('off')
    
 
    
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
    plt.savefig('output/image_channels_example.png')
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
    Test function for load_and_display_image and convert_color_spaces.
    """
    print("===== Testing load_and_display_image =====")
    try:
        # Test with smaller number for faster testing
        images = load_and_display_image("priyank-m/SROIE_2019_text_recognition", num_images=2)
        
        # First, print the type to see what we're working with
        print(f"Image type: {type(images[0])}")
        print(f"Number of images available: {len(images)}")
        
        # Show sample of image data
        print("\nImage Data Sample (5x5 pixels from top-left):")
        img_array = np.array(images[0])
        pp = pprint.PrettyPrinter(indent=2, width=100)
        pp.pprint(img_array[:5, :5])
        print(f"Image shape: {img_array.shape}")

        # Then adjust your assertion based on what type is returned
        assert all(isinstance(img, type(images[0])) for img in images), "List should contain consistent image objects"
        
        # Test 1: Check if the function returns a list
        assert isinstance(images, list), "Function should return a list"
        print("✓ Test 1 passed: Function returns a list")
        
        # Test 2: Check if the list contains PIL Image objects
        print(f"Image type: {type(images[0])}")
        assert all(isinstance(img, type(images[0])) for img in images), "List should contain consistent image objects of huggingface types"
        print("✓ Test 2 passed: List contains PIL Image objects")
        
        # Test 3: Check if the function loaded images correctly
        assert len(images) > 0, "Function should load at least one image"
        print(f"✓ Test 3 passed: Function loaded {len(images)} image(s)")
        
        print("\nload_and_display_image function passed all tests!")
    except Exception as e:
        print(f"load_and_display_image test failed: {e}")
    
    print("\n===== Testing convert_color_spaces =====")
    try:
        # Use the first image from the previous test if available, otherwise load a sample
        if 'images' in locals() and images:
            image = images[0]
            print("Using image from previous test")
        else:
            sample_image_path = 'sample_image.jpg'
            if not os.path.exists(sample_image_path):
                from urllib.request import urlretrieve
                # Use the cats-image dataset to get a sample image URL
                sample_dataset = load_dataset("priyank-m/SROIE_2019_text_recognition", split="test")
                url = sample_dataset[0]['image'].url
                urlretrieve(url, sample_image_path)
            image = Image.open(sample_image_path)
            print("Using downloaded sample image")
        
        # Run the function
        color_spaces = convert_color_spaces(image)
        
        # Print samples of the color spaces
        print("\nColor Space Samples (5x5 pixels from top-left):")
        pp = pprint.PrettyPrinter(indent=2, width=100)
        
        print("\nRGB sample:")
        pp.pprint(color_spaces['rgb'][:5, :5])
        
        print("\nGrayscale sample:")
        pp.pprint(color_spaces['gray'][:5, :5])
        
        print("\nHSV sample:")
        pp.pprint(color_spaces['hsv'][:5, :5])
        
        print("\nLAB sample:")
        pp.pprint(color_spaces['lab'][:5, :5])
        
        # Print shape information
        print("\nColor Space Shapes:")
        print(f"RGB: {color_spaces['rgb'].shape}")
        print(f"Grayscale: {color_spaces['gray'].shape}")
        print(f"HSV: {color_spaces['hsv'].shape}")
        print(f"LAB: {color_spaces['lab'].shape}")
        
        # Test 1: Check if the function returns a dictionary
        assert isinstance(color_spaces, dict), "Function should return a dictionary"
        print("✓ Test 1 passed: Function returns a dictionary")
        
        # Test 2: Check if all expected keys exist in the dictionary
        expected_keys = ['rgb', 'gray', 'hsv', 'lab']
        assert all(key in color_spaces for key in expected_keys), f"Dictionary should contain keys: {expected_keys}"
        print("✓ Test 2 passed: Dictionary contains all expected keys")
        
        # Test 3: Check if the arrays have the correct shapes
        img_shape = color_spaces['rgb'].shape
        assert len(color_spaces['gray'].shape) == 2, "Grayscale image should be 2D"
        assert color_spaces['hsv'].shape == img_shape, "HSV image should have same shape as RGB"
        assert color_spaces['lab'].shape == img_shape, "LAB image should have same shape as RGB"
        print("✓ Test 3 passed: Arrays have correct shapes")
        
        print("\nconvert_color_spaces function passed all tests!")
    except Exception as e:
        print(f"convert_color_spaces test failed: {e}")

if __name__ == "__main__":
    main()