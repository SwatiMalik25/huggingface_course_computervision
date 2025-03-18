#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 1: CLIP Basics
=========================

This assignment introduces you to OpenAI's CLIP (Contrastive Language-Image Pre-training) model,
a powerful multimodal model that connects text and images. You'll learn how to use CLIP for 
zero-shot image classification and understand its capabilities.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

def load_clip_model():
    """
    Load a pre-trained CLIP model and processor.
    
    Returns:
        tuple: CLIP model and processor
    """
    # TODO: Load the CLIP model and processor from Hugging Face
    # model = ...
    # processor = ...
    
    return model, processor

def zero_shot_classification(model, processor, image, candidate_labels):
    """
    Perform zero-shot classification on an image using CLIP.
    
    Args:
        model: Pre-trained CLIP model
        processor: CLIP processor
        image (PIL.Image): Input image
        candidate_labels (list): List of candidate class names
        
    Returns:
        tuple: Probabilities and predicted label
    """
    # TODO: Prepare the text inputs
    # text_inputs = ...
    
    # TODO: Prepare the image inputs
    # image_inputs = ...
    
    # TODO: Get the model outputs
    # with torch.no_grad():
    #     outputs = ...
    
    # TODO: Calculate probabilities
    # logits_per_image = ...
    # probs = ...
    
    # TODO: Get the predicted label
    # predicted_label_idx = ...
    # predicted_label = ...
    
    return probs, predicted_label

def visualize_classification_results(image, candidate_labels, probs):
    """
    Visualize the zero-shot classification results.
    
    Args:
        image (PIL.Image): Input image
        candidate_labels (list): List of candidate class names
        probs (torch.Tensor): Classification probabilities
    """
    # Convert probabilities to a list
    probs = probs.cpu().numpy()[0]
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display the classification results
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(candidate_labels))
    
    # TODO: Create a bar chart of the probabilities
    # plt.barh(...)
    # plt.yticks(...)
    # plt.xlabel(...)
    # plt.title(...)
    
    plt.tight_layout()
    plt.savefig('output/classification_results.png')
    plt.show()

def compare_categories(model, processor, image, category_sets):
    """
    Compare how CLIP performs with different sets of categories.
    
    Args:
        model: Pre-trained CLIP model
        processor: CLIP processor
        image (PIL.Image): Input image
        category_sets (dict): Dictionary of category sets to compare
        
    Returns:
        dict: Dictionary of classification results for each category set
    """
    results = {}
    
    plt.figure(figsize=(15, 5 * len(category_sets)))
    
    for i, (name, categories) in enumerate(category_sets.items()):
        # TODO: Perform zero-shot classification with the current category set
        # probs, predicted_label = ...
        
        # Save the results
        results[name] = {
            'probs': probs,
            'predicted_label': predicted_label
        }
        
        # Display the results
        plt.subplot(len(category_sets), 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f'Input Image\nPredicted as: {predicted_label} ({name} categories)')
        plt.axis('off')
        
        plt.subplot(len(category_sets), 2, 2*i + 2)
        probs_list = probs.cpu().numpy()[0]
        y_pos = np.arange(len(categories))
        
        # TODO: Create a bar chart of the probabilities
        # plt.barh(...)
        # plt.yticks(...)
        # plt.xlabel(...)
        # plt.title(...)
    
    plt.tight_layout()
    plt.savefig('output/category_comparison.png')
    plt.show()
    
    return results

def prompt_engineering(model, processor, image, prompt_templates):
    """
    Explore how different prompt templates affect CLIP's classification.
    
    Args:
        model: Pre-trained CLIP model
        processor: CLIP processor
        image (PIL.Image): Input image
        prompt_templates (list): List of prompt templates
        
    Returns:
        dict: Dictionary of classification results for each prompt template
    """
    # Define some basic categories
    categories = ["dog", "cat", "bird", "fish", "rabbit"]
    
    results = {}
    
    plt.figure(figsize=(15, 4 * len(prompt_templates)))
    
    for i, template in enumerate(prompt_templates):
        # TODO: Create prompts using the template
        # prompts = ...
        
        # TODO: Perform zero-shot classification with the prompts
        # text_inputs = ...
        # image_inputs = ...
        # 
        # with torch.no_grad():
        #     outputs = ...
        # 
        # logits_per_image = ...
        # probs = ...
        # predicted_idx = ...
        # predicted_category = ...
        
        # Save the results
        results[template] = {
            'probs': probs,
            'predicted_category': predicted_category
        }
        
        # Display the results
        plt.subplot(len(prompt_templates), 2, 2*i + 1)
        plt.imshow(image)
        plt.title(f'Input Image\nPredicted as: {predicted_category}\nTemplate: "{template}"')
        plt.axis('off')
        
        plt.subplot(len(prompt_templates), 2, 2*i + 2)
        probs_list = probs.cpu().numpy()[0]
        y_pos = np.arange(len(categories))
        
        # TODO: Create a bar chart of the probabilities
        # plt.barh(...)
        # plt.yticks(...)
        # plt.xlabel(...)
        # plt.title(...)
    
    plt.tight_layout()
    plt.savefig('output/prompt_engineering.png')
    plt.show()
    
    return results

def main():
    """
    Main function to run the assignment.
    """
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    
    # Load CLIP model and processor
    model, processor = load_clip_model()
    
    # Load a test image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks.png"
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Perform zero-shot classification with basic categories
    candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a fish", "a photo of a rabbit"]
    probs, predicted_label = zero_shot_classification(model, processor, image, candidate_labels)
    
    # Visualize the classification results
    visualize_classification_results(image, candidate_labels, probs)
    
    # Compare different sets of categories
    category_sets = {
        'animals': ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a fish", "a photo of a rabbit"],
        'vehicles': ["a photo of a car", "a photo of a bicycle", "a photo of a motorcycle", "a photo of a bus", "a photo of a truck"],
        'furniture': ["a photo of a chair", "a photo of a table", "a photo of a sofa", "a photo of a bed", "a photo of a desk"]
    }
    compare_results = compare_categories(model, processor, image, category_sets)
    
    # Explore prompt engineering
    prompt_templates = [
        "a photo of a {}",
        "a portrait of a {}",
        "a close-up of a {}",
        "a {} in the wild",
        "a {} in its natural habitat"
    ]
    prompt_results = prompt_engineering(model, processor, image, prompt_templates)
    
    print("Assignment completed successfully! Check the output directory for saved visualizations.")

if __name__ == "__main__":
    main() 