#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 1: Diffusion Models
==============================

This assignment introduces you to diffusion models, a powerful class of generative
models in computer vision. You'll learn how to use pre-trained diffusion models to
generate images and understand the key concepts behind them.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from tqdm.auto import tqdm

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

def visualize_forward_diffusion():
    """
    Visualize the forward diffusion process on a sample image.
    """
    # Load a sample image
    img_path = "sample_image.jpg"
    if not os.path.exists(img_path):
        print("Downloading a sample image...")
        from urllib.request import urlretrieve
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-explanation.png"
        urlretrieve(url, img_path)
    
    img = Image.open(img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig('output/original_image.png')
    plt.show()
    
    # Convert to tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    x_0 = transform(img).unsqueeze(0)
    
    # TODO: Create a DDPM noise scheduler
    # noise_scheduler = ...
    
    # Visualize the forward process
    plt.figure(figsize=(15, 8))
    
    # TODO: Apply increasing amounts of noise and visualize
    # num_steps = 10
    # for i in range(num_steps):
    #     # Compute t value
    #     t = ...
    #     
    #     # Add noise
    #     noisy_image = ...
    #     
    #     # Display the noisy image
    #     plt.subplot(2, 5, i+1)
    #     img_to_show = ...
    #     plt.imshow(...)
    #     plt.title(...)
    #     plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/forward_diffusion.png')
    plt.show()

def visualize_reverse_diffusion(model_id="google/ddpm-cifar10-32"):
    """
    Visualize the reverse diffusion process using a pre-trained model.
    
    Args:
        model_id (str): Hugging Face model ID for a pre-trained diffusion model
    """
    # TODO: Load a pre-trained U-Net model and noise scheduler
    # model = ...
    # scheduler = ...
    
    # TODO: Set up the reverse diffusion process parameters
    # torch_device = ...
    # model = ...
    # sample_size = ...
    # batch_size = ...
    # num_inference_steps = ...
    
    # TODO: Initialize random noise
    # noise = ...
    
    # TODO: Prepare for sampling
    # scheduler.set_timesteps(...)
    # reversed_timesteps = ...
    
    # TODO: Perform the reverse diffusion process
    # sample = ...
    # reversed_diffusion_images = [...]
    # 
    # for t in reversed_timesteps:
    #     with torch.no_grad():
    #         # Predict noise residual
    #         ...
    #         
    #         # Update sample
    #         ...
    #     
    #     # Save intermediate result
    #     ...
    
    # Visualize the reverse process
    plt.figure(figsize=(15, 8))
    
    # TODO: Display the final result and intermediate steps
    # num_to_show = 10
    # indices = ...
    # 
    # for i, idx in enumerate(indices):
    #     plt.subplot(...)
    #     plt.imshow(...)
    #     plt.title(...)
    #     plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/reverse_diffusion.png')
    plt.show()
    
    return sample

def generate_with_stable_diffusion(prompt, num_images=4, guidance_scale=7.5):
    """
    Generate images using Stable Diffusion based on a text prompt.
    
    Args:
        prompt (str): Text prompt for image generation
        num_images (int): Number of images to generate
        guidance_scale (float): Scale for classifier-free guidance
        
    Returns:
        list: Generated images
    """
    # TODO: Load the Stable Diffusion pipeline
    # pipe = ...
    
    # TODO: Move the pipeline to the GPU if available
    # device = ...
    # pipe = ...
    
    # TODO: Generate images from the prompt
    # with torch.no_grad():
    #     images = ...
    
    # Display the generated images
    plt.figure(figsize=(15, 15))
    
    # TODO: Display each generated image
    # for i, image in enumerate(images):
    #     plt.subplot(...)
    #     plt.imshow(...)
    #     plt.title(...)
    #     plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/stable_diffusion_results.png')
    plt.show()
    
    # Save individual images
    for i, image in enumerate(images):
        image.save(f'output/generated_image_{i+1}.png')
    
    return images

def explore_guidance_scale(prompt, scales=[1.0, 3.0, 7.5, 15.0]):
    """
    Explore the effect of guidance scale on Stable Diffusion generations.
    
    Args:
        prompt (str): Text prompt for image generation
        scales (list): List of guidance scales to try
        
    Returns:
        dict: Dictionary of generated images indexed by guidance scale
    """
    # TODO: Load the Stable Diffusion pipeline
    # pipe = ...
    
    # TODO: Move the pipeline to the GPU if available
    # device = ...
    # pipe = ...
    
    results = {}
    
    plt.figure(figsize=(15, 5 * len(scales)))
    
    # TODO: Generate and display images for each guidance scale
    # for i, scale in enumerate(scales):
    #     with torch.no_grad():
    #         # Generate image with current scale
    #         ...
    #     
    #     # Save result
    #     ...
    #     
    #     # Display result
    #     plt.subplot(...)
    #     plt.imshow(...)
    #     plt.title(...)
    #     plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/guidance_scale_comparison.png')
    plt.show()
    
    return results

def main():
    """
    Main function to run the assignment.
    """
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    
    # Visualize the forward diffusion process
    visualize_forward_diffusion()
    
    # Visualize the reverse diffusion process
    visualize_reverse_diffusion()
    
    # Generate images with Stable Diffusion
    prompt = "A photograph of a cat sitting on a beach at sunset"
    generated_images = generate_with_stable_diffusion(prompt)
    
    # Explore the effect of guidance scale
    guidance_results = explore_guidance_scale(prompt)
    
    print("Assignment completed successfully! Check the output directory for saved visualizations.")

if __name__ == "__main__":
    main() 