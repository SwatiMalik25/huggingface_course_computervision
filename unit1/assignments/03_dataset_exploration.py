#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 3: Dataset Exploration
=================================

This assignment introduces you to working with computer vision datasets using the
Hugging Face datasets library. You'll learn how to load, explore, and visualize 
datasets, which is an important step before building computer vision models.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

def load_and_explore_dataset():
    """
    Load a vision dataset from Hugging Face and explore its properties.
    
    Returns:
        datasets.Dataset: The loaded dataset
    """
    print("Loading dataset...")
    
    # TODO: Load a small vision dataset from Hugging Face (e.g., 'mnist', 'fashion_mnist', 'cifar10', etc.)
    # dataset = ...
    
    # TODO: Print information about the dataset
    # print(f"Dataset description: {dataset}")
    # print(f"Dataset features: {dataset['train'].features}")
    # print(f"Number of training examples: {len(dataset['train'])}")
    # print(f"Number of test examples: {len(dataset['test'])}")
    
    return dataset

def visualize_dataset_samples(dataset, num_samples=10):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset (datasets.Dataset): The dataset to visualize
        num_samples (int): Number of samples to visualize
        
    Returns:
        list: The sampled examples
    """
    # TODO: Get random samples from the training set
    # train_dataset = dataset['train']
    # sample_indices = ...
    # samples = ...
    
    # Display the samples
    plt.figure(figsize=(15, 8))
    
    # TODO: Display each sample
    # for i, sample in enumerate(samples):
    #     plt.subplot(...)
    #     plt.imshow(...)
    #     plt.title(...)
    #     plt.axis(...)
    
    plt.tight_layout()
    plt.savefig('output/dataset_samples.png')
    plt.show()
    
    return samples

def analyze_class_distribution(dataset):
    """
    Analyze and visualize the class distribution in the dataset.
    
    Args:
        dataset (datasets.Dataset): The dataset to analyze
        
    Returns:
        dict: Class distribution statistics
    """
    train_dataset = dataset['train']
    
    # TODO: Count the occurrences of each class
    # labels = ...
    # label_counts = ...
    # label_names = ...  # Get label names if available
    
    # TODO: Create a bar chart of class distribution
    # plt.figure(figsize=...)
    # plt.bar(...)
    # plt.title(...)
    # plt.xlabel(...)
    # plt.ylabel(...)
    # plt.xticks(...)
    
    plt.tight_layout()
    plt.savefig('output/class_distribution.png')
    plt.show()
    
    return {'label_counts': label_counts, 'label_names': label_names}

def create_data_loaders(dataset, batch_size=32):
    """
    Create PyTorch DataLoaders for the dataset with appropriate transformations.
    
    Args:
        dataset (datasets.Dataset): The dataset
        batch_size (int): Batch size for the data loaders
        
    Returns:
        tuple: Training and test DataLoaders
    """
    # TODO: Define transformations for the images
    # transform = transforms.Compose([
    #     ...
    # ])
    
    # TODO: Define a function to apply transformations to the dataset
    # def transform_examples(examples):
    #     ...
    #     return examples
    
    # TODO: Apply the transformations to the dataset
    # transformed_train_dataset = ...
    # transformed_test_dataset = ...
    
    # TODO: Create PyTorch DataLoaders
    # train_loader = ...
    # test_loader = ...
    
    return train_loader, test_loader

def visualize_batch(data_loader):
    """
    Visualize a batch of data from a DataLoader.
    
    Args:
        data_loader (torch.utils.data.DataLoader): The DataLoader
        
    Returns:
        tuple: The batch of images and labels
    """
    # TODO: Get a batch of data from the data loader
    # images, labels = ...
    
    # TODO: Convert the batch to a grid for visualization
    # grid_size = int(np.sqrt(len(images)))
    # grid = ...
    
    # Display the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.title('Batch of Images')
    plt.axis('off')
    plt.savefig('output/batch_visualization.png')
    plt.show()
    
    return images, labels

def compute_dataset_statistics(dataset):
    """
    Compute mean and standard deviation of the dataset.
    
    Args:
        dataset (datasets.Dataset): The dataset
        
    Returns:
        tuple: Mean and standard deviation of the dataset
    """
    train_dataset = dataset['train']
    
    # TODO: Compute the mean and standard deviation of the dataset
    # Extract pixel values from all images
    # all_pixels = ...
    
    # Compute mean and std
    # mean = ...
    # std = ...
    
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    
    return mean, std

def main():
    """
    Main function to run the assignment.
    """
    # Load and explore the dataset
    dataset = load_and_explore_dataset()
    
    # Visualize dataset samples
    samples = visualize_dataset_samples(dataset)
    
    # Analyze class distribution
    distribution = analyze_class_distribution(dataset)
    
    # Compute dataset statistics
    mean, std = compute_dataset_statistics(dataset)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(dataset)
    
    # Visualize a batch
    images, labels = visualize_batch(train_loader)
    
    print("Assignment completed successfully! Check the output directory for saved visualizations.")

if __name__ == "__main__":
    main() 