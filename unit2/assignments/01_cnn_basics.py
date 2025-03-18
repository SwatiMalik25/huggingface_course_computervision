#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 1: CNN Basics
========================

This assignment covers the fundamentals of Convolutional Neural Networks (CNNs).
You will build and train a simple CNN for image classification on the CIFAR-10 dataset.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # TODO: Define the CNN architecture
        # Define convolutional layers
        # self.conv1 = ...
        # self.conv2 = ...
        
        # TODO: Define pooling layers if needed
        # self.pool = ...
        
        # TODO: Define fully connected layers
        # self.fc1 = ...
        # self.fc2 = ...
        # self.fc3 = ...
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # TODO: Implement the forward pass
        # Apply first convolution and pooling
        # x = ...
        
        # Apply second convolution and pooling
        # x = ...
        
        # Flatten the output for the fully connected layers
        # x = ...
        
        # Apply fully connected layers
        # x = ...
        # x = ...
        
        return x

def load_cifar10_dataset():
    """
    Load the CIFAR-10 dataset and apply transformations.
    
    Returns:
        tuple: Training and test datasets
    """
    # TODO: Define transformations for training and test datasets
    # transform_train = transforms.Compose([
    #     ...
    # ])
    
    # transform_test = transforms.Compose([
    #     ...
    # ])
    
    # TODO: Load datasets using Hugging Face datasets
    # cifar10 = ...
    
    # TODO: Apply transformations to the datasets
    # def transform_train_examples(examples):
    #     ...
    #     return examples
    
    # def transform_test_examples(examples):
    #     ...
    #     return examples
    
    # TODO: Apply transformations
    # train_dataset = ...
    # test_dataset = ...
    
    return train_dataset, test_dataset

def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    """
    Create DataLoaders for the training and test datasets.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size (int): Batch size for training
        
    Returns:
        tuple: Training and test DataLoaders
    """
    # TODO: Create DataLoaders
    # train_loader = ...
    # test_loader = ...
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=5):
    """
    Train the CNN model.
    
    Args:
        model (nn.Module): The CNN model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: Lists of training losses and test accuracies
    """
    # TODO: Define loss function and optimizer
    # criterion = ...
    # optimizer = ...
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # TODO: Implement the training loop
        # for inputs, targets in train_loader:
        #     inputs, targets = ...
        #     
        #     # Zero the parameter gradients
        #     ...
        #     
        #     # Forward pass
        #     ...
        #     
        #     # Compute loss
        #     ...
        #     
        #     # Backward pass and optimize
        #     ...
        #     ...
        #     
        #     running_loss += ...
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        # TODO: Implement the evaluation loop
        # with torch.no_grad():
        #     for inputs, targets in test_loader:
        #         ...
        #         outputs = ...
        #         _, predicted = ...
        #         total += ...
        #         correct += ...
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Accuracy: {accuracy:.2f}%")
    
    return train_losses, test_accuracies

def plot_training_results(train_losses, test_accuracies):
    """
    Plot the training loss and test accuracy.
    
    Args:
        train_losses (list): Training losses for each epoch
        test_accuracies (list): Test accuracies for each epoch
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'r-')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('output/training_results.png')
    plt.show()

def visualize_filters(model):
    """
    Visualize the filters learned by the first convolutional layer.
    
    Args:
        model (nn.Module): The trained CNN model
    """
    # TODO: Extract and visualize filters from the first convolutional layer
    # filters = ...
    
    # Display the filters
    plt.figure(figsize=(10, 10))
    
    # TODO: Plot each filter
    # for i in range(min(16, filters.shape[0])):
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(...)
    #     plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/conv_filters.png')
    plt.show()

def main():
    """
    Main function to run the assignment.
    """
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    
    # Load the dataset
    train_dataset, test_dataset = load_cifar10_dataset()
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    
    # Create and initialize the CNN model
    model = SimpleCNN()
    
    # Print model architecture
    print(model)
    
    # Train the model
    train_losses, test_accuracies = train_model(model, train_loader, test_loader)
    
    # Plot training results
    plot_training_results(train_losses, test_accuracies)
    
    # Visualize filters
    visualize_filters(model)
    
    # Save the trained model
    torch.save(model.state_dict(), 'output/simple_cnn.pth')
    
    print("Assignment completed successfully! Check the output directory for saved visualizations and model.")

if __name__ == "__main__":
    main() 