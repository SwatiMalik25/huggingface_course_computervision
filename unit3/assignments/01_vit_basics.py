#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Assignment 1: Vision Transformer Basics
=======================================

This assignment covers the fundamentals of Vision Transformers (ViT).
You will implement key components of the ViT architecture and understand how they work.

Instructions:
1. Fill in the code in the sections marked with TODO
2. Run this script to see if your implementation works correctly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Create a directory for saving outputs if it doesn't exist
os.makedirs("output", exist_ok=True)

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Args:
        img_size (int): Size of the input image (assumed to be square)
        patch_size (int): Size of each patch (assumed to be square)
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # TODO: Create a projection layer to convert patches to embeddings
        # self.proj = ...
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Embedded patches of shape [batch_size, n_patches, embed_dim]
        """
        # TODO: Implement the forward pass
        # Rearrange to obtain patch embeddings
        # x = ...
        # x = ...
        
        return x

class Attention(nn.Module):
    """
    Self-attention mechanism.
    
    Args:
        dim (int): Input dimension
        n_heads (int): Number of attention heads
        qkv_bias (bool): Whether to include bias in the query, key, value projections
    """
    def __init__(self, dim, n_heads=8, qkv_bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # TODO: Create query, key, value projections
        # self.qkv = ...
        # self.proj = ...
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_patches + 1, dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_patches + 1, dim]
        """
        batch_size, n_tokens, dim = x.shape
        
        # TODO: Implement the self-attention mechanism
        # Generate query, key, value matrices
        # qkv = ...
        # qkv = ...
        # q, k, v = ...
        
        # Compute attention scores
        # attn = ...
        # attn = ...
        
        # Apply attention to values
        # x = ...
        # x = ...
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    
    Args:
        dim (int): Input dimension
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): Whether to include bias in the query, key, value projections
        p (float): Dropout probability
    """
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, qkv_bias=True, p=0.0):
        super().__init__()
        
        # TODO: Create layer norm, attention, and MLP components
        # self.norm1 = ...
        # self.attn = ...
        # self.norm2 = ...
        # self.mlp = ...
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_patches + 1, dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_patches + 1, dim]
        """
        # TODO: Implement the transformer block forward pass
        # Apply attention with residual connection
        # x = ...
        
        # Apply MLP with residual connection
        # x = ...
        
        return x

class SimpleViT(nn.Module):
    """
    Simplified Vision Transformer for educational purposes.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        n_classes (int): Number of classes for classification
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): Whether to include bias in the query, key, value projections
        p (float): Dropout probability
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        n_classes=1000, 
        embed_dim=768, 
        depth=12, 
        n_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True, 
        p=0.0
    ):
        super().__init__()
        
        # TODO: Create patch embedding, positional embedding, and CLS token
        # self.patch_embed = ...
        # self.cls_token = ...
        # self.pos_embed = ...
        
        # TODO: Create transformer blocks
        # self.blocks = ...
        
        # TODO: Create classification head
        # self.norm = ...
        # self.head = ...
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Classification logits of shape [batch_size, n_classes]
        """
        # TODO: Implement the forward pass
        # Patch embedding
        # x = ...
        
        # Add CLS token and positional embedding
        # cls_token = ...
        # x = ...
        # x = ...
        
        # Apply transformer blocks
        # x = ...
        
        # Classification from CLS token
        # x = ...
        # x = ...
        
        return x

def visualize_patches(image_path, patch_size=16):
    """
    Load an image and visualize how it gets split into patches.
    
    Args:
        image_path (str): Path to the image file
        patch_size (int): Size of each patch (assumed to be square)
    """
    # Load and resize the image
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    
    # TODO: Split the image into patches
    # patches = ...
    
    # Display the original image and its patches
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # TODO: Create a grid to display patches
    # plt.subplot(1, 2, 2)
    # plt.imshow(...)
    # plt.title(f'Image split into {patches.shape[0]} patches')
    # plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/image_patches.png')
    plt.show()
    
    return patches

def visualize_attention(model, image_path):
    """
    Visualize attention maps from a trained Vision Transformer.
    
    Args:
        model (nn.Module): Trained ViT model
        image_path (str): Path to the image file
    """
    # This is a simplified placeholder - in a real implementation, 
    # we would need to extract attention weights from the model and visualize them
    
    # Load and preprocess the image
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # TODO: Extract attention weights (this is just a placeholder for demonstration)
    # In a real implementation, you would need to modify the model to return attention weights
    # attention_weights = ...
    
    # Create a dummy attention map for visualization purposes
    attention_map = torch.randn(14, 14)  # For a 224x224 image with patch size 16
    attention_map = F.softmax(attention_map.view(-1), dim=0).view(14, 14)
    
    # Display the original image and the attention map
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor[0].permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(attention_map.detach().numpy(), cmap='viridis')
    plt.title('Attention Map (Placeholder)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/attention_map.png')
    plt.show()

def main():
    """
    Main function to run the assignment.
    """
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    
    # Download a sample image if it doesn't exist
    sample_image_path = 'sample_image.jpg'
    if not os.path.exists(sample_image_path):
        print("Downloading a sample image...")
        from urllib.request import urlretrieve
        # Example URL, replace with a stable image source
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks.png"
        urlretrieve(url, sample_image_path)
    
    # Visualize how an image is split into patches
    patches = visualize_patches(sample_image_path)
    
    # Create a simple Vision Transformer model
    model = SimpleViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=1000,
        embed_dim=384,
        depth=4,
        n_heads=6,
        mlp_ratio=4.0
    )
    
    # Print model architecture
    print(model)
    
    # Visualize attention (this is a placeholder - in a real implementation, 
    # we would use a trained model)
    visualize_attention(model, sample_image_path)
    
    print("Assignment completed successfully! Check the output directory for saved visualizations.")

if __name__ == "__main__":
    main() 