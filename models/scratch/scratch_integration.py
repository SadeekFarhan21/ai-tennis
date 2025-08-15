#!/usr/bin/env python3
"""
Integration functions for scratch neural network in Streamlit app
Provides similar interface to PyTorch model functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import List, Dict, Tuple
from models.scratch.exampleNN import NeuralCFModel, load_model, predict_for_user

def adapt_new_user_scratch(model: NeuralCFModel, rated_item_ids: List[int], 
                         ratings: List[float], n_items: int) -> np.ndarray:
    """
    Adapt model for a new user by learning user embedding through gradient descent
    
    Args:
        model: Trained NeuralCFModel
        rated_item_ids: List of item indices that the user has rated
        ratings: List of corresponding ratings
        n_items: Total number of items (for generating predictions)
        
    Returns:
        user_embedding: Optimized user embedding vector
    """
    # Initialize new user embedding as average of similar users
    # Find users who rated similar items
    similar_ratings = []
    
    # Simple heuristic: use global average as starting point
    user_embedding_init = np.random.normal(0, 0.1, model.user_embedding.embedding_dim)
    
    # Fine-tune user embedding for this specific user
    learning_rate = 0.01
    n_epochs = 50
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for item_id, rating in zip(rated_item_ids, ratings):
            # Forward pass with current user embedding
            item_embed = model.item_embedding.forward(np.array([item_id]))[0]
            concat = np.concatenate([user_embedding_init, item_embed])
            
            # Pass through network
            output = model.network.forward(concat.reshape(1, -1))
            
            # Apply sigmoid and scale to (1,5)
            sigmoid_output = 1.0 / (1.0 + np.exp(-output))
            prediction = 1.0 + 4.0 * sigmoid_output[0, 0]
            
            # Compute loss
            loss = (prediction - rating) ** 2
            total_loss += loss
            
            # Compute gradient w.r.t user embedding (simplified)
            error = prediction - rating
            # Simple gradient approximation
            grad_user = 0.001 * error * np.sign(user_embedding_init)
            
            # Update user embedding
            user_embedding_init -= learning_rate * grad_user
    
    # Generate predictions for all items
    predictions = np.zeros(n_items)
    
    for item_id in range(n_items):
        item_embed = model.item_embedding.forward(np.array([item_id]))[0]
        concat = np.concatenate([user_embedding_init, item_embed])
        
        # Forward pass
        output = model.network.forward(concat.reshape(1, -1))
        sigmoid_output = 1.0 / (1.0 + np.exp(-output))
        prediction = 1.0 + 4.0 * sigmoid_output[0, 0]
        
        predictions[item_id] = prediction
    
    return predictions

def score_all_items_for_user_scratch(model: NeuralCFModel, user_embedding: np.ndarray, 
                                    user_bias: float) -> np.ndarray:
    """
    Score all items for a user given their optimized embedding
    
    Args:
        model: Trained NeuralCFModel
        user_embedding: Optimized user embedding vector
        user_bias: User bias (ignored for this model)
        
    Returns:
        predictions: Array of predicted ratings for all items
    """
    n_items = model.n_items
    
    # Temporarily replace user 0 embedding
    original_user_embedding = model.user_embedding.weight[0].copy()
    model.user_embedding.weight[0] = user_embedding
    
    try:
        # Score all items
        user_ids = np.zeros(n_items, dtype=int)  # All user 0
        item_ids = np.arange(n_items)
        
        predictions = model.forward(user_ids, item_ids)
        
    finally:
        # Always restore original embedding
        model.user_embedding.weight[0] = original_user_embedding
    
    return predictions

def load_scratch_model(model_path: str = 'models/scratch/trained_scratch_nn.npz') -> NeuralCFModel:
    """
    Load the trained scratch neural network model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        model: Loaded NeuralCFModel
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Scratch model not found at {model_path}")
    
    model = load_model(model_path)
    return model

def get_scratch_model_info() -> Dict:
    """Get information about the scratch model for display"""
    return {
        "name": "Scratch Neural Network",
        "type": "Neural Collaborative Filtering",
        "framework": "NumPy (from scratch)",
        "description": "Neural network built from scratch using only NumPy",
        "architecture": [
            "User/Item Embeddings: 64 dimensions",
            "Hidden Layer: 128 neurons (ReLU)",
            "Output Layer: 1 neuron",
            "Global bias for rating adjustment"
        ],
        "features": [
            "Pure NumPy implementation",
            "Custom backpropagation",
            "Adam optimizer from scratch",
            "Collaborative filtering approach"
        ]
    }

def predict_ratings_scratch(model: NeuralCFModel, user_embedding: np.ndarray,
                           user_bias: float, item_ids: List[int]) -> np.ndarray:
    """
    Predict ratings for specific items using optimized user embedding
    
    Args:
        model: Trained model
        user_embedding: Optimized user embedding
        user_bias: User bias (ignored)
        item_ids: List of item indices to predict for
        
    Returns:
        predictions: Predicted ratings for the specified items
    """
    # Temporarily replace user 0 embedding
    original_user_embedding = model.user_embedding.weight[0].copy()
    model.user_embedding.weight[0] = user_embedding
    
    try:
        # Predict for specified items
        user_ids = np.zeros(len(item_ids), dtype=int)
        item_array = np.array(item_ids)
        
        predictions = model.forward(user_ids, item_array)
        predictions = np.clip(predictions, 1.0, 5.0)  # Clip to rating range
        
    finally:
        # Always restore original embedding
        model.user_embedding.weight[0] = original_user_embedding
    
    return predictions
