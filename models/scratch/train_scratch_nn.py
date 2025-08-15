#!/usr/bin/env python3
"""
Training script for Scratch Neural Network
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.simple_data_loader import SimpleMovieLensLoader

from models.scratch.exampleNN import NeuralCFModel, train_model, save_model 

import numpy as np
from typing import List, Tuple

def create_training_data_scratch(train_matrix: np.ndarray) -> Tuple[List, List, List]:
    """
    Create training data from sparse matrix for scratch model
    
    Args:
        train_matrix: (n_users, n_items) sparse matrix with ratings
    
    Returns:
        user_ids: list of user indices
        item_ids: list of item indices  
        ratings: list of ratings
    """
    user_ids = []
    item_ids = []
    ratings = []
    
    # Get all non-zero ratings
    for user_idx in range(train_matrix.shape[0]):
        for item_idx in range(train_matrix.shape[1]):
            rating = train_matrix[user_idx, item_idx]
            if rating > 0:  # Non-zero rating
                user_ids.append(user_idx)
                item_ids.append(item_idx)
                ratings.append(rating)
    
    return user_ids, item_ids, ratings

def create_validation_split(user_ids: List, item_ids: List, ratings: List, 
                          val_ratio: float = 0.2) -> Tuple[Tuple, Tuple]:
    """
    Split training data into train/validation sets
    
    Args:
        user_ids, item_ids, ratings: Training data
        val_ratio: Fraction of data to use for validation
        
    Returns:
        train_data: (train_users, train_items, train_ratings)
        val_data: (val_users, val_items, val_ratings)
    """
    n_samples = len(ratings)
    indices = np.random.permutation(n_samples)
    
    val_size = int(n_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Split data
    train_users = [user_ids[i] for i in train_indices]
    train_items = [item_ids[i] for i in train_indices]
    train_ratings = [ratings[i] for i in train_indices]
    
    val_users = [user_ids[i] for i in val_indices]
    val_items = [item_ids[i] for i in val_indices]
    val_ratings = [ratings[i] for i in val_indices]
    
    return (train_users, train_items, train_ratings), (val_users, val_items, val_ratings)

def evaluate_on_test(model: NeuralCFModel, test_matrix: np.ndarray) -> dict:
    """Evaluate model on test set"""
    test_users, test_items, test_ratings = create_training_data_scratch(test_matrix)
    
    if len(test_ratings) == 0:
        return {"rmse": float('inf'), "mae": float('inf')}
    
    # Get predictions
    test_users_array = np.array(test_users)
    test_items_array = np.array(test_items)
    test_ratings_array = np.array(test_ratings)
    
    predictions = model.forward(test_users_array, test_items_array)
    predictions = np.clip(predictions, 1.0, 5.0)  # Clip to rating range
    
    # Calculate metrics
    mse = np.mean((predictions - test_ratings_array) ** 2)
    mae = np.mean(np.abs(predictions - test_ratings_array))
    rmse = np.sqrt(mse)
    
    return {"rmse": rmse, "mae": mae, "mse": mse}

def train_scratch_neural_network():
    """Train the Scratch Neural Network model"""
    
    print("🚀 Training Scratch Neural Network for Movie Recommendations")
    print("=" * 60)
    
    # Load data
    print("📁 Loading data...")
    loader = SimpleMovieLensLoader()
    data = loader.process_data(split_id=1, normalize=False)  # Don't normalize for neural net
    
    train_matrix = data['train_matrix']
    test_matrix = data['test_matrix']
    
    n_users, n_items = train_matrix.shape
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Training data shape: {train_matrix.shape}")
    print(f"   👥 Users: {n_users}, 🎬 Items: {n_items}")
    
    # Create training data
    print("\n🔄 Converting sparse matrix to training data...")
    user_ids, item_ids, ratings = create_training_data_scratch(train_matrix)
    print(f"   💫 Total ratings: {len(ratings):,}")
    print(f"   ⭐ Rating range: {min(ratings):.1f} - {max(ratings):.1f}")
    print(f"   📈 Average rating: {np.mean(ratings):.2f}")
    
    # Create validation split
    print("\n✂️ Creating train/validation split...")
    train_data, val_data = create_validation_split(user_ids, item_ids, ratings, val_ratio=0.15)
    train_users, train_items, train_ratings = train_data
    val_users, val_items, val_ratings = val_data
    
    print(f"   🏋️ Training samples: {len(train_ratings):,}")
    print(f"   🎯 Validation samples: {len(val_ratings):,}")
    
    # Create model
    print(f"\n🧠 Creating Neural Collaborative Filtering model...")
    model = NeuralCFModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=64,
        hidden_dim=128
    )
    
    print(f"   🎯 Architecture:")
    print(f"      • User embeddings: {n_users} × 64")
    print(f"      • Item embeddings: {n_items} × 64") 
    print(f"      • Hidden layer: 128 → 128 (ReLU)")
    print(f"      • Output layer: 128 → 1")
    
    # Training parameters
    training_params = {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 1024,
        'epochs': 30,
        'patience': 5
    }
    
    print(f"\n⚙️ Training parameters:")
    for key, value in training_params.items():
        print(f"   • {key}: {value}")
    
    # Train model
    print(f"\n🚀 Starting training...")
    print("-" * 60)
    
    history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        **training_params
    )
    
    print("-" * 60)
    print("✅ Training completed!")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_metrics = evaluate_on_test(model, test_matrix)
    
    print(f"   🎯 Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"   📏 Test MAE: {test_metrics['mae']:.4f}")
    print(f"   📈 Test MSE: {test_metrics['mse']:.4f}")
    
    # Save model
    model_path = 'models/scratch/trained_scratch_nn.npz'
    print(f"\n💾 Saving model to {model_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    save_model(model, model_path)
    
    # Save training history
    history_path = 'models/scratch/training_history.npz'
    np.savez(history_path, 
             train_loss=history['train_loss'],
             val_rmse=history['val_rmse'],
             best_val_rmse=history['best_val_rmse'])
    
    print(f"📈 Training history saved to {history_path}")
    
    print(f"\n🎉 All done! Model ready for use in Streamlit app.")
    print(f"📋 Final Results:")
    print(f"   • Best Validation RMSE: {history['best_val_rmse']:.4f}")
    print(f"   • Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"   • Test MAE: {test_metrics['mae']:.4f}")
    
    return model, data, history, test_metrics

if __name__ == "__main__":
    try:
        train_scratch_neural_network()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
