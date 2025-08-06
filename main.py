#!/usr/bin/env python3
"""
Movie Recommendation System - Learning Project

Starter script to load the MovieLens dataset.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import MovieLensDataLoader


def main():
    """Load data and show basic information"""
    print("Movie Recommendation System - Learning Project")
    print("=" * 50)
    
    # Load and process the data
    print("\nLoading MovieLens dataset...")
    data_loader = MovieLensDataLoader()
    
    try:
        # Process the data
        data = data_loader.process_data()
        
        print("\nData loaded successfully!")
        print(f"Train matrix shape: {data['train_matrix'].shape}")
        print(f"Test matrix shape: {data['test_matrix'].shape}")
        print(f"Number of users: {len(data['user_mapping'])}")
        print(f"Number of movies: {len(data['movie_mapping'])}")
        
        # Show some example movies
        print("\nExample movies:")
        movies_df = data['movies_df']
        for i, (_, movie) in enumerate(movies_df.head(5).iterrows()):
            print(f"  {i+1}. {movie['title']} ({movie['genres']})")
        
        print("\nReady to start building your models!")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Make sure the MovieLens dataset is in data/raw/ directory")


if __name__ == "__main__":
    main() 