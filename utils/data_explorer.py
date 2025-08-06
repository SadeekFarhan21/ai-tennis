"""
Example script showing how to load and use the processed MovieLens 100K dataset.
This demonstrates basic data exploration and preparation for machine learning models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Tuple


def load_processed_data(data_dir: str = "data/processed") -> Dict:
    """
    Load all processed MovieLens data into a dictionary
    
    Returns:
        Dictionary containing all the processed datasets
    """
    print("Loading processed MovieLens 100K dataset...")
    
    data = {}
    
    # Load main dataframes
    data['ratings'] = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    data['movies'] = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    data['users'] = pd.read_csv(os.path.join(data_dir, 'users.csv'))
    
    # Load rating matrix
    data['rating_matrix'] = np.load(os.path.join(data_dir, 'rating_matrix.npy'))
    
    # Load mappings from CSV files
    try:
        user_mappings = pd.read_csv(os.path.join(data_dir, 'user_mappings.csv'))
        item_mappings = pd.read_csv(os.path.join(data_dir, 'item_mappings.csv'))
        
        # Create mapping dictionaries
        data['mappings'] = {
            'user_to_idx': dict(zip(user_mappings['user_id'], user_mappings['matrix_index'])),
            'item_to_idx': dict(zip(item_mappings['item_id'], item_mappings['matrix_index'])),
            'idx_to_user': dict(zip(user_mappings['matrix_index'], user_mappings['user_id'])),
            'idx_to_item': dict(zip(item_mappings['matrix_index'], item_mappings['item_id']))
        }
    except FileNotFoundError:
        # Fallback to pickle format if CSV doesn't exist
        with open(os.path.join(data_dir, 'mappings.pkl'), 'rb') as f:
            data['mappings'] = pickle.load(f)
    
    # Load statistics from CSV files
    try:
        data['dataset_info'] = pd.read_csv(os.path.join(data_dir, 'dataset_info.csv'), index_col=0)['value'].to_dict()
        data['rating_distribution'] = pd.read_csv(os.path.join(data_dir, 'rating_distribution.csv'))
        data['age_distribution'] = pd.read_csv(os.path.join(data_dir, 'age_distribution.csv'))
        data['gender_distribution'] = pd.read_csv(os.path.join(data_dir, 'gender_distribution.csv'))
        data['occupation_distribution'] = pd.read_csv(os.path.join(data_dir, 'occupation_distribution.csv'))
        data['genre_distribution'] = pd.read_csv(os.path.join(data_dir, 'genre_distribution.csv'))
        data['ratings_by_year'] = pd.read_csv(os.path.join(data_dir, 'ratings_by_year.csv'))
        data['ratings_by_month'] = pd.read_csv(os.path.join(data_dir, 'ratings_by_month.csv'))
    except FileNotFoundError:
        # Fallback to pickle format if CSV doesn't exist
        with open(os.path.join(data_dir, 'dataset_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            data['stats'] = stats
    
    # Convert datetime columns
    data['ratings']['datetime'] = pd.to_datetime(data['ratings']['datetime'])
    data['ratings']['date'] = pd.to_datetime(data['ratings']['date'])
    data['movies']['release_date'] = pd.to_datetime(data['movies']['release_date'])
    
    print(f"✓ Loaded {len(data['ratings'])} ratings")
    print(f"✓ Loaded {len(data['movies'])} movies") 
    print(f"✓ Loaded {len(data['users'])} users")
    print(f"✓ Rating matrix shape: {data['rating_matrix'].shape}")
    
    return data


def load_train_test_split(split_name: str = "u1", data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a specific train/test split
    
    Args:
        split_name: Name of the split (u1-u5, ua, ub)
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_file = os.path.join(data_dir, f'{split_name}_train.csv')
    test_file = os.path.join(data_dir, f'{split_name}_test.csv')
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Loaded {split_name} split: {len(train_df)} train, {len(test_df)} test samples")
    
    return train_df, test_df


def explore_dataset(data: Dict):
    """Basic dataset exploration"""
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    ratings_df = data['ratings']
    movies_df = data['movies']
    users_df = data['users']
    
    # Get dataset info (try CSV first, fallback to pickle format)
    if 'dataset_info' in data:
        total_ratings = int(data['dataset_info']['total_ratings'])
        sparsity = float(data['dataset_info']['sparsity'])
    else:
        # Fallback calculations
        total_ratings = len(ratings_df)
        sparsity = 1 - (total_ratings / (len(users_df) * len(movies_df)))
    
    # Basic info
    print(f"Dataset contains:")
    print(f"  - {len(ratings_df):,} ratings")
    print(f"  - {len(users_df):,} users")
    print(f"  - {len(movies_df):,} movies")
    print(f"  - Rating scale: {ratings_df['rating'].min()}-{ratings_df['rating'].max()}")
    print(f"  - Data sparsity: {sparsity:.2%}")
    
    # Most popular movies
    print(f"\nTop 5 most rated movies:")
    popular_movies = ratings_df.groupby('item_id').size().reset_index(name='rating_count')
    popular_movies = popular_movies.merge(movies_df[['movie_id', 'title']], left_on='item_id', right_on='movie_id')
    popular_movies = popular_movies.nlargest(5, 'rating_count')
    
    for i, row in popular_movies.iterrows():
        print(f"  {row['title']}: {row['rating_count']} ratings")
    
    # User activity distribution
    user_activity = ratings_df.groupby('user_id').size()
    print(f"\nUser activity:")
    print(f"  - Most active user: {user_activity.max()} ratings")
    print(f"  - Average ratings per user: {user_activity.mean():.1f}")
    print(f"  - Least active user: {user_activity.min()} ratings")
    
    # Rating distribution
    print(f"\nRating distribution:")
    for rating in sorted(ratings_df['rating'].unique()):
        count = (ratings_df['rating'] == rating).sum()
        pct = count / len(ratings_df) * 100
        print(f"  {rating}: {count:,} ({pct:.1f}%)")


def create_user_item_matrix(ratings_df: pd.DataFrame, fill_value: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    Create a user-item rating matrix from ratings dataframe
    
    Args:
        ratings_df: DataFrame with user_id, item_id, rating columns
        fill_value: Value to fill for missing ratings
        
    Returns:
        Tuple of (rating_matrix, mappings_dict)
    """
    print("\nCreating user-item matrix...")
    
    # Get unique users and items
    users = sorted(ratings_df['user_id'].unique())
    items = sorted(ratings_df['item_id'].unique())
    
    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    
    # Create matrix
    matrix = np.full((len(users), len(items)), fill_value)
    
    # Fill matrix with ratings
    for _, row in ratings_df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['item_id']]
        matrix[user_idx, item_idx] = row['rating']
    
    mappings = {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': {idx: user for user, idx in user_to_idx.items()},
        'idx_to_item': {idx: item for item, idx in item_to_idx.items()}
    }
    
    print(f"Created matrix of shape: {matrix.shape}")
    print(f"Sparsity: {(matrix == fill_value).sum() / matrix.size * 100:.2f}%")
    
    return matrix, mappings


def get_user_recommendations(user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n: int = 5):
    """
    Simple recommendation based on average ratings of unrated movies
    
    Args:
        user_id: ID of user to get recommendations for
        ratings_df: Ratings dataframe
        movies_df: Movies dataframe
        n: Number of recommendations
        
    Returns:
        DataFrame with top movie recommendations
    """
    # Get movies the user has already rated
    user_movies = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'])
    
    # Get all movies
    all_movies = set(movies_df['movie_id'])
    
    # Find unrated movies
    unrated_movies = all_movies - user_movies
    
    # Calculate average rating for each unrated movie
    movie_avg_ratings = ratings_df.groupby('item_id')['rating'].mean()
    movie_rating_counts = ratings_df.groupby('item_id').size()
    
    # Filter for unrated movies with at least 10 ratings
    recommendations = []
    for movie_id in unrated_movies:
        if movie_id in movie_avg_ratings and movie_rating_counts[movie_id] >= 10:
            recommendations.append({
                'movie_id': movie_id,
                'avg_rating': movie_avg_ratings[movie_id],
                'rating_count': movie_rating_counts[movie_id]
            })
    
    # Sort by average rating and take top n
    recommendations.sort(key=lambda x: x['avg_rating'], reverse=True)
    top_recs = recommendations[:n]
    
    # Add movie details
    rec_df = pd.DataFrame(top_recs)
    if not rec_df.empty:
        rec_df = rec_df.merge(movies_df[['movie_id', 'title', 'year']], on='movie_id')
    
    return rec_df


def main():
    """Example usage of the processed MovieLens dataset"""
    
    # Load processed data
    data = load_processed_data()
    
    # Explore the dataset
    explore_dataset(data)
    
    # Load a specific train/test split
    train_df, test_df = load_train_test_split("u1")
    
    # Create user-item matrix for training data
    train_matrix, mappings = create_user_item_matrix(train_df)
    
    # Example: Get recommendations for user 1
    print(f"\nExample recommendations for user 1:")
    recommendations = get_user_recommendations(1, data['ratings'], data['movies'])
    if not recommendations.empty:
        for i, row in recommendations.iterrows():
            print(f"  {row['title']} ({row['year']:.0f}) - {row['avg_rating']:.2f} avg rating ({row['rating_count']} ratings)")
    else:
        print("  No recommendations found")
    
    print(f"\n" + "="*50)
    print("DATASET READY FOR MACHINE LEARNING")
    print("="*50)
    print("The processed data includes:")
    print("- Clean CSV files for ratings, movies, and users")
    print("- Pre-built user-item rating matrix")
    print("- Multiple train/test splits for cross-validation")
    print("- Comprehensive statistics and metadata")
    print("- ID mappings for matrix operations")
    print(f"\nFiles available in: data/processed/")


if __name__ == "__main__":
    main()
