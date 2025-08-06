"""
Example script showing how to work with the CSV mapping files.
This demonstrates loading and using the mappings for matrix operations.
"""

import pandas as pd
import numpy as np

def load_mappings_from_csv(data_dir: str = None):
    """Load user and item mappings from CSV files"""
    
    if data_dir is None:
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/processed")
    
    # Load user mappings
    user_mappings = pd.read_csv(f"{data_dir}/user_mappings.csv")
    print(f"Loaded {len(user_mappings)} user mappings")
    print("User mappings sample:")
    print(user_mappings.head())
    
    # Load item mappings  
    item_mappings = pd.read_csv(f"{data_dir}/item_mappings.csv")
    print(f"\nLoaded {len(item_mappings)} item mappings")
    print("Item mappings sample:")
    print(item_mappings.head())
    
    # Create lookup dictionaries
    user_to_idx = dict(zip(user_mappings['user_id'], user_mappings['matrix_index']))
    item_to_idx = dict(zip(item_mappings['item_id'], item_mappings['matrix_index']))
    idx_to_user = dict(zip(user_mappings['matrix_index'], user_mappings['user_id']))
    idx_to_item = dict(zip(item_mappings['matrix_index'], item_mappings['item_id']))
    
    return {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'user_mappings_df': user_mappings,
        'item_mappings_df': item_mappings
    }

def load_statistics_from_csv(data_dir: str = None):
    """Load dataset statistics from CSV files"""
    
    if data_dir is None:
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/processed")
    
    stats = {}
    
    # Load basic dataset info
    dataset_info = pd.read_csv(f"{data_dir}/dataset_info.csv", index_col=0)
    stats['dataset_info'] = dataset_info['value'].to_dict()
    print("Dataset Info:")
    print(dataset_info)
    
    # Load rating distribution
    rating_dist = pd.read_csv(f"{data_dir}/rating_distribution.csv")
    stats['rating_distribution'] = rating_dist
    print(f"\nRating Distribution:")
    print(rating_dist)
    
    # Load genre distribution
    genre_dist = pd.read_csv(f"{data_dir}/genre_distribution.csv")
    stats['genre_distribution'] = genre_dist
    print(f"\nTop 10 Genres by Movie Count:")
    print(genre_dist.head(10))
    
    # Load demographic distributions
    age_dist = pd.read_csv(f"{data_dir}/age_distribution.csv")
    gender_dist = pd.read_csv(f"{data_dir}/gender_distribution.csv")
    occupation_dist = pd.read_csv(f"{data_dir}/occupation_distribution.csv")
    
    stats['demographics'] = {
        'age': age_dist,
        'gender': gender_dist,
        'occupation': occupation_dist
    }
    
    print(f"\nUser Demographics:")
    print("Age Distribution:")
    print(age_dist)
    print("\nGender Distribution:")
    print(gender_dist)
    
    return stats

def demonstrate_matrix_operations():
    """Demonstrate how to use the mappings for matrix operations"""
    
    print("\n" + "="*60)
    print("MATRIX OPERATIONS EXAMPLE")
    print("="*60)
    
    # Load mappings
    mappings = load_mappings_from_csv()
    
    # Load the rating matrix
    import os
    matrix_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/processed/rating_matrix.npy")
    rating_matrix = np.load(matrix_path)
    print(f"\nLoaded rating matrix of shape: {rating_matrix.shape}")
    
    # Example: Get ratings for a specific user
    user_id = 1
    user_idx = mappings['user_to_idx'][user_id]
    user_ratings = rating_matrix[user_idx, :]
    
    # Find movies rated by this user
    rated_items = np.where(user_ratings > 0)[0]
    print(f"\nUser {user_id} rated {len(rated_items)} movies:")
    
    for i, item_idx in enumerate(rated_items[:5]):  # Show first 5
        item_id = mappings['idx_to_item'][item_idx]
        rating = user_ratings[item_idx]
        print(f"  Movie ID {item_id}: {rating} stars")
    
    # Example: Get all ratings for a specific movie
    item_id = 1  # Toy Story
    item_idx = mappings['item_to_idx'][item_id]
    movie_ratings = rating_matrix[:, item_idx]
    
    # Find users who rated this movie
    rating_users = np.where(movie_ratings > 0)[0]
    print(f"\nMovie {item_id} was rated by {len(rating_users)} users:")
    print(f"Average rating: {movie_ratings[movie_ratings > 0].mean():.2f}")
    print(f"Rating distribution: {np.bincount(movie_ratings[movie_ratings > 0].astype(int))[1:]}")

def create_mapping_summary():
    """Create a summary of the mapping files"""
    
    print("\n" + "="*60)
    print("MAPPING SUMMARY")
    print("="*60)
    
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/processed")
    
    # Load mapping summary
    mappings_summary = pd.read_csv(os.path.join(data_dir, "mappings_summary.csv"))
    print("Mappings Summary:")
    print(mappings_summary)
    
    # Additional analysis
    user_mappings = pd.read_csv(os.path.join(data_dir, "user_mappings.csv"))
    item_mappings = pd.read_csv(os.path.join(data_dir, "item_mappings.csv"))
    
    print(f"\nDetailed Mapping Analysis:")
    print(f"Users:")
    print(f"  - ID range: {user_mappings['user_id'].min()} to {user_mappings['user_id'].max()}")
    print(f"  - Matrix indices: 0 to {user_mappings['matrix_index'].max()}")
    print(f"  - Sequential: {(user_mappings['user_id'] == user_mappings['matrix_index'] + 1).all()}")
    
    print(f"\nItems:")
    print(f"  - ID range: {item_mappings['item_id'].min()} to {item_mappings['item_id'].max()}")
    print(f"  - Matrix indices: 0 to {item_mappings['matrix_index'].max()}")
    print(f"  - Sequential: {(item_mappings['item_id'] == item_mappings['matrix_index'] + 1).all()}")

def main():
    """Main function demonstrating CSV usage"""
    
    print("="*60)
    print("CSV MAPPINGS AND STATISTICS DEMO")
    print("="*60)
    
    # Load and display mappings
    mappings = load_mappings_from_csv()
    
    print("\n" + "-"*40)
    
    # Load and display statistics
    stats = load_statistics_from_csv()
    
    # Create mapping summary
    create_mapping_summary()
    
    # Demonstrate matrix operations
    demonstrate_matrix_operations()
    
    print("\n" + "="*60)
    print("CSV FORMAT BENEFITS")
    print("="*60)
    print("✅ Human readable - can open in Excel, Google Sheets, etc.")
    print("✅ Language agnostic - can be used with R, Python, Java, etc.")
    print("✅ Version control friendly - easy to see changes in git diffs")
    print("✅ Lightweight - smaller file sizes than pickle")
    print("✅ Platform independent - works across different operating systems")
    print("✅ Easy to query - can use SQL tools or pandas for analysis")
    print("✅ Backup friendly - can be easily imported/exported")

if __name__ == "__main__":
    main()
