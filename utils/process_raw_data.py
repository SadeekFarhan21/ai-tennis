import pandas as pd
import numpy as np
import os
import datetime
from typing import Dict, List, Tuple
import pickle


class MovieLens100KProcessor:
    """
    Processor for the MovieLens 100K dataset in raw format.
    Converts raw data files into clean, structured datasets.
    """
    
    def __init__(self, raw_data_dir: str = "data/raw", output_dir: str = "data/processed"):
        # Get the project root directory (one level up from utils)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Make paths absolute if they're relative
        if not os.path.isabs(raw_data_dir):
            self.raw_data_dir = os.path.join(project_root, raw_data_dir)
        else:
            self.raw_data_dir = raw_data_dir
            
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(project_root, output_dir)
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Genre mappings
        self.genre_mapping = {}
        self.genre_names = []
        
    def load_genres(self) -> Dict[str, int]:
        """Load genre mappings from u.genre file"""
        genre_file = os.path.join(self.raw_data_dir, "u.genre")
        
        with open(genre_file, 'r', encoding='latin-1') as f:
            for line in f:
                if line.strip():
                    genre, genre_id = line.strip().split('|')
                    self.genre_mapping[genre] = int(genre_id)
                    self.genre_names.append(genre)
        
        print(f"Loaded {len(self.genre_mapping)} genres")
        return self.genre_mapping
    
    def process_ratings(self) -> pd.DataFrame:
        """Process ratings data from u.data file"""
        print("Processing ratings data...")
        
        # Column names for ratings data
        ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        
        # Load ratings data
        ratings_file = os.path.join(self.raw_data_dir, "u.data")
        ratings_df = pd.read_csv(
            ratings_file, 
            sep='\t', 
            names=ratings_cols,
            header=None
        )
        
        # Convert timestamp to datetime
        ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        ratings_df['date'] = ratings_df['datetime'].dt.date
        
        # Add some derived features
        ratings_df['year'] = ratings_df['datetime'].dt.year
        ratings_df['month'] = ratings_df['datetime'].dt.month
        ratings_df['day_of_week'] = ratings_df['datetime'].dt.dayofweek
        ratings_df['hour'] = ratings_df['datetime'].dt.hour
        
        print(f"Processed {len(ratings_df)} ratings")
        print(f"Rating range: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
        print(f"Date range: {ratings_df['date'].min()} to {ratings_df['date'].max()}")
        
        return ratings_df
    
    def process_movies(self) -> pd.DataFrame:
        """Process movie data from u.item file"""
        print("Processing movies data...")
        
        # Column names for movie data
        movie_cols = [
            'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'
        ] + self.genre_names
        
        # Load movie data
        movies_file = os.path.join(self.raw_data_dir, "u.item")
        movies_df = pd.read_csv(
            movies_file, 
            sep='|', 
            names=movie_cols,
            header=None,
            encoding='latin-1'
        )
        
        # Clean up movie titles (extract year from title)
        movies_df['title_clean'] = movies_df['title'].copy()
        movies_df['year_from_title'] = movies_df['title'].str.extract(r'\((\d{4})\)')
        movies_df['year_from_title'] = pd.to_numeric(movies_df['year_from_title'], errors='coerce')
        
        # Process release date
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format='%d-%b-%Y', errors='coerce')
        movies_df['release_year'] = movies_df['release_date'].dt.year
        
        # Use year from title if release year is missing
        movies_df['year'] = movies_df['release_year'].fillna(movies_df['year_from_title'])
        
        # Create genre lists for each movie
        genre_columns = self.genre_names
        movies_df['genres'] = movies_df[genre_columns].apply(
            lambda row: [genre for genre, val in zip(genre_columns, row) if val == 1],
            axis=1
        )
        
        # Count of genres per movie
        movies_df['genre_count'] = movies_df[genre_columns].sum(axis=1)
        
        # Most common genre (first one if tie)
        movies_df['primary_genre'] = movies_df['genres'].apply(
            lambda x: x[0] if x else 'unknown'
        )
        
        print(f"Processed {len(movies_df)} movies")
        print(f"Year range: {movies_df['year'].min()} - {movies_df['year'].max()}")
        print(f"Average genres per movie: {movies_df['genre_count'].mean():.2f}")
        
        return movies_df
    
    def process_users(self) -> pd.DataFrame:
        """Process user data from u.user file"""
        print("Processing users data...")
        
        # Column names for user data
        user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        
        # Load user data
        users_file = os.path.join(self.raw_data_dir, "u.user")
        users_df = pd.read_csv(
            users_file, 
            sep='|', 
            names=user_cols,
            header=None
        )
        
        # Create age groups
        users_df['age_group'] = pd.cut(
            users_df['age'], 
            bins=[0, 18, 25, 35, 45, 55, 65, 100],
            labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        )
        
        print(f"Processed {len(users_df)} users")
        print(f"Age range: {users_df['age'].min()} - {users_df['age'].max()}")
        print(f"Gender distribution: {users_df['gender'].value_counts().to_dict()}")
        
        return users_df
    
    def create_rating_matrix(self, ratings_df: pd.DataFrame) -> np.ndarray:
        """Create user-item rating matrix"""
        print("Creating rating matrix...")
        
        # Get unique users and items
        users = sorted(ratings_df['user_id'].unique())
        items = sorted(ratings_df['item_id'].unique())
        
        # Create mapping dictionaries
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        # Create rating matrix
        rating_matrix = np.zeros((len(users), len(items)))
        
        for _, row in ratings_df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            rating_matrix[user_idx, item_idx] = row['rating']
        
        # Save mappings as CSV files
        # User mappings
        user_mapping_df = pd.DataFrame([
            {'user_id': user_id, 'matrix_index': idx} 
            for user_id, idx in user_to_idx.items()
        ])
        user_mapping_df.to_csv(os.path.join(self.output_dir, 'user_mappings.csv'), index=False)
        
        # Item mappings
        item_mapping_df = pd.DataFrame([
            {'item_id': item_id, 'matrix_index': idx} 
            for item_id, idx in item_to_idx.items()
        ])
        item_mapping_df.to_csv(os.path.join(self.output_dir, 'item_mappings.csv'), index=False)
        
        # Also keep the dictionary format for backward compatibility
        mappings = {
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': {idx: user for user, idx in user_to_idx.items()},
            'idx_to_item': {idx: item for item, idx in item_to_idx.items()}
        }
        
        # Save as CSV for easy reading
        mappings_summary_df = pd.DataFrame({
            'mapping_type': ['users', 'items'],
            'total_count': [len(user_to_idx), len(item_to_idx)],
            'min_id': [min(user_to_idx.keys()), min(item_to_idx.keys())],
            'max_id': [max(user_to_idx.keys()), max(item_to_idx.keys())]
        })
        mappings_summary_df.to_csv(os.path.join(self.output_dir, 'mappings_summary.csv'), index=False)
        
        print(f"Created rating matrix of shape: {rating_matrix.shape}")
        print(f"Sparsity: {(rating_matrix == 0).sum() / rating_matrix.size * 100:.2f}%")
        
        return rating_matrix
    
    def create_statistics(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame):
        """Create dataset statistics"""
        print("Creating dataset statistics...")
        
        stats = {
            'dataset_info': {
                'total_ratings': len(ratings_df),
                'total_users': len(users_df),
                'total_movies': len(movies_df),
                'rating_scale': (ratings_df['rating'].min(), ratings_df['rating'].max()),
                'sparsity': 1 - (len(ratings_df) / (len(users_df) * len(movies_df))),
                'avg_ratings_per_user': len(ratings_df) / len(users_df),
                'avg_ratings_per_movie': len(ratings_df) / len(movies_df)
            },
            'rating_distribution': ratings_df['rating'].value_counts().sort_index().to_dict(),
            'user_stats': {
                'age_distribution': users_df['age_group'].value_counts().to_dict(),
                'gender_distribution': users_df['gender'].value_counts().to_dict(),
                'occupation_distribution': users_df['occupation'].value_counts().to_dict()
            },
            'movie_stats': {
                'year_range': (movies_df['year'].min(), movies_df['year'].max()),
                'genre_distribution': {genre: movies_df[genre].sum() for genre in self.genre_names},
                'avg_genres_per_movie': movies_df['genre_count'].mean()
            },
            'temporal_stats': {
                'date_range': (str(ratings_df['date'].min()), str(ratings_df['date'].max())),
                'ratings_by_year': ratings_df['year'].value_counts().sort_index().to_dict(),
                'ratings_by_month': ratings_df['month'].value_counts().sort_index().to_dict()
            }
        }
        
        # Save statistics as CSV files for easy reading
        
        # Dataset info
        dataset_info_df = pd.DataFrame([stats['dataset_info']]).T
        dataset_info_df.columns = ['value']
        dataset_info_df.index.name = 'metric'
        dataset_info_df.to_csv(os.path.join(self.output_dir, 'dataset_info.csv'))
        
        # Rating distribution
        rating_dist_df = pd.DataFrame([
            {'rating': rating, 'count': count, 'percentage': count/stats['dataset_info']['total_ratings']*100}
            for rating, count in sorted(stats['rating_distribution'].items())
        ])
        rating_dist_df.to_csv(os.path.join(self.output_dir, 'rating_distribution.csv'), index=False)
        
        # User demographics
        age_dist_df = pd.DataFrame([
            {'age_group': age_group, 'count': count}
            for age_group, count in stats['user_stats']['age_distribution'].items()
        ])
        age_dist_df.to_csv(os.path.join(self.output_dir, 'age_distribution.csv'), index=False)
        
        gender_dist_df = pd.DataFrame([
            {'gender': gender, 'count': count}
            for gender, count in stats['user_stats']['gender_distribution'].items()
        ])
        gender_dist_df.to_csv(os.path.join(self.output_dir, 'gender_distribution.csv'), index=False)
        
        occupation_dist_df = pd.DataFrame([
            {'occupation': occupation, 'count': count}
            for occupation, count in stats['user_stats']['occupation_distribution'].items()
        ])
        occupation_dist_df.to_csv(os.path.join(self.output_dir, 'occupation_distribution.csv'), index=False)
        
        # Genre distribution
        genre_dist_df = pd.DataFrame([
            {'genre': genre, 'movie_count': count}
            for genre, count in sorted(stats['movie_stats']['genre_distribution'].items(), key=lambda x: x[1], reverse=True)
        ])
        genre_dist_df.to_csv(os.path.join(self.output_dir, 'genre_distribution.csv'), index=False)
        
        # Temporal statistics
        temporal_df = pd.DataFrame([
            {'year': year, 'rating_count': count}
            for year, count in sorted(stats['temporal_stats']['ratings_by_year'].items())
        ])
        temporal_df.to_csv(os.path.join(self.output_dir, 'ratings_by_year.csv'), index=False)
        
        monthly_df = pd.DataFrame([
            {'month': month, 'rating_count': count}
            for month, count in sorted(stats['temporal_stats']['ratings_by_month'].items())
        ])
        monthly_df.to_csv(os.path.join(self.output_dir, 'ratings_by_month.csv'), index=False)
        
        return stats
    
    def load_train_test_splits(self) -> Dict[str, pd.DataFrame]:
        """Load pre-defined train/test splits"""
        print("Loading train/test splits...")
        
        splits = {}
        
        # Load u1-u5 splits (5-fold cross validation)
        for i in range(1, 6):
            train_file = os.path.join(self.raw_data_dir, f"u{i}.base")
            test_file = os.path.join(self.raw_data_dir, f"u{i}.test")
            
            train_df = pd.read_csv(
                train_file, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                header=None
            )
            test_df = pd.read_csv(
                test_file, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                header=None
            )
            
            splits[f'u{i}'] = {'train': train_df, 'test': test_df}
        
        # Load ua/ub splits (10 ratings per user in test set)
        for split_name in ['ua', 'ub']:
            train_file = os.path.join(self.raw_data_dir, f"{split_name}.base")
            test_file = os.path.join(self.raw_data_dir, f"{split_name}.test")
            
            train_df = pd.read_csv(
                train_file, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                header=None
            )
            test_df = pd.read_csv(
                test_file, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                header=None
            )
            
            splits[split_name] = {'train': train_df, 'test': test_df}
        
        print(f"Loaded {len(splits)} train/test splits")
        return splits
    
    def process_all(self) -> None:
        """Process all raw data files"""
        print("=" * 50)
        print("Processing MovieLens 100K Dataset")
        print("=" * 50)
        
        # Load genres first
        self.load_genres()
        
        # Process main data files
        ratings_df = self.process_ratings()
        movies_df = self.process_movies()
        users_df = self.process_users()
        
        # Create rating matrix
        rating_matrix = self.create_rating_matrix(ratings_df)
        
        # Create statistics
        stats = self.create_statistics(ratings_df, movies_df, users_df)
        
        # Load train/test splits
        splits = self.load_train_test_splits()
        
        # Save processed data
        print("\nSaving processed data...")
        
        # Save main dataframes
        ratings_df.to_csv(os.path.join(self.output_dir, 'ratings.csv'), index=False)
        movies_df.to_csv(os.path.join(self.output_dir, 'movies.csv'), index=False)
        users_df.to_csv(os.path.join(self.output_dir, 'users.csv'), index=False)
        
        # Save rating matrix
        np.save(os.path.join(self.output_dir, 'rating_matrix.npy'), rating_matrix)
        
        # Save train/test splits
        for split_name, split_data in splits.items():
            split_data['train'].to_csv(
                os.path.join(self.output_dir, f'{split_name}_train.csv'), 
                index=False
            )
            split_data['test'].to_csv(
                os.path.join(self.output_dir, f'{split_name}_test.csv'), 
                index=False
            )
        
        print(f"All processed data saved to: {self.output_dir}")
        
        # Print summary statistics
        self.print_summary(stats)
    
    def print_summary(self, stats: Dict):
        """Print dataset summary"""
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        
        info = stats['dataset_info']
        print(f"Total ratings: {info['total_ratings']:,}")
        print(f"Total users: {info['total_users']:,}")
        print(f"Total movies: {info['total_movies']:,}")
        print(f"Rating scale: {info['rating_scale'][0]} - {info['rating_scale'][1]}")
        print(f"Sparsity: {info['sparsity']:.2%}")
        print(f"Avg ratings per user: {info['avg_ratings_per_user']:.1f}")
        print(f"Avg ratings per movie: {info['avg_ratings_per_movie']:.1f}")
        
        print(f"\nRating distribution:")
        for rating, count in sorted(stats['rating_distribution'].items()):
            percentage = count / info['total_ratings'] * 100
            print(f"  {rating}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nTop genres:")
        genre_dist = stats['movie_stats']['genre_distribution']
        for genre, count in sorted(genre_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {genre}: {count}")


def main():
    """Main function to run the data processing"""
    processor = MovieLens100KProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()
