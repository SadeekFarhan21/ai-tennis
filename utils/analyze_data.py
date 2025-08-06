import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional


class MovieLensAnalyzer:
    """
    Analyzer for the processed MovieLens 100K dataset.
    Provides comprehensive analysis and visualization capabilities.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.rating_matrix: Optional[np.ndarray] = None
        self.mappings: Optional[Dict] = None
        self.stats: Optional[Dict] = None
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load all processed data"""
        print("Loading processed data...")
        
        # Load main dataframes
        self.ratings_df = pd.read_csv(os.path.join(self.data_dir, 'ratings.csv'))
        self.movies_df = pd.read_csv(os.path.join(self.data_dir, 'movies.csv'))
        self.users_df = pd.read_csv(os.path.join(self.data_dir, 'users.csv'))
        
        # Load rating matrix
        self.rating_matrix = np.load(os.path.join(self.data_dir, 'rating_matrix.npy'))
        
        # Load mappings and statistics
        with open(os.path.join(self.data_dir, 'mappings.pkl'), 'rb') as f:
            self.mappings = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'dataset_stats.pkl'), 'rb') as f:
            self.stats = pickle.load(f)
        
        # Convert datetime columns
        self.ratings_df['datetime'] = pd.to_datetime(self.ratings_df['datetime'])
        self.ratings_df['date'] = pd.to_datetime(self.ratings_df['date'])
        self.movies_df['release_date'] = pd.to_datetime(self.movies_df['release_date'])
        
        print("Data loaded successfully!")
        
    def analyze_ratings_distribution(self):
        """Analyze rating patterns"""
        if self.ratings_df is None:
            self.load_data()
            
        print("\\n" + "="*50)
        print("RATING ANALYSIS")
        print("="*50)
        
        # Basic rating statistics
        print(f"Rating Statistics:")
        print(f"Mean: {self.ratings_df['rating'].mean():.2f}")
        print(f"Median: {self.ratings_df['rating'].median():.2f}")
        print(f"Standard Deviation: {self.ratings_df['rating'].std():.2f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Rating Distribution Analysis', fontsize=16)
        
        # Rating distribution
        axes[0, 0].hist(self.ratings_df['rating'], bins=5, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        
        # Ratings over time
        monthly_ratings = self.ratings_df.groupby(self.ratings_df['date'].dt.to_period('M')).size()
        monthly_ratings.plot(ax=axes[0, 1])
        axes[0, 1].set_title('Ratings Over Time')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Number of Ratings')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average rating by hour
        hourly_avg = self.ratings_df.groupby('hour')['rating'].mean()
        axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
        axes[1, 0].set_title('Average Rating by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average Rating')
        axes[1, 0].set_xticks(range(0, 24, 2))
        
        # Day of week ratings
        dow_avg = self.ratings_df.groupby('day_of_week')['rating'].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), dow_avg.values)
        axes[1, 1].set_title('Average Rating by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Rating')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(dow_names)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'rating_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_users(self):
        """Analyze user demographics and behavior"""
        if self.ratings_df is None:
            self.load_data()
            
        print("\\n" + "="*50)
        print("USER ANALYSIS")
        print("="*50)
        
        # User rating statistics
        assert self.ratings_df is not None and self.users_df is not None
        user_stats = self.ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'item_id': 'nunique'
        }).round(2)
        
        user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
        user_stats = user_stats.merge(self.users_df, on='user_id')
        
        print(f"User Statistics:")
        print(f"Most active user: {user_stats['rating_count'].max()} ratings")
        print(f"Least active user: {user_stats['rating_count'].min()} ratings")
        print(f"Average ratings per user: {user_stats['rating_count'].mean():.1f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('User Demographics and Behavior Analysis', fontsize=16)
        
        # Age distribution
        axes[0, 0].hist(user_stats['age'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Number of Users')
        
        # Gender distribution
        gender_counts = user_stats['gender'].value_counts()
        axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Gender Distribution')
        
        # Occupation distribution (top 10)
        occ_counts = user_stats['occupation'].value_counts().head(10)
        axes[0, 2].barh(range(len(occ_counts)), occ_counts.values)
        axes[0, 2].set_title('Top 10 Occupations')
        axes[0, 2].set_yticks(range(len(occ_counts)))
        axes[0, 2].set_yticklabels(occ_counts.index)
        
        # Rating activity distribution
        axes[1, 0].hist(user_stats['rating_count'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('User Activity Distribution')
        axes[1, 0].set_xlabel('Number of Ratings')
        axes[1, 0].set_ylabel('Number of Users')
        
        # Average rating by age
        age_rating = user_stats.groupby('age')['avg_rating'].mean()
        axes[1, 1].scatter(age_rating.index, age_rating.values, alpha=0.6)
        axes[1, 1].set_title('Average Rating vs Age')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Average Rating')
        
        # Rating behavior by gender
        gender_rating = user_stats.groupby('gender')['avg_rating'].mean()
        axes[1, 2].bar(gender_rating.index, gender_rating.values)
        axes[1, 2].set_title('Average Rating by Gender')
        axes[1, 2].set_xlabel('Gender')
        axes[1, 2].set_ylabel('Average Rating')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'user_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return user_stats
        
    def analyze_movies(self):
        """Analyze movie characteristics and popularity"""
        if self.ratings_df is None:
            self.load_data()
            
        print("\\n" + "="*50)
        print("MOVIE ANALYSIS")
        print("="*50)
        
        # Movie rating statistics
        assert self.ratings_df is not None and self.movies_df is not None
        movie_stats = self.ratings_df.groupby('item_id').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_users']
        movie_stats = movie_stats.merge(
            self.movies_df[['movie_id', 'title', 'year', 'primary_genre', 'genre_count']], 
            left_index=True, 
            right_on='movie_id'
        )
        
        print(f"Movie Statistics:")
        print(f"Most rated movie: {movie_stats.loc[movie_stats['rating_count'].idxmax(), 'title']}")
        print(f"  - {movie_stats['rating_count'].max()} ratings")
        print(f"Highest rated movie (min 50 ratings): {movie_stats[movie_stats['rating_count'] >= 50].loc[movie_stats[movie_stats['rating_count'] >= 50]['avg_rating'].idxmax(), 'title']}")
        print(f"  - {movie_stats[movie_stats['rating_count'] >= 50]['avg_rating'].max():.2f} average rating")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Movie Analysis', fontsize=16)
        
        # Movie popularity distribution
        axes[0, 0].hist(movie_stats['rating_count'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Movie Popularity Distribution')
        axes[0, 0].set_xlabel('Number of Ratings')
        axes[0, 0].set_ylabel('Number of Movies')
        axes[0, 0].set_yscale('log')
        
        # Average rating distribution
        axes[0, 1].hist(movie_stats['avg_rating'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Average Rating Distribution')
        axes[0, 1].set_xlabel('Average Rating')
        axes[0, 1].set_ylabel('Number of Movies')
        
        # Movies by release year
        year_counts = movie_stats['year'].value_counts().sort_index()
        axes[0, 2].plot(year_counts.index, year_counts.values)
        axes[0, 2].set_title('Movies by Release Year')
        axes[0, 2].set_xlabel('Year')
        axes[0, 2].set_ylabel('Number of Movies')
        
        # Genre popularity
        assert self.ratings_df is not None and self.movies_df is not None
        genre_ratings = self.ratings_df.merge(
            self.movies_df[['movie_id', 'primary_genre']], 
            left_on='item_id', 
            right_on='movie_id'
        )
        genre_avg = genre_ratings.groupby('primary_genre')['rating'].mean().sort_values(ascending=False)
        
        axes[1, 0].barh(range(len(genre_avg)), genre_avg.values)
        axes[1, 0].set_title('Average Rating by Primary Genre')
        axes[1, 0].set_yticks(range(len(genre_avg)))
        axes[1, 0].set_yticklabels(genre_avg.index)
        
        # Popularity vs Quality
        # Filter movies with at least 10 ratings for meaningful comparison
        popular_movies = movie_stats[movie_stats['rating_count'] >= 10]
        axes[1, 1].scatter(popular_movies['rating_count'], popular_movies['avg_rating'], alpha=0.6)
        axes[1, 1].set_title('Popularity vs Quality')
        axes[1, 1].set_xlabel('Number of Ratings (Popularity)')
        axes[1, 1].set_ylabel('Average Rating (Quality)')
        axes[1, 1].set_xscale('log')
        
        # Genre count distribution
        axes[1, 2].hist(movie_stats['genre_count'], bins=range(1, 8), alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Number of Genres per Movie')
        axes[1, 2].set_xlabel('Number of Genres')
        axes[1, 2].set_ylabel('Number of Movies')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'movie_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return movie_stats
    
    def analyze_genre_preferences(self):
        """Analyze genre preferences across different user groups"""
        if self.ratings_df is None:
            self.load_data()
            
        print("\\n" + "="*50)
        print("GENRE PREFERENCE ANALYSIS")
        print("="*50)
        
        # Create genre preference matrix
        assert self.ratings_df is not None and self.movies_df is not None and self.users_df is not None
        genre_ratings = self.ratings_df.merge(
            self.movies_df[['movie_id', 'primary_genre']], 
            left_on='item_id', 
            right_on='movie_id'
        ).merge(
            self.users_df[['user_id', 'age', 'gender', 'occupation']], 
            on='user_id'
        )
        
        # Gender preferences
        gender_genre = genre_ratings.groupby(['gender', 'primary_genre'])['rating'].mean().unstack()
        
        # Age group preferences  
        genre_ratings['age_group'] = pd.cut(
            genre_ratings['age'], 
            bins=[0, 25, 35, 50, 100], 
            labels=['<25', '25-34', '35-49', '50+']
        )
        age_genre = genre_ratings.groupby(['age_group', 'primary_genre'])['rating'].mean().unstack()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre Preferences Analysis', fontsize=16)
        
        # Gender preference heatmap
        sns.heatmap(gender_genre, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=axes[0, 0])
        axes[0, 0].set_title('Average Rating by Gender and Genre')
        axes[0, 0].set_ylabel('Gender')
        
        # Age group preference heatmap
        sns.heatmap(age_genre, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=axes[0, 1])
        axes[0, 1].set_title('Average Rating by Age Group and Genre')
        axes[0, 1].set_ylabel('Age Group')
        
        # Most popular genres by count
        genre_counts = genre_ratings['primary_genre'].value_counts()
        axes[1, 0].barh(range(len(genre_counts)), genre_counts.values)
        axes[1, 0].set_title('Genre Popularity (by number of ratings)')
        axes[1, 0].set_yticks(range(len(genre_counts)))
        axes[1, 0].set_yticklabels(genre_counts.index)
        
        # Average rating by genre
        genre_avg_rating = genre_ratings.groupby('primary_genre')['rating'].mean().sort_values(ascending=False)
        axes[1, 1].barh(range(len(genre_avg_rating)), genre_avg_rating.values)
        axes[1, 1].set_title('Average Rating by Genre')
        axes[1, 1].set_yticks(range(len(genre_avg_rating)))
        axes[1, 1].set_yticklabels(genre_avg_rating.index)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'genre_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def find_top_recommendations(self, n=10):
        """Find top movie recommendations based on different criteria"""
        if self.ratings_df is None:
            self.load_data()
            
        print("\\n" + "="*50)
        print("TOP MOVIE RECOMMENDATIONS")
        print("="*50)
        
        # Movie statistics
        assert self.ratings_df is not None and self.movies_df is not None
        movie_stats = self.ratings_df.groupby('item_id').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        movie_stats.columns = ['rating_count', 'avg_rating', 'rating_std']
        
        # Merge with movie info
        movie_stats = movie_stats.merge(
            self.movies_df[['movie_id', 'title', 'year', 'primary_genre']], 
            left_index=True, 
            right_on='movie_id'
        )
        
        # Most popular movies
        print(f"\\nMost Popular Movies (by number of ratings):")
        popular = movie_stats.nlargest(n, 'rating_count')[['title', 'year', 'rating_count', 'avg_rating']]
        for i, (_, row) in enumerate(popular.iterrows(), 1):
            print(f"{i:2d}. {row['title']} ({row['year']:.0f}) - {row['rating_count']} ratings, {row['avg_rating']:.2f} avg")
        
        # Highest rated movies (with minimum rating threshold)
        min_ratings = 50
        print(f"\\nHighest Rated Movies (min {min_ratings} ratings):")
        highly_rated = movie_stats[movie_stats['rating_count'] >= min_ratings].nlargest(n, 'avg_rating')
        for i, (_, row) in enumerate(highly_rated[['title', 'year', 'rating_count', 'avg_rating']].iterrows(), 1):
            print(f"{i:2d}. {row['title']} ({row['year']:.0f}) - {row['avg_rating']:.2f} avg, {row['rating_count']} ratings")
        
        # Hidden gems (high rating, low popularity)
        print(f"\\nHidden Gems (high quality, less known):")
        min_quality = 4.0
        max_popularity = 30
        hidden_gems = movie_stats[
            (movie_stats['avg_rating'] >= min_quality) & 
            (movie_stats['rating_count'] <= max_popularity) &
            (movie_stats['rating_count'] >= 10)  # Still need some ratings for reliability
        ].nlargest(n, 'avg_rating')
        for i, (_, row) in enumerate(hidden_gems[['title', 'year', 'rating_count', 'avg_rating']].iterrows(), 1):
            print(f"{i:2d}. {row['title']} ({row['year']:.0f}) - {row['avg_rating']:.2f} avg, {row['rating_count']} ratings")
    
    def generate_full_report(self):
        """Generate comprehensive analysis report"""
        print("\\n" + "="*60)
        print("MOVIELENS 100K DATASET - COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Load data if not already loaded
        if self.ratings_df is None:
            self.load_data()
        
        # Run all analyses
        self.analyze_ratings_distribution()
        user_stats = self.analyze_users()
        movie_stats = self.analyze_movies()
        self.analyze_genre_preferences()
        self.find_top_recommendations()
        
        # Save processed statistics
        analysis_results = {
            'user_statistics': user_stats.to_dict(),
            'movie_statistics': movie_stats.to_dict(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.data_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(analysis_results, f)
        
        print(f"\\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"All visualizations and results saved to: {self.data_dir}")
        print("Files generated:")
        print("- rating_analysis.png")
        print("- user_analysis.png") 
        print("- movie_analysis.png")
        print("- genre_analysis.png")
        print("- analysis_results.pkl")


def main():
    """Main function to run the analysis"""
    analyzer = MovieLensAnalyzer()
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()
