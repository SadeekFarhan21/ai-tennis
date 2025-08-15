# Movie Recommendation System - Streamlit UI
# This app allows users to rate movies and get recommendations

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.simple_data_loader import SimpleMovieLensLoader
from models.pytorch.matrix_factorization import MatrixFactorization, adapt_new_user, score_all_items_for_user
from models.scratch.scratch_integration import (
    load_scratch_model, adapt_new_user_scratch, score_all_items_for_user_scratch, 
    get_scratch_model_info
)
import torch
import numpy as np
import pandas as pd

# App title and setup
st.title("🎬 Movie Recommendation System")
st.markdown("Rate some movies and get personalized recommendations!")

# Sidebar for model selection
st.sidebar.header("🤖 Model Selection")
model_choice = st.sidebar.radio(
    "Choose recommendation model:",
    ["PyTorch Matrix Factorization", "Scratch Neural Network"],
    help="Select which model to use for generating recommendations"
)

# Display model info
st.sidebar.header("ℹ️ Model Info")
if model_choice == "PyTorch Matrix Factorization":
    st.sidebar.markdown("""
    **Matrix Factorization (PyTorch)**
    - Framework: PyTorch
    - Architecture: User/Item embeddings + biases
    - Embedding dimensions: 64
    - Fast training and inference
    """)
else:
    scratch_info = get_scratch_model_info()
    st.sidebar.markdown(f"""
    **{scratch_info['name']}**
    - Framework: {scratch_info['framework']}
    - Type: {scratch_info['type']}
    - {scratch_info['description']}
    """)
    with st.sidebar.expander("Architecture Details"):
        for arch in scratch_info['architecture']:
            st.markdown(f"• {arch}")
    with st.sidebar.expander("Features"):
        for feature in scratch_info['features']:
            st.markdown(f"• {feature}")

# Sidebar for model info (kept for backward compatibility)
st.sidebar.header("Model Info")

# Load data and model (with caching)
@st.cache_data
def load_data_and_models():
    """Load movie data and both trained models with caching"""
    # Load movie data
    loader = SimpleMovieLensLoader()
    data = loader.process_data(split_id=1, normalize=True)
    
    # Get movie information
    movies_df = data['movies_df']
    movie_mapping = data['movie_mapping']
    
    # Create reverse mapping (index -> movie_id)
    reverse_movie_mapping = {idx: movie_id for movie_id, idx in movie_mapping.items()}
    
    # Create movie selection dataframe
    movie_options = []
    for idx, movie_id in reverse_movie_mapping.items():
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            # Get genres (columns 5-23 are genre columns)
            genres = []
            for i in range(5, 24):
                if movie_info.iloc[0].iloc[i] == 1:
                    genre_name = movies_df.columns[i]
                    genres.append(genre_name.replace('genre', ''))
            # Map genre numbers to names
            genre_names = {
                '1': 'unknown', '2': 'action', '3': 'adventure', '4': 'animation',
                '5': 'children', '6': 'comedy', '7': 'crime', '8': 'documentary',
                '9': 'drama', '10': 'fantasy', '11': 'film-noir', '12': 'horror',
                '13': 'musical', '14': 'mystery', '15': 'romance', '16': 'sci-fi',
                '17': 'thriller', '18': 'war', '19': 'western'
            }
            genre_names_list = [genre_names.get(g, g) for g in genres if g is not None]
            genre_str = ', '.join(genre_names_list) if genre_names_list else 'Unknown'
            movie_options.append({
                'movie_id': movie_id,
                'title': title,
                'genres': genre_str,
                'index': idx
            })
    
    movie_df = pd.DataFrame(movie_options)
    
    # Load PyTorch model
    n_users = data['train_matrix'].shape[0]
    n_items = data['train_matrix'].shape[1]
    
    pytorch_model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=64
    )
    
    # Load trained PyTorch weights
    pytorch_model_path = 'models/pytorch/trained_mf_model.pth'
    pytorch_loaded = False
    
    if os.path.exists(pytorch_model_path):
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_model.eval()
        pytorch_loaded = True
    
    # Load Scratch model
    scratch_model = None
    scratch_loaded = False
    
    try:
        scratch_model = load_scratch_model()
        scratch_loaded = True
    except FileNotFoundError:
        pass  # Will show warning in UI
    
    return data, pytorch_model, scratch_model, movie_df, pytorch_loaded, scratch_loaded

# Load data and models
with st.spinner("Loading data and models..."):
    data, pytorch_model, scratch_model, movie_df, pytorch_loaded, scratch_loaded = load_data_and_models()

# Show model loading status
if model_choice == "PyTorch Matrix Factorization":
    if pytorch_loaded:
        st.success("✅ Loaded trained PyTorch Matrix Factorization model!")
    else:
        st.warning("⚠️ No trained PyTorch model found. Using random weights.")
        
else:  # Scratch Neural Network
    if scratch_loaded:
        st.success("✅ Loaded trained Scratch Neural Network model!")
    else:
        st.error("❌ No trained Scratch model found. Please train the model first.")
        st.info("💡 Run the training script: `python models/scratch/train_scratch_nn.py`")
        st.stop()

# Movie rating section
st.header("📝 Rate Some Movies")
st.markdown("Rate 5-10 movies to get personalized recommendations:")

# Create rating interface
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
user_ratings = st.session_state.user_ratings
num_movies_to_rate = st.slider("Number of movies to rate:", 1, 40, 10)

# Get random sample of movies for rating (store in session state)
if 'num_movies_to_rate' not in st.session_state or st.session_state.num_movies_to_rate != num_movies_to_rate:
    st.session_state.num_movies_to_rate = num_movies_to_rate
    st.session_state.user_ratings = {}  # Reset ratings when changing number of movies
    # Generate new random sample only when slider changes
    st.session_state.sample_movies = movie_df.sample(n=min(num_movies_to_rate, len(movie_df)))

# Use stored movies (don't regenerate on every interaction)
sample_movies = st.session_state.sample_movies

# Create rating interface
for _, movie in sample_movies.iterrows():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{movie['title']}** ({movie['genres']})")
    with col2:
        rating = st.selectbox(
            f"Rate {movie['title']}",
            ["No Rating", "1", "2", "3", "4", "5"],
            key=f"rating_{movie['movie_id']}"
        )
        if rating != "No Rating":
            st.session_state.user_ratings[movie['index']] = float(rating)
        elif movie['index'] in st.session_state.user_ratings:
            del st.session_state.user_ratings[movie['index']]

# Recommendation button
if st.button("🎯 Get Recommendations", type="primary"):
    if len(st.session_state.user_ratings) < 3:
        st.error("Please rate at least 3 movies to get recommendations!")
    else:
        with st.spinner("Generating recommendations..."):
            # Get user ratings
            rated_item_ids = list(st.session_state.user_ratings.keys())
            ratings = list(st.session_state.user_ratings.values())
            
            if model_choice == "PyTorch Matrix Factorization":
                # PyTorch Matrix Factorization approach
                user_embedding, user_bias = adapt_new_user(
                    model=pytorch_model,
                    rated_item_ids=rated_item_ids,
                    ratings=ratings,
                    embedding_dim=64,
                    lr=0.01,
                    steps=200,
                    weight_decay=1e-2
                )
                
                # Score all items for this user
                raw_predictions = score_all_items_for_user(pytorch_model, user_embedding, user_bias)
                raw_predictions = raw_predictions.numpy()
                
            else:  # Scratch Neural Network
                if scratch_model is None:
                    st.error("❌ Scratch model not loaded. Cannot generate recommendations.")
                    st.stop()
                    
                # Scratch Neural Network approach
                user_embedding, user_bias = adapt_new_user_scratch(
                    model=scratch_model,
                    rated_item_ids=rated_item_ids,
                    ratings=ratings,
                    embedding_dim=64,
                    lr=0.01,
                    steps=200,
                    weight_decay=1e-2
                )
                
                # Score all items for this user
                raw_predictions = score_all_items_for_user_scratch(scratch_model, user_embedding, user_bias)
            
            # Debug: Check raw prediction range
            st.write(f"Debug - Raw predictions: min={raw_predictions.min():.3f}, max={raw_predictions.max():.3f}, mean={raw_predictions.mean():.3f}")
            st.write(f"Debug - Model used: {model_choice}")
            
            # Get top recommendations using raw scores (no clipping for ranking)
            # Exclude movies the user already rated
            rated_indices = set(st.session_state.user_ratings.keys())
            candidate_indices = [i for i in range(len(raw_predictions)) if i not in rated_indices]
            candidate_predictions = [(i, raw_predictions[i]) for i in candidate_indices]
            candidate_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Clip only for display (stars)
            display_predictions = np.clip(raw_predictions, 1.0, 5.0)
            
            # Display top 10 recommendations
            st.header("🎬 Your Top Recommendations")
            
            for i, (movie_idx, raw_rating) in enumerate(candidate_predictions[:10]):
                # Find movie info
                movie_info = movie_df[movie_df['index'] == movie_idx]
                if not movie_info.empty:
                    movie = movie_info.iloc[0]
                    
                    # Use clipped rating for display only
                    display_rating = display_predictions[movie_idx]
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{i+1}. {movie['title']}**")
                        st.caption(f"Genres: {movie['genres']}")
                    with col2:
                        st.write(f"Predicted Rating: **{display_rating:.1f}/5**")
                        st.caption(f"Raw score: {raw_rating:.2f}")
                    with col3:
                        if display_rating >= 4.0:
                            st.success("⭐⭐⭐⭐⭐")
                        elif display_rating >= 3.0:
                            st.info("⭐⭐⭐⭐")
                        elif display_rating >= 2.0:
                            st.warning("⭐⭐⭐")
                        else:
                            st.error("⭐⭐")

# Model information
st.sidebar.header("📋 About")
if model_choice == "PyTorch Matrix Factorization":
    st.sidebar.markdown("""
    This app uses **Matrix Factorization (PyTorch)** to learn user preferences and make movie recommendations.

    **How it works:**
    1. Rate some movies (5-10 recommended)
    2. The model adapts to your taste profile
    3. Predicts ratings for movies you haven't seen
    4. Shows top recommendations
    """)

    st.sidebar.markdown("""
    **Architecture:**
    - User embeddings: 64 dimensions
    - Item embeddings: 64 dimensions
    - User/item biases for personalization
    - Fast adaptation for new users
    """)
else:
    st.sidebar.markdown("""
    This app uses a **Neural Network built from scratch** to learn user preferences and make movie recommendations.

    **How it works:**
    1. Rate some movies (5-10 recommended)
    2. The neural network adapts to your taste profile
    3. Predicts ratings using collaborative filtering
    4. Shows top recommendations
    """)

    st.sidebar.markdown("""
    **Architecture:**
    - User embeddings: 64 dimensions
    - Item embeddings: 64 dimensions
    - Hidden layer: 128 neurons (ReLU)
    - Pure NumPy implementation
    """)

# Footer
st.markdown("---")
if model_choice == "PyTorch Matrix Factorization":
    st.markdown("*Built with Streamlit and PyTorch*")
else:
    st.markdown("*Built with Streamlit and NumPy (Neural Network from Scratch)*") 