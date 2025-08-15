# Movie Recommendation System - Learning Project

A movie recommendation system built from scratch to learn neural networks and recommendation algorithms.

## Project Structure

```
movie-recommendation/
├── data/
│   ├── raw/           # MovieLens 100K dataset (official splits)
│   └── processed/     # Processed data matrices
├── models/
│   ├── tensorflow/    # TensorFlow implementation
│   └── scratch/       # NumPy implementation
├── ui/                # User interface
├── scratchnet/        # Neural network from scratch
├── utils/
│   ├── data_loader.py      # Original complex loader
│   └── simple_data_loader.py # Simplified loader (✅ Recommended)
├── tests/             # Unit tests
├── main.py           # Original starter script
├── main_simple.py    # Simplified starter script (✅ Recommended)
└── requirements.txt   # Dependencies
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simplified starter script:
```bash
python main_simple.py
```

## Data Pipeline

### ✅ **Simplified Pipeline (Recommended)**
- Uses **official MovieLens pre-split files** (`u1.base`/`u1.test`)
- **No data loss** - uses the exact splits researchers have used for decades
- **Simple and reliable** - just loads and converts to matrices
- **Optional normalization** - subtracts user means for better model performance

### 📊 **Dataset Information**
- **Official splits**: 5-fold cross-validation (u1-u5)
- **Train/Test**: 80,000 train / 20,000 test ratings
- **Users**: 943 users
- **Movies**: 1,682 movies with genres
- **Ratings**: 1-5 scale

## Learning Goals

- Build neural networks from scratch using NumPy
- Implement the same model using TensorFlow/PyTorch
- Create a movie recommendation system
- Build an interactive UI
- Learn evaluation metrics for recommendation systems

## Next Steps

1. **Implement PyTorch autoencoder** in `models/pytorch/`
2. **Build neural network from scratch** in `scratchnet/`
3. **Create UI** in `ui/`
4. **Add evaluation metrics** in `utils/metrics.py`

The simplified data loading is complete - you can focus on building the models!