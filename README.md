# Movie Recommendation System - Learning Project

A movie recommendation system built from scratch to learn neural networks and recommendation algorithms.

## Project Structure

```
movie-recommendation/
├── data/
│   ├── raw/           # MovieLens 100K dataset
│   └── processed/     # Processed data
├── models/
│   ├── tensorflow/    # TensorFlow implementation
│   └── scratch/       # NumPy implementation
├── ui/                # User interface
├── scratchnet/        # Neural network from scratch
├── utils/
│   └── data_loader.py # Data processing (✅ Complete)
├── tests/             # Unit tests
├── main.py           # Starter script (✅ Complete)
└── requirements.txt   # Dependencies
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the starter script:
```bash
python main.py
```

## Learning Goals

- Build neural networks from scratch using NumPy
- Implement the same model using TensorFlow
- Create a movie recommendation system
- Build an interactive UI
- Learn evaluation metrics for recommendation systems

## Next Steps

1. Implement TensorFlow autoencoder in `models/tensorflow/`
2. Build neural network from scratch in `scratchnet/`
3. Create UI in `ui/`
4. Add evaluation metrics in `utils/metrics.py`

The data loading is already implemented - you can focus on building the models!