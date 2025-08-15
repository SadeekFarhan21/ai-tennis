import numpy as np


class EmbeddingLayer:
    """Embedding layer for user/item representations"""
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = np.random.normal(0, 0.01, (num_embeddings, embedding_dim))
        self.ids = None
    
    def forward(self, ids):
        self.ids = ids
        return self.weight[ids]
    
    def backward(self, grad_out):
        grad_W = np.zeros_like(self.weight)
        np.add.at(grad_W, self.ids, grad_out)
        return grad_W
    
"""
NeuralCF connects all of that together
Sequential dictates the number of hidden layers 
DenseLayers count the number of neurons
ReLU, AdamOptimizer they make the individual neurons
"""

class DenseLayer:
    """Dense layer for fully connected neural networks"""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.weight = np.random.normal(0, 0.01, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weight + self.bias

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        self.weight_grad = self.x.T @ grad_y
        self.bias_grad = grad_y.sum(0)
        grad_x = grad_y @ self.weight.T
        return grad_x








class ReLU:
    def __init__(self) -> None:
        pass

class Sequential:
    def __init__(self, *layers) -> None:
        self.layers = layers


class AdamOptimizer:
    def __init__(self) -> None:
        pass
    
class NeuralCFModel:
    def __init__(self) -> None:
        pass