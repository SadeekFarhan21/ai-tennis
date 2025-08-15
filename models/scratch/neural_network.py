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
        self.z = None
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.z = z
        return np.maximum(0, z)
    
    def backward(self, grad_h: np.ndarray) -> np.ndarray:
        return grad_h * (self.z > 0)


class Sequential:
    def __init__(self, *layers) -> None:
        self.layers = layers
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y


class AdamOptimizer:
    def __init__(self, params: Dict, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize momentum and variance for each parameter
        self.m = {}
        self.v = {}
        for name, param in params.items():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
    
    def step(self, params: Dict, grads: Dict):
        self.t += 1
        for name in params.keys():
            if name in grads and grads[name] is not None:
                # Update momentum and variance
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grads[name] ** 2)
                
                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    
class NeuralCFModel:
    def __init__(self) -> None:
        pass