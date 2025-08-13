import numpy as np

class EmbeddingLayer:
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