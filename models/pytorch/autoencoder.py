import torch

class Autoencoder(torch.nn.Module):
    def __init__(self, input_size=1682, hidden_size=100):
        self.encoder = torch.nn.Linear(input_size, hidden_size)
        self.decoder = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return decoded
    
