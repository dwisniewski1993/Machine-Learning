import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from models.AMI import AbstractModelInterface

class Autoencoder(nn.Module):
    """
    PyTorch implementation of a deep autoencoder for dimensionality reduction.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16)
        )
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Returns:
        - reconstructed: Output from the decoder
        - latent: Encoded representation
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class AutoencoderModel(AbstractModelInterface):
    """
    Anomaly detection model using a deep autoencoder.
    """

    def __init__(self, input_dim=None, device='cpu'):
        self.input_dim = input_dim
        self.device = device
        self.model = Autoencoder(input_dim).to(device) if input_dim else None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.threshold = None
        self.centroid = None

    def train(self, data: np.ndarray):
        """
        Trains the autoencoder and computes anomaly threshold.

        Parameters:
        - data: Input data as a NumPy array
        """
        if self.model is None:
            self.input_dim = data.shape[1]
            self.model = Autoencoder(self.input_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        X = torch.tensor(data, dtype=torch.float32).to(self.device)

        best_loss = float('inf')
        patience = 10
        wait = 0

        # Training loop with early stopping
        for epoch in range(200):
            self.optimizer.zero_grad()
            output, latent = self.model(X)

            reconstruction_loss = self.criterion(output, X)
            latent_mean = torch.mean(latent, dim=0)
            latent_reg = torch.norm(latent_mean, p=2)
            loss = reconstruction_loss + 0.001 * latent_reg

            loss.backward()
            self.optimizer.step()

            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Compute centroid and threshold for anomaly detection
        self.model.eval()
        with torch.no_grad():
            _, latent = self.model(X)
            latent_np = latent.cpu().numpy()
            self.centroid = np.mean(latent_np, axis=0)
            dists = np.linalg.norm(latent_np - self.centroid, axis=1)
            self.threshold = np.percentile(dists, 15)

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts anomalies based on distance from latent centroid.

        Parameters:
        - data: Input data as a NumPy array

        Returns:
        - Array of predictions: -1 for anomaly, 1 for normal
        """
        self.model.eval()
        X = torch.tensor(data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, latent = self.model(X)
            latent_np = latent.cpu().numpy()
            dists = np.linalg.norm(latent_np - self.centroid, axis=1)

        return np.where(dists > self.threshold, -1, 1)

    def save(self, path: str):
        """
        Saves model weights and metadata to disk.

        Parameters:
        - path: Base path for saving files
        """
        torch.save(self.model.state_dict(), path + "_weights.pt")
        joblib.dump({
            "input_dim": self.input_dim,
            "centroid": self.centroid,
            "threshold": self.threshold
        }, path + "_meta.pkl")

    def load(self, path: str):
        """
        Loads model weights and metadata from disk.

        Parameters:
        - path: Base path for loading files
        """
        meta = joblib.load(path + "_meta.pkl")
        self.input_dim = meta["input_dim"]
        self.centroid = meta["centroid"]
        self.threshold = meta["threshold"]
        self.model = Autoencoder(self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(path + "_weights.pt", map_location=self.device))
        self.model.eval()
