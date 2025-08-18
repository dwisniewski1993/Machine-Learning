import numpy as np
from models.AutoencoderModel import AutoencoderModel
from models.KMeansModel import KMeansModel
from models.DBSSCANModel import DBSCANModel
from models.cluster_ensemble_model import ClusterEnsembleModel
from models.PCAModel import PCAModel

def test_autoencoder_model():
    """
    Tests the AutoencoderModel for basic functionality:
    - Training on synthetic data
    - Producing valid anomaly predictions
    """
    x = np.random.rand(100, 50)
    model = AutoencoderModel(input_dim=50)
    model.train(x)
    preds = model.score(x)
    assert len(preds) == 100
    assert all(p in [-1, 1] for p in preds)

def test_kmeans_model():
    """
    Tests the KMeansModel for basic functionality:
    - Training on synthetic data
    - Producing valid anomaly predictions
    """
    x = np.random.rand(100, 50)
    model = KMeansModel(n_clusters=2)
    model.train(x)
    preds = model.score(x)
    assert len(preds) == 100
    assert all(p in [-1, 1] for p in preds)

def test_dbscan_model():
    """
    Tests the DBSCANModel for basic functionality:
    - Training on synthetic data
    - Producing valid anomaly predictions
    """
    x = np.random.rand(100, 50)
    model = DBSCANModel()
    model.train(x)
    preds = model.score(x)
    assert len(preds) == 100
    assert all(p in [-1, 1] for p in preds)

def test_pca_model():
    """
    Tests the PCAModel for dimensionality reduction:
    - Training on high-dimensional data
    - Producing reduced output with correct shape
    """
    x = np.random.rand(100, 100)
    model = PCAModel(n_components=50)
    model.train(x)
    x_reduced = model.score(x)
    assert x_reduced.shape == (100, 50)

def test_cluster_ensemble_model():
    """
    Tests the ClusterEnsembleModel:
    - Combines predictions from three models
    - Produces valid ensemble predictions
    """
    x = np.random.rand(100, 50)
    ae = AutoencoderModel(input_dim=50); ae.train(x)
    km = KMeansModel(n_clusters=2); km.train(x)
    db = DBSCANModel(); db.train(x)
    weights = {"ae": 0.4, "kmeans": 0.3, "dbscan": 0.3}
    ensemble = ClusterEnsembleModel(ae, km, db, weights)
    preds = ensemble.score(x)
    assert len(preds) == 100
    assert all(p in [-1, 1] for p in preds)

