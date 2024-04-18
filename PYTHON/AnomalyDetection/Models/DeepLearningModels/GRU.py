import absl.logging as log
from keras.layers import Input
from keras.models import Sequential
from numpy import array, ndarray
from tensorflow.keras.layers import Dense, GRU

from Models.DeepLearningModels.AbstractDlModel import DLAbstractModel
from config import GRU_NETWORK, GRU_CELLS_NUMBER, GRU_LAYERS_NUMBER, MEAN_ABSOLUTE_ERROR, ADAM_OPTIMIZER

log.set_verbosity(log.INFO)


class GRUModel(DLAbstractModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str,
                 windows_size: int) -> None:
        """
        Initialize the GRUModel class.

        Args:
            healthy_data (ndarray): Healthy data for training.
            broken_data (ndarray): Data with anomalies to detect.
            data_labels (array): Data labels.
            dataset_name (str): Name of the dataset.
            windows_size (int): Step in time per example.
        """
        super().__init__(healthy_data, broken_data, data_labels, dataset_name, windows_size)
        self.model_name = GRU_NETWORK
        self.reshape_data()
        self.model = self.define_model()

    def define_model(self) -> Sequential:
        """
        Define the GRU neural network architecture: activation functions, number of layers,
        optimizer, and error measure.

        Returns:
            Sequential: Keras Sequential model.
        """
        rnn_cell = GRU

        model = Sequential([
            Input(shape=(self.window_size, self.dim)),
            *[rnn_cell(units=GRU_CELLS_NUMBER, return_sequences=True) for _ in range(GRU_LAYERS_NUMBER)],
            Dense(units=self.dim, activation='tanh')
        ])
        model.compile(loss=MEAN_ABSOLUTE_ERROR, optimizer=ADAM_OPTIMIZER)
        log.info(f"Defining {self.model_name} neural network architecture...")

        return model
