import absl.logging as log
from numpy import array, ndarray
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from Models.DeepLearningModels.AbstractDlModel import DLAbstractModel
from config import MEAN_ABSOLUTE_ERROR, ADAM_OPTIMIZER, FORWARD_NETWORK, PRIMARY_UNITS_SIZE, SECONDARY_UNITS_SIZE, \
    TERTIARY_UNITS_SIZE, QUATERNARY_UNITS_SIZE

log.set_verbosity(log.INFO)


class FFModel(DLAbstractModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str,
                 windows_size: int) -> None:
        """
        Initialize the FFModel class.

        Args:
            healthy_data (ndarray): Healthy data for training.
            broken_data (ndarray): Data with anomalies to detect.
            data_labels (array): Data labels.
            dataset_name (str): Name of the dataset.
            windows_size (int): Step in time per example.
        """
        super().__init__(healthy_data, broken_data, data_labels, dataset_name, windows_size)
        self.model_name = FORWARD_NETWORK
        self.reshape_data()
        self.model = self.define_model()

    def define_model(self) -> Sequential:
        """
        Define the FeedForward neural network architecture: activation functions, number of layers,
        optimizer, and error measure.

        Returns:
            Sequential: Keras Sequential model.
        """
        log.info('Defining FeedForward Autoencoder neural network architecture...')
        model = Sequential()
        model.add(Dense(PRIMARY_UNITS_SIZE, activation='tanh', activity_regularizer=regularizers.l1(10e-5),
                        input_shape=(self.window_size, self.dim)))
        model.add(Dense(SECONDARY_UNITS_SIZE, activation='relu'))
        model.add(Dense(TERTIARY_UNITS_SIZE, activation='relu'))
        model.add(Dense(QUATERNARY_UNITS_SIZE, activation='relu'))

        model.add(Dense(QUATERNARY_UNITS_SIZE, activation='relu'))
        model.add(Dense(TERTIARY_UNITS_SIZE, activation='relu'))
        model.add(Dense(SECONDARY_UNITS_SIZE, activation='relu'))
        model.add(Dense(PRIMARY_UNITS_SIZE, activation='relu'))
        model.add(Dense(self.dim, activation='relu'))

        model.compile(optimizer=ADAM_OPTIMIZER, loss=MEAN_ABSOLUTE_ERROR)
        return model
