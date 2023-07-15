import absl.logging as log
import tensorflow as tf
from numpy import array, ndarray
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, LSTM
from tensorflow.python.keras.models import Sequential

from Models.DeepLearningModels.AbstractDlModel import DLAbstractModel
from config import LSTM_NETWORK, LSTM_CELLS_NUMBER, LSTM_LAYERS_NUMBER, MEAN_ABSOLUTE_ERROR, ADAM_OPTIMIZER

log.set_verbosity(log.INFO)


class LSTMModel(DLAbstractModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str,
                 windows_size: int) -> None:
        """
        LSTM-based deep learning model for anomaly detection.

        :param healthy_data: Healthy data for training
        :param broken_data: Broken data with anomalies to detect
        :param data_labels: Data labels
        :param dataset_name: Name of the dataset
        :param windows_size: Window size for input data
        """
        super().__init__(healthy_data, broken_data, data_labels, dataset_name, windows_size)
        self.model_name = LSTM_NETWORK
        self.reshape_data()
        self.model = self.define_model()

    def define_model(self) -> Sequential:
        """
        Define the architecture of the LSTM model.

        :return: Keras Sequential model
        """
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            rnn_cell = CuDNNLSTM
        else:
            rnn_cell = LSTM

        model = Sequential()
        for _ in range(LSTM_LAYERS_NUMBER):
            model.add(rnn_cell(units=LSTM_CELLS_NUMBER, input_shape=(self.window_size, self.dim),
                               return_sequences=True))
        model.add(Dense(units=self.dim, activation='tanh'))
        model.compile(loss=MEAN_ABSOLUTE_ERROR, optimizer=ADAM_OPTIMIZER)
        log.info(f"Defining {self.model_name} neural network architecture...")

        return model
