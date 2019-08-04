import tensorflow as tf
import absl.logging as log
from numpy import array, ndarray
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, LSTM
from Models.DeepLearningModels.AbstractDlModel import DLAbstractModel
from config import LSTM_NETWORK, LSTM_CELLS_NUMBER, LSTM_LAYERS_NUMBER, MEAN_ABSOLUTE_ERROR, ADAM_OPTIMIZER


log.set_verbosity(log.INFO)


class LSTMModel(DLAbstractModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str,
                 windows_size: int):
        super().__init__(healthy_data, broken_data, data_labels, dataset_name, windows_size)
        self.model_name = LSTM_NETWORK
        self.reshape_data()
        self.model = self.define_model()

    def define_model(self) -> Sequential:
        """
        Defining the specify LSTM architecteure: activation, number of layers, optimizer and error measure.
        :return: Keras Sequential Model
        """
        if tf.test.is_gpu_available(cuda_only=True):
            rnn_cell = CuDNNLSTM
        else:
            rnn_cell = LSTM

        model = Sequential()
        for i in range(LSTM_LAYERS_NUMBER):
            model.add(rnn_cell(units=LSTM_CELLS_NUMBER, input_shape=(self.window_size, self.dim), return_sequences=True))
        model.add(Dense(units=self.dim, activation='tanh'))
        model.compile(loss=MEAN_ABSOLUTE_ERROR, optimizer=ADAM_OPTIMIZER)
        log.info(f"Defining {self.model_name} neural network architecture...")

        return model
