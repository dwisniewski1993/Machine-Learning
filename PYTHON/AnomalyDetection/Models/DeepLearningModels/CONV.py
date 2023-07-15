import absl.logging as log
from numpy import array, ndarray
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Sequential

from Models.DeepLearningModels.AbstractDlModel import DLAbstractModel
from Models.exeptions import InvalidShapes
from config import CONV_CELLS_NUMBER, CONV_KERNEL_SIZE, CONV_POOL_SIZE, CONV_NETWORK, MEAN_ABSOLUTE_ERROR, \
    ADAM_OPTIMIZER, SINGLE_UNIT

log.set_verbosity(log.INFO)


class CONVModel(DLAbstractModel):
    def __init__(self, healthy_data: ndarray, broken_data: ndarray, data_labels: array, dataset_name: str,
                 windows_size: int):
        super().__init__(healthy_data, broken_data, data_labels, dataset_name, windows_size)
        self.model_name = CONV_NETWORK
        self.reshape_data()
        self.model = self.define_model()

    def define_model(self) -> Sequential:
        """
        Define the specific Conv2D architecture: activation, number of layers, optimizer, and error measure.
        :return: Keras Sequential Model
        """
        model = Sequential()
        model.add(Conv2D(filters=CONV_CELLS_NUMBER, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same',
                         input_shape=[self.window_size, self.dim, SINGLE_UNIT]))
        model.add(MaxPooling2D(CONV_POOL_SIZE))

        model.add(Conv2D(filters=CONV_CELLS_NUMBER, kernel_size=CONV_KERNEL_SIZE, activation='relu', padding='same'))
        model.add(UpSampling2D(CONV_POOL_SIZE))
        model.add(Conv2D(SINGLE_UNIT, kernel_size=CONV_KERNEL_SIZE, activation='sigmoid', padding='same'))

        model.compile(optimizer=ADAM_OPTIMIZER, loss=MEAN_ABSOLUTE_ERROR)
        log.info(f"Defining {self.model_name} network architecture...")
        return model

    def reshape_data(self) -> None:
        """
        Reshape data for regression.
        :return: None
        """
        normal_samples = len(self.normal_data)
        normal_dim = len(self.normal_data[0])

        try:
            self.normal_data.shape = (int(normal_samples / self.window_size), self.window_size, normal_dim, SINGLE_UNIT)
        except InvalidShapes:
            raise InvalidShapes("Something is wrong with the dataset shapes.")

        anomaly_samples = len(self.anomaly_data)
        anomaly_dim = len(self.anomaly_data[0])
        try:
            self.anomaly_data.shape = (int(anomaly_samples / self.window_size), self.window_size, anomaly_dim,
                                       SINGLE_UNIT)
        except InvalidShapes:
            raise InvalidShapes("Something is wrong with the dataset shapes.")

        self.dim = self.normal_data.shape[2]
        self.samples = self.normal_data.shape[0]
