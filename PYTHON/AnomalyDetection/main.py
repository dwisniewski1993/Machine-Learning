from Models.DeepLearningModels.CONV import CONVModel
from Models.DeepLearningModels.Forward import FFModel
from Models.DeepLearningModels.GRU import GRUModel
from Models.DeepLearningModels.LSTM import LSTMModel
from Models.MachineLearningModels.IsolationForrest import IsolationForrestModel
from Models.MachineLearningModels.OneClassSVMM import OneClassSVMModel
from Models.Utils import DataHandler, Results
from config import WINDOW_SIZE, EARLY_STOPPING, TENSORBOARD, RETRAIN


def main() -> None:
    """
    Main function. Anomaly/Outliers detection with neural networks architectures, fuzzy logic, one-class svm and
    isolation forrest.
    train set: health data
    validation set: broken data
    :return: None
    """

    # Paths to files with data
    normal_data = r'Datasets/data_normal_small.csv'
    anomaly_file = r'Datasets/data_attk_small.csv'

    # Load and get data
    data_handler = DataHandler(file_normal=normal_data, file_broken=anomaly_file)
    health_data = data_handler.get_normal_dataset()
    anomaly_data = data_handler.get_broken_dataset()
    labels = data_handler.get_broken_labels()

    # One Class SVM
    svm = OneClassSVMModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated',
                           data_labels=labels)
    svm.train()
    svm.score(svm.anomaly_data)

    # Isolation Forest
    iso = IsolationForrestModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated',
                                data_labels=labels)
    iso.train()
    iso.score(iso.anomaly_data)

    # Feed Forward Auto-Encoder
    forward = FFModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                      windows_size=WINDOW_SIZE)
    forward.train(retrain=RETRAIN, tensor_board=TENSORBOARD, early_stopping=EARLY_STOPPING)
    yhat_healthy = forward.score(data=forward.normal_data)
    yhat_broken = forward.score(data=forward.anomaly_data)
    forward.calculate_threshold(health=yhat_healthy)
    forward.anomaly_score(pred=yhat_broken)

    # LSTM Auto-Encoder
    lstm = LSTMModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                     windows_size=WINDOW_SIZE)
    lstm.train(retrain=RETRAIN, tensor_board=TENSORBOARD, early_stopping=EARLY_STOPPING)
    yhat_healthy = lstm.score(data=lstm.normal_data)
    yhat_broken = lstm.score(data=lstm.anomaly_data)
    lstm.calculate_threshold(health=yhat_healthy)
    lstm.anomaly_score(pred=yhat_broken)

    # GRU Auto-Encoder
    gru = GRUModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                   windows_size=WINDOW_SIZE)
    gru.train(retrain=RETRAIN, tensor_board=TENSORBOARD, early_stopping=EARLY_STOPPING)
    yhat_healthy = gru.score(data=gru.normal_data)
    yhat_broken = gru.score(data=gru.anomaly_data)
    gru.calculate_threshold(health=yhat_healthy)
    gru.anomaly_score(pred=yhat_broken)

    # Convolutional Auto-Encoder
    conv = CONVModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                     windows_size=WINDOW_SIZE)
    conv.train(retrain=RETRAIN, tensor_board=TENSORBOARD, early_stopping=EARLY_STOPPING)
    yhat_healthy = conv.score(data=conv.normal_data)
    yhat_broken = conv.score(data=conv.anomaly_data)
    conv.calculate_threshold(health=yhat_healthy)
    conv.anomaly_score(pred=yhat_broken)

    # Calculate results
    Results()


if __name__ == '__main__':
    main()
