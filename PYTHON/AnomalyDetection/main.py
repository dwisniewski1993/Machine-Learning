from Models.DeepLearningModels.CONV import CONVModel
from Models.DeepLearningModels.Forward import FFModel
from Models.DeepLearningModels.GRU import GRUModel
from Models.DeepLearningModels.LSTM import LSTMModel
from Models.FuzzyLogicModel.Fuzzy import FuzzyModel
from Models.MachineLearningModels.IsolationForrest import IsolationForrestModel
from Models.MachineLearningModels.OneClassSVMM import OneClassSVMModel
from Models.Utils import DataHandler, Results
from config import WINDOW_SIZE


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
    health_data = data_handler.get_dataset_normal()
    anomaly_data = data_handler.get_dataset_broken()
    labels = data_handler.get_broken_labels()

    # Fuzzy Time Series
    fts = FuzzyModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels)
    fts.train()
    yhat_healthy = fts.score(data=fts.get_normal_data())
    yhat_broken = fts.score(data=fts.get_anomaly_data())
    fts.calculate_threshold(health=yhat_healthy)
    fts.anomaly_score(pred=yhat_broken)

    # One Class SVM
    svm = OneClassSVMModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated',
                           data_labels=labels)
    svm.train()
    svm.score(svm.anomaly_data)

    # Isolation Forrest
    iso = IsolationForrestModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated',
                                data_labels=labels)
    iso.train()
    iso.score(iso.anomaly_data)

    # Feed Forward Auto-Encoder
    forward = FFModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                      windows_size=WINDOW_SIZE)
    forward.train()
    yhat_healthy = forward.score(data=forward.get_normal_data())
    yhat_broken = forward.score(data=forward.get_anomaly_data())
    forward.calculate_threshold(health=yhat_healthy)
    forward.anomaly_score(pred=yhat_broken)

    # LSTM Auto-Encoder
    lstm = LSTMModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                     windows_size=WINDOW_SIZE)
    lstm.train()
    yhat_healthy = lstm.score(data=lstm.get_normal_data())
    yhat_broken = lstm.score(data=lstm.get_anomaly_data())
    lstm.calculate_threshold(health=yhat_healthy)
    lstm.anomaly_score(pred=yhat_broken)

    # GRU Auto-Encoder
    gru = GRUModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                   windows_size=WINDOW_SIZE)
    gru.train()
    yhat_healthy = gru.score(data=gru.get_normal_data())
    yhat_broken = gru.score(data=gru.get_anomaly_data())
    gru.calculate_threshold(health=yhat_healthy)
    gru.anomaly_score(pred=yhat_broken)

    # Convolutional Auto-Encoder
    conv = CONVModel(healthy_data=health_data, broken_data=anomaly_data, dataset_name='Generated', data_labels=labels,
                     windows_size=WINDOW_SIZE)
    conv.train()
    yhat_healthy = conv.score(data=conv.get_normal_data())
    yhat_broken = conv.score(data=conv.get_anomaly_data())
    conv.calculate_threshold(health=yhat_healthy)
    conv.anomaly_score(pred=yhat_broken)

    Results()


if __name__ == '__main__':
    main()
