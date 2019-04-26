from Models.Conv2DModel import Conv2DModel
from Models.FeedForwardModel import FFModel
from Models.FuzzyModel import FuzzyModel
from Models.GRUModel import GRUModel
from Models.IsolationForrestModel import IsolationForrestModel
from Models.LSTMModel import LSTMModel
from Models.OneClassSVMModel import OneClassSVMModel
from Models.Utils import Results


def main():
    """
    Main function. Anomaly/Outliers detection with neural networks architectures.
    train set: health data 
    validation set: broken data
    :return: None
    """
    swat_normal_file = r'Datasets/data_normal.csv'
    swat_attk_file = r'Datasets/data_attk.csv'

    # LSTM Autoencoder
    lstm = LSTMModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated', timesteps=10)
    lstm.train()
    yhat_healthy = lstm.score(data=lstm.get_normal_data())
    yhat_broken = lstm.score(data=lstm.get_attk_data())
    lstm.calculate_threshold(helth=yhat_healthy)
    lstm.anomaly_score(pred=yhat_broken)

    # Conv2D Autoencoder
    con = Conv2DModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated', timesteps=10)
    con.train()
    yhat_healthy = con.score(data=con.get_normal_data())
    yhat_broken = con.score(data=con.get_attk_data())
    con.calculate_threshold(helth=yhat_healthy)
    con.anomaly_score(pred=yhat_broken)

    # FeedForward Autoencoder
    ff = FFModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated', timesteps=10)
    ff.train()
    yhat_healthy = ff.score(data=ff.get_normal_data())
    yhat_broken = ff.score(data=ff.get_attk_data())
    ff.calculate_threshold(helth=yhat_healthy)
    ff.anomaly_score(pred=yhat_broken)

    # GRU Autoencoder
    gru = GRUModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated', timesteps=10)
    gru.train()
    yhat_healthy = gru.score(data=gru.get_normal_data())
    yhat_broken = gru.score(data=gru.get_attk_data())
    gru.calculate_threshold(helth=yhat_healthy)
    gru.anomaly_score(pred=yhat_broken)

    # One Class SVM
    svm = OneClassSVMModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated')
    svm.train()
    svm.score()

    # Isolation Forrest
    iso = IsolationForrestModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated')
    iso.train()
    iso.score()

    # Fuzzy Time Series
    fst = FuzzyModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='Generated')
    fst.train()
    yhat_healthy = fst.score(data=fst.get_normal_data())
    yhat_broken = fst.score(data=fst.get_attk_data())
    fst.calculate_threshold(helth=yhat_healthy)
    fst.anomaly_score(pred=yhat_broken)

    # Anomaly detection / Outliers detection results
    Results()
