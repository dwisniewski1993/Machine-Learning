from Models.Conv1D.Conv2DModel import Conv2DModel
from Models.FeedForward.FeedForwardModel import FFModel
from Models.GRU.GRUModel import GRUModel
from Models.LSTM.LSTMModel import LSTMModel
from Models.Utils import Results


def main():
    """
    Main function. Anomaly/Outliers detection with neural networks architectures.
    train set: SWAT health data
    validation set: SWAT broken data
    :return: None
    """
    swat_normal_file = r'Datasets/SWAT/data_normal_small.csv'
    swat_attk_file = r'Datasets/SWAT/data_attk_small.csv'

    # LSTM Autoencoder
    lstm = LSTMModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    lstm.train()
    yhat_healthy = lstm.score(data=lstm.get_normal_data())
    yhat_broken = lstm.score(data=lstm.get_attk_data())
    lstm.calculate_threshold(helth=yhat_healthy)
    lstm.anomaly_score(pred=yhat_broken)

    # Conv2D Autoencoder
    con = Conv2DModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    con.train()
    yhat_healthy = con.score(data=con.get_normal_data())
    yhat_broken = con.score(data=con.get_attk_data())
    con.calculate_threshold(helth=yhat_healthy)
    con.anomaly_score(pred=yhat_broken)

    # FeedForward Autoencoder
    ff = FFModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    ff.train()
    yhat_healthy = ff.score(data=ff.get_normal_data())
    yhat_broken = ff.score(data=ff.get_attk_data())
    ff.calculate_threshold(helth=yhat_healthy)
    ff.anomaly_score(pred=yhat_broken)

    # GRU Autoencoder
    gru = GRUModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    gru.train()
    yhat_healthy = gru.score(data=gru.get_normal_data())
    yhat_broken = gru.score(data=gru.get_attk_data())
    gru.calculate_threshold(helth=yhat_healthy)
    gru.anomaly_score(pred=yhat_broken)

    # Anomaly detection / Outliers detection results
    Results()