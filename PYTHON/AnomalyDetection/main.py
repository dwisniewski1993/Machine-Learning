from Models.Conv1D.Conv1DModel import Conv1DModel
from Models.FeedForward.FeedForwardModel import FFModel
from Models.LSTM.LSTMModel import LSTMModel


def main():
    swat_normal_file = r'Datasets/SWAT/data_normal.csv'
    swat_attk_file = r'Datasets/SWAT/data_attk.csv'

    # LSTM Autoencoder
    lstm = LSTMModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    lstm.train()
    yhat_healthy = lstm.score(lstm.get_normal_data())
    yhat_broken = lstm.score(lstm.get_attk_data())
    lstm.calculate_threshold(helth=yhat_healthy)
    lstm.anomaly_score(pred=yhat_broken)

    # Conv2D Autoencoder
    con = Conv1DModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    con.train()
    yhat_healthy = con.score(con.get_normal_data())
    yhat_broken = con.score(con.get_attk_data())
    con.calculate_threshold(helth=yhat_healthy)
    con.anomaly_score(pred=yhat_broken)

    # FeedForward Autoencoder
    ff = FFModel(healthy_data=swat_normal_file, broken_data=swat_attk_file, dataset_name='SWAT', timesteps=10)
    ff.train()
    yhat_healthy = ff.score(ff.get_normal_data())
    yhat_broken = ff.score(ff.get_attk_data())
    ff.calculate_threshold(helth=yhat_healthy)
    ff.anomaly_score(pred=yhat_broken)
