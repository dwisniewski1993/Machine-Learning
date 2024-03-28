from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

parameters = {
    'kernel': ('linear', 'rbf', 'poly'),
    'gamma': [0.001, 0.01, 0.1, 1],
    'C': [2, 3, 5, 7, 11]
}
scaler_type = ["standard", "normalizer", "robust", "min-max", "max-abs"]

scaler = {"standard": StandardScaler(),
          "normalizer": Normalizer(),
          "robust": RobustScaler(),
          "min-max": MinMaxScaler(),
          "max-abs": MaxAbsScaler()}
