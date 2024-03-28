from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler


parameters = {
            'max_iter': [1000000],
            'penalty': ('l1', 'l2', 'elasticnet'),
            'learning_rate': ('constant', 'optimal', 'invscaling', 'adaptive'),
            'eta0': [1, 3, 5, 7, 9, 11]
        }

scaler_type = ["standard", "robust", "min-max", "max-abs"]

scaler = {"standard": StandardScaler(),
          "robust": RobustScaler(),
          "min-max": MinMaxScaler(),
          "max-abs": MaxAbsScaler()}
