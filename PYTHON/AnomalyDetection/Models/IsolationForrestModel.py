import logging
import os.path

import numpy as np
from sklearn.ensemble.iforest import IsolationForest
from tqdm import tqdm

from Models.Utils import SWATDataHandler, Preprocessing


class IsolationForrestModel:
    def __init__(self, healthy_data: str, broken_data: str, dataset_name: str) -> None:
        logging.basicConfig(level=logging.INFO)
        self.data_name = dataset_name
        self.logger = logging.getLogger(__name__)
        handler = SWATDataHandler(file_normal=healthy_data, file_brocken=broken_data)

        self.normal_data = handler.get_dataset_normal()
        self.attk_data = handler.get_dataset_broken()
        self.Y = handler.get_broken_labels()

        scaler = Preprocessing()
        self.normal_data = scaler.scaleData(data=self.normal_data)
        self.attk_data = scaler.scaleData(data=self.attk_data)

        self.logger.info('Initializing Isolation Forrest model...')
        self.model = IsolationForest(n_estimators=20, n_jobs=4, verbose=1)

    def train(self, retrain=False) -> None:
        data = self.normal_data

        if retrain:
            self.logger.info('Start training Isolation Forrest model...')
            self.model.fit(data)
        else:
            if os.path.exists(self.data_name + '__IsolationForrest_model.npy'):
                self.logger.info('Loading Isolation Forrest model...')
                self.model = np.load(self.data_name + '__IsolationForrest_model.npy')
            else:
                self.logger.info('Start training Isolation Forrest model...')
                self.model.fit(data)
                np.save(self.data_name + '__IsolationForrest_model.npy', self.model)

    def score(self) -> None:
        self.logger.info('Calculating anomaly score...')
        data = self.attk_data
        if os.path.exists('IsolationForrest_detected_SWAT.csv'):
            logging.info('Predictions already exist')
        else:
            with open('IsolationForrest_detected_SWAT.csv', 'w') as results:
                predictions = [self.model.predict(data[i].reshape(1, -1)) for i in tqdm(range(len(data)))]
                for i in tqdm(range(len(predictions))):
                    if predictions[i]:
                        results.write(f"1.0,{self.Y[i]},Normal\n")
                    else:
                        results.write(f"0.0,{self.Y[i]},Anomaly\n")
            results.close()
