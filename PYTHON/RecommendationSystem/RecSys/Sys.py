import logging as log
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Embedding, multiply, Dropout, Dense, Flatten, concatenate
from tensorflow.keras.models import Model


class RecommendationSystem:
    def __init__(self, dataset_path: str) -> None:
        log.getLogger().setLevel(log.INFO)
        log.info('Build Recommendation System')
        df = pd.read_csv(dataset_path)

        self.mapped_ids = self.map_data(df.movieId)
        df.movieId = df.movieId.map(self.mapped_ids)

        mapped_users = self.map_data(df.userId)
        df.userId = df.userId.map(mapped_users)

        self.train, self.test = self.train_test_split(df=df, percent=80)
        self.max_user_id = max(df.userId.tolist())
        self.max_movie_id = max(df.movieId.tolist())

        self.model = self.build_model(self.max_movie_id, self.max_user_id)
        self.tb = TensorBoard(log_dir=f"logs\\Recomender-{int(time.time())}")

    @staticmethod
    def map_data(series: pd.Series) -> dict:
        return dict([(y, x + 1) for x, y in enumerate(series.unique())])

    @staticmethod
    def train_test_split(df: pd.DataFrame, percent: int) -> tuple:
        percentile = np.percentile(df.timestamp, percent)
        cols = list(df)
        train_data = df[df.timestamp < percentile][cols]
        test_data = df[df.timestamp > percentile][cols]
        return train_data, test_data

    @staticmethod
    def build_model(max_movie: int, max_user: int) -> Model:
        log.info('Start building neural network model...')
        dim_embedddings = 30
        bias = 1

        movie_inputs = Input(shape=(1,), dtype='int32')
        movie = Embedding(max_movie + 1, dim_embedddings, name="movie")(movie_inputs)
        movie_bias = Embedding(max_movie + 1, bias, name="movie_bias")(movie_inputs)

        user_inputs = Input(shape=(1,), dtype='int32')
        user = Embedding(max_user + 1, dim_embedddings, name="user")(user_inputs)
        user_bias = Embedding(max_user + 1, bias, name="user_bias")(user_inputs)

        network = multiply([movie, user])
        network = Dropout(0.3)(network)
        network = concatenate([network, user_bias, movie_bias])
        network = Flatten()(network)
        network = Dense(10, activation="relu")(network)
        network = Dense(1)(network)

        model = Model(inputs=[movie_inputs, user_inputs], outputs=network)
        model.summary()
        model.compile(loss='mae', optimizer='adam', metrics=['mse'])
        return model

    def save_model(self, path: str) -> None:
        log.info('Saving model...')
        self.model.save(path)

    def load_model(self, path: str) -> None:
        log.info('Load model...')
        self.model = tf.keras.models.load_model(path)

    def train_model(self) -> None:
        path = 'Recommendation__Model.h5'
        if os.path.exists(path):
            log.info('Model detected')
            self.load_model(path)
        else:
            log.info('Start training model')
            self.model.fit([self.train.movieId.values, self.train.userId.values], self.train.rating.values, epochs=100,
                           verbose=0, validation_split=0.2, callbacks=[self.tb])
            self.save_model(path)
        log.info('Model training complete!')

    def model_evaluation(self) -> None:
        log.info('Start ML model evaluation')
        predictions = self.model.predict([self.test.movieId.values, self.test.userId.values])
        score = r2_score(self.test.rating.values, predictions)
        log.info(f"R2 Score for trained model: {score}")
