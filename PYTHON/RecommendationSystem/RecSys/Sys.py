import pandas as pd
import numpy as np
from keras import Input, Model
from keras.layers import Embedding, merge, Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.regularizers import l2


class RecSystem:
    def __init__(self, datasetloc):
        self.ratings = pd.read_csv(datasetloc)

        users = self.ratings.userId.unique()
        movies = self.ratings.movieId.unique()

        userid2idx = {o: i for i, o in enumerate(users)}
        movieid2idx = {o: i for i, o in enumerate(movies)}

        self.ratings.userId = self.ratings.userId.apply(lambda x: userid2idx[x])
        self.ratings.movieId = self.ratings.movieId.apply(lambda x: movieid2idx[x])

        np.random.seed = 42

        msk = np.random.rand(len(self.ratings)) < 0.8
        self.trn = self.ratings[msk]
        self.val = self.ratings[~msk]

        g = self.ratings.groupby('userId')['rating'].count()
        self.topUsers = g.sort_values(ascending=False)[:15]

        g = self.ratings.groupby('movieId')['rating'].count()
        self.topMovies = g.sort_values(ascending=False)[:15]

        top_r = self.ratings.join(self.topUsers, rsuffix='_r', how='inner', on='userId')
        top_r = top_r.join(self.topMovies, rsuffix='_r', how='inner', on='movieId')
        self.dataset = pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)

        n_users = self.ratings.userId.nunique()
        n_movies = self.ratings.movieId.nunique()
        n_factors = 50

        self.user_in, self.u = self.embedding_input('user_in', n_users, n_factors, 1e-4)
        self.movie_in, self.m = self.embedding_input('movie_in', n_movies, n_factors, 1e-4)

        self.model = self.build_network()

    def show_dataset(self):
        print(self.dataset)

    @staticmethod
    def embedding_input(name, n_in, n_out, reg):
        inp = Input(shape=(1,), dtype='int64', name=name)
        return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

    def build_network(self):
        net = merge([self.u, self.m], mode='concat')
        net = Flatten()(net)
        net = Dropout(0.3)(net)
        net = Dense(70, activation='relu')(net)
        net = Dropout(0.75)(net)
        net = Dense(1)(net)

        nn = Model([self.user_in, self.movie_in], net)
        nn.compile(Adam(0.001), loss='mse')

        return nn

    def train_network(self):
        self.model.fit([self.trn.userId, self.trn.movieId], self.trn.rating, batch_size=64, nb_epoch=1,
                       validation_data=([self.val.userId, self.val.movieId], self.val.rating))

    def output(self, user_id, movie_id):
        return self.model.predict([user_id, movie_id])

    def give_first_ten_pred(self):
        predicted_scores = self.model.predict([self.trn.userId, self.trn.movieId])
        return predicted_scores[:10]
