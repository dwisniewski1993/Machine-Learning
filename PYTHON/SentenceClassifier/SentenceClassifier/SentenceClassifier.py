import logging as log
import random
from collections import Counter
from statistics import mean

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


class Sentencer:
    def __init__(self, possitive_data: str, negative_data: str) -> None:
        log.getLogger().setLevel(log.INFO)
        log.info('Sentence Classifier')

        self.pos_path = possitive_data
        self.neg_path = negative_data

        self.lemmatizer = WordNetLemmatizer()
        self.hm_lines = 1000000

        lexicon = self.create_lexicon()

        log.info('Start pre-processing data...')
        self.train_x, self.train_y, self.test_x, self.test_y = self.create_set(pos=self.pos_path, neg=self.neg_path,
                                                                               lexicon=lexicon, test_size=0.1)
        log.info('Data processed')

        self.model = self.define_feedforward_dnn_model(hidden_layer_number=1,
                                                       hidden_units_number=int(mean([2, self.train_x.shape[0]])))

    def create_lexicon(self) -> list:
        lexicon = []

        with open(self.pos_path, 'r') as file:
            contents = file.readlines()
            for line in contents[:self.hm_lines]:
                lexicon += list(word_tokenize(line))

        with open(self.neg_path, 'r') as file:
            contents = file.readlines()
            for line in contents[:self.hm_lines]:
                lexicon += list(word_tokenize(line))

        lexicon = [self.lemmatizer.lemmatize(i) for i in lexicon]
        w_counts = Counter(lexicon)
        lex = [w for w in w_counts if 1000 > w_counts[w] > 50]

        return lex

    def sample_handling(self, sample: str, lexicon: list, label: list) -> list:
        dataset = []
        with open(sample, 'r') as f:
            contents = f.readlines()
            for l in contents[:self.hm_lines]:
                current_words = word_tokenize(l.lower())
                current_words = [self.lemmatizer.lemmatize(i) for i in current_words]
                features = np.zeros(len(lexicon))
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1

                features = list(features)
                dataset.append([features, label])

        return dataset

    def create_set(self, pos: str, neg: str, lexicon: list, test_size: float = 0.3) -> tuple:
        features = []
        features += self.sample_handling(pos, lexicon, [1, 0])
        features += self.sample_handling(neg, lexicon, [0, 1])
        random.shuffle(features)
        features = np.array(features)

        testing_size = int(test_size * len(features))

        train_x = np.array(list(features[:, 0][:-testing_size]))
        train_y = np.array(list(features[:, 1][:-testing_size]))
        test_x = np.array(list(features[:, 0][-testing_size:]))
        test_y = np.array(list(features[:, 1][-testing_size:]))

        return train_x, train_y, test_x, test_y

    @staticmethod
    def define_feedforward_dnn_model(hidden_layer_number: int = 1, hidden_units_number: int = 512) -> Sequential:
        log.info(f"Building model with {hidden_layer_number} hidden layers with {hidden_units_number} neurons in each")
        model = Sequential()
        for i in range(hidden_layer_number):
            model.add(Dense(hidden_units_number, activation=activations.relu))
        model.add(Dense(2, activation=activations.softmax))
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, num_epochs: int = 10):
        log.info('Start training model...')
        self.model.fit(self.train_x, self.train_y, epochs=num_epochs, verbose=0)
        val_loss, val_acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        log.info(f'Model trained with loss {val_loss} and accuracy {val_acc}')
