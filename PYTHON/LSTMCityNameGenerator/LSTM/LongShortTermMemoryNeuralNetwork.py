import logging as log

import tflearn
from tflearn.data_utils import string_to_semi_redundant_sequences, random_sequence_from_string


class LongShortTermMemoryNeuralNetwork:
    """
    LSTM Neural Network generate city name
    """

    def __init__(self, path: str) -> None:
        """
        City name generator constructor for loading and preparing data
        :param path: path to file with city names
        """
        log.getLogger().setLevel(log.INFO)
        log.info('City Name Generator')
        self.path = path
        self.max_length = 20
        self.string_utf8 = open(self.path, "r").read()
        self.seed = 0
        self.X, self.Y, self.char_idx = string_to_semi_redundant_sequences(self.string_utf8, seq_maxlen=self.max_length,
                                                                           redun_step=3)
        self.generator = self.build_model(self.max_length, self.char_idx)

    @staticmethod
    def build_model(max_len: int, char_idx: dict) -> tflearn.models.generator.SequenceGenerator:
        """
        Building LSTM Neural Network Model
        :param max_len: maximum length of char
        :param char_idx: dictionary - each char to int number mapped
        :return: tflearn model object
        """
        log.info('Building model')
        net = tflearn.input_data(shape=[None, max_len, len(char_idx)])
        net = tflearn.lstm(net, 128, return_seq=True)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 128)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, len(char_idx), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

        genarator = tflearn.SequenceGenerator(net,
                                              dictionary=char_idx,
                                              seq_maxlen=max_len,
                                              clip_gradients=5.0,
                                              checkpoint_path='model_us_cities')
        return genarator

    def train(self, num_epochs: int, temperature: float = 1.2, num_characters: int = 10) -> None:
        """
        Train iterable on batch, after each test with generated score for each tamperature
        :param num_epochs: number of training intervals
        :param temperature: network temperature parameter
        :param num_characters: number of character to generate
        :return: None
        """
        log.info(f"Start training on {num_epochs} epochs....")
        for i in range(num_epochs):
            self.seed = random_sequence_from_string(self.string_utf8, self.max_length)
            self.generator.fit(self.X, self.Y, validation_set=0.1, batch_size=128, n_epoch=1, run_id='us_cities')

            log.info('----------TESTING----------')
            log.info(self.generator.generate(num_characters, temperature=temperature, seq_seed=self.seed)
                     .encode('utf-8'))

    def output(self, temperature: float = 1.2, num_characters: int = 10) -> str:
        """
        Generate final score for given parameters
        :param temperature: network temperature parameter
        :param num_characters: number of character to generate
        :return: generated string
        """
        log.info('Generating Score....')
        return self.generator.generate(num_characters, temperature=temperature, seq_seed=self.seed)
