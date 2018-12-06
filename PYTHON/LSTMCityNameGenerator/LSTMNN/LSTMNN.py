import tflearn
from tflearn.data_utils import string_to_semi_redundant_sequences, random_sequence_from_string


class LSTMNN:
    def __init__(self, path):

        self.path = path
        self.max_lenght = 20

        self.string_utf8 = open(self.path, "r").read()

        self.X, self.Y, self.char_idx = \
                    string_to_semi_redundant_sequences(self.string_utf8, seq_maxlen=self.max_lenght, redun_step=3)

        self.generator = self.build_model(self.max_lenght, self.char_idx)

    @staticmethod
    def build_model(maxlen, char_idx):

        net = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
        net = tflearn.lstm(net, 512, return_seq=True)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.lstm(net, 512)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, len(char_idx), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',
                                 learning_rate=0.001)

        genarator = tflearn.SequenceGenerator(net, dictionary=char_idx,
                                              seq_maxlen=maxlen,
                                              clip_gradients=5.0,
                                              checkpoint_path='model_us_cities')
        return genarator

    def train(self, num_epochs, temperature=1.2, num_characters=10):
        for i in range(num_epochs):
            self.seed = random_sequence_from_string(self.string_utf8, self.max_lenght)
            self.generator.fit(self.X, self.Y, validation_set=0.1, batch_size=128,
                n_epoch=1, run_id='us_cities')

            print("---TESTING---")
            print("Test with temperature ", temperature)
            print(self.generator.generate(num_characters, temperature=temperature, seq_seed=self.seed).encode('utf-8'))

    def output(self, temperature=1.2, num_characters=10):
        return self.generator.generate(num_characters, temperature=temperature, seq_seed=self.seed).encode('utf-8')
