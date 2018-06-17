import tflearn
from tflearn.data_utils import string_to_semi_redundant_sequences, random_sequence_from_string

def main():
    print("City Name Generator")

    path = "Cities.txt"

    maxlen = 20

    string_utf8 = open(path, "r").read()
    X, Y, char_idx = \
        string_to_semi_redundant_sequences(string_utf8, seq_maxlen=maxlen, redun_step=3)

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

    for i in range(40):
        seed = random_sequence_from_string(string_utf8, maxlen)
        genarator.fit(X, Y, validation_set=0.1, batch_size=128,
                n_epoch=1, run_id='us_cities')
        print("-- TESTING...")
        print("-- Test with temperature of 1.2 --")
        print(genarator.generate(10, temperature=1.2, seq_seed=seed).encode('utf-8'))
        print("-- Test with temperature of 1.0 --")
        print(genarator.generate(10, temperature=1.0, seq_seed=seed).encode('utf-8'))
        print("-- Test with temperature of 0.5 --")
        print(genarator.generate(10, temperature=0.5, seq_seed=seed).encode('utf-8'))

if __name__ == '__main__':
    main()