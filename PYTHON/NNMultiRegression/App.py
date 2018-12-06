import os.path
from data_generator import generate_dataset
from MultiRegressionDir.MultiRegressionPython import MRNN


def main():
    print("Neural Networks for Multi Regression")

    NUM_FEATURES = 12
    NUM_LABELS = 7
    SET_SIZE = 10000

    features = r'x.csv'
    labels = r'y.csv'

    if (os.path.exists(features) & os.path.exists(labels)) is False:
        print("Brak plikow")
        generate_dataset(features_number=NUM_FEATURES, labels_number=NUM_LABELS, set_size=SET_SIZE)

    network = MRNN(featuresfile=features, labelsfile=labels)
    network.train_network()


if __name__ == '__main__':
    main()
