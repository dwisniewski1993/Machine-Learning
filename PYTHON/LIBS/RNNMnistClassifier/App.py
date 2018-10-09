from NN.RecNeuNet import NeuralNetwork
from NN.Utils import Utils


def main():
    print("RNN Mnist Classifier")

    neuralnetwork = NeuralNetwork()
    data = Utils()
    dataset = data.get_x_test()
    expected = data.get_y_test()

    num = neuralnetwork.get_prediction(data=dataset, id=0)
    acc = neuralnetwork.get_val_acc()
    loss = neuralnetwork.get_val_loss()
    print("Accuracy: ", acc, ", Loss: ", loss, " with predicted: ", num, " expected: ", expected[0])


if __name__ == '__main__':
    main()
