from NN.NeuNet import NeuralNetwork, Utils


def main():
    print("MNIST Classifier")

    neuralnet = NeuralNetwork()
    data = Utils()
    set = data.get_x_test()

    num = neuralnet.get_prediction(data=set, id=0)
    acc = neuralnet.get_val_acc()
    loss = neuralnet.get_val_loss()
    print("Accuracy: ", acc, ", Loss: ", loss, " with predicted: ", num)


if __name__ == '__main__':
    main()
