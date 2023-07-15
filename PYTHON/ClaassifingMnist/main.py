from NN.FeedForwardNeuralNetworkClassifier import FFNeuralNetwork
from NN.RecurentNeuralNetworkClassfier import RecurrentNeuralNetwork
from NN.ConvolutionalNeuralNetworkClassifier import ConvNeuralNetwork
from NN.DataUtils import Utils


def main() -> None:
    """
    Main function. Run 3 neural networks for classification task.
    Classifying MNIST dataset
    :return: None
    """
    # Make load data object
    data = Utils()
    dataset = data.get_x_test()

    # Feed Forward Network Classification
    nn1 = FFNeuralNetwork()
    num1 = nn1.get_prediction(data=dataset, id=0)
    acc1 = nn1.get_val_acc()
    loss1 = nn1.get_val_loss()
    f11 = nn1.get_f1_score()

    # Recurent Network Classification
    nn2 = RecurrentNeuralNetwork()
    num2 = nn2.get_prediction(data=dataset, id=0)
    acc2 = nn2.get_val_acc()
    loss2 = nn2.get_val_loss()
    f12 = nn2.get_f1_score()

    # Conv Network Classification
    nn3 = ConvNeuralNetwork()
    num3 = nn3.get_prediction(data=dataset, id=0)
    acc3 = nn3.get_val_acc()
    loss3 = nn3.get_val_loss()
    f13 = nn3.get_f1_score()

    # Results
    print("---------------RESULTS---------------")
    print(f"Feed Forward model - Accuracy: {acc1}, Loss: {loss1}, F1 Score:{f11}, with predicted: {num1}")
    print(f"Recurrent model - Accuracy: {acc2}, Loss: {loss2}, F1 Score:{f12}, with predicted: {num2}")
    print(f"Convolution model - Accuracy: {acc3}, Loss: {loss3}, F1 Score:{f13}, with predicted: {num3}")


if __name__ == '__main__':
    main()
