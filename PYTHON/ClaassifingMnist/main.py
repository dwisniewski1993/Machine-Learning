import numpy as np
from typing import Tuple

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
    num1, acc1, loss1, f1_1 = classify(nn1, dataset)

    # Recurrent Network Classification
    nn2 = RecurrentNeuralNetwork()
    num2, acc2, loss2, f1_2 = classify(nn2, dataset)

    # Convolutional Network Classification
    nn3 = ConvNeuralNetwork()
    num3, acc3, loss3, f1_3 = classify(nn3, dataset)

    # Results
    print("---------------RESULTS---------------")
    print(f"Feed Forward model - Accuracy: {acc1}, Loss: {loss1}, F1 Score: {f1_1}, Predicted: {num1}")
    print(f"Recurrent model - Accuracy: {acc2}, Loss: {loss2}, F1 Score: {f1_2}, Predicted: {num2}")
    print(f"Convolutional model - Accuracy: {acc3}, Loss: {loss3}, F1 Score: {f1_3}, Predicted: {num3}")


def classify(model: object, dataset: np.array) -> Tuple[int, float, float, float]:
    """
    Perform classification using a given neural network model on the dataset
    :param model: Neural network model for classification
    :param dataset: Dataset to classify
    :return: Tuple containing the predicted class, accuracy, loss, and F1 score
    """
    num = model.get_prediction(data=dataset, id=0)
    acc = model.get_val_acc()
    loss = model.get_val_loss()
    f1_score = model.get_f1_score()
    return num, acc, loss, f1_score


if __name__ == '__main__':
    main()
