from NN.ConvolutionalNeuralNetworkClassifier import ConvNeuralNetwork
from NN.DataUtils import Utils
from NN.FeedForwardNeuralNetworkClassifier import FFNeuralNetwork
from NN.RecurentNeuralNetworkClassfier import RecurentNeuralNetwork


def main() -> None:
    """
    Main function. Run 3 neural networks for classification task.
    Classifing MNIST dataset
    :return: None
    """
    # Make load data object
    data = Utils()
    set = data.get_x_test()

    # Feed Forward Network Classification
    nn1 = FFNeuralNetwork()
    num1 = nn1.get_prediction(data=set, id=0)
    acc1 = nn1.get_val_acc()
    loss1 = nn1.get_val_loss()

    # Recurent Network Classification
    nn2 = RecurentNeuralNetwork()
    num2 = nn2.get_prediction(data=set, id=0)
    acc2 = nn2.get_val_acc()
    loss2 = nn2.get_val_loss()

    # Conv Network Classification
    nn3 = ConvNeuralNetwork()
    num3 = nn3.get_prediction(data=set, id=0)
    acc3 = nn3.get_val_acc()
    loss3 = nn3.get_val_loss()

    # Results
    print("---------------RESULTS---------------")
    print(f"Feed Forward model - Accuracy: {acc1}, Loss: {loss1}, with predicted: {num1}")
    print(f"Recurrent model - Accuracy: {acc2}, Loss: {loss2}, with predicted: {num2}")
    print(f"Convolution model - Accuracy: {acc3}, Loss: {loss3}, with predicted: {num3}")
