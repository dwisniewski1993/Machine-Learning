from NN.DataUtils import Utils
from NN.FeedForwardNeuralNetworkClassifier import FFNeuralNetwork
from NN.RecurentNeuralNetworkClassfier import RecurentNeuralNetwork
from NN.ConvolutionalNeuralNetworkClassifier import ConvNeuralNetwork


def main():
    # Make load data object
    data = Utils()
    set = data.get_x_test()

    # Feed Forward Network Classification
    nn1 = FFNeuralNetwork()
    num1 = nn1.get_prediction(data=set, id=0)
    acc1 = nn1.get_val_acc()
    loss1 = nn1.get_val_loss()
    print("Accuracy: ", acc1, ", Loss: ", loss1, " with predicted: ", num1)

    # Recurent Network Classification
    nn2 = RecurentNeuralNetwork()
    num2 = nn2.get_prediction(data=set, id=0)
    acc2 = nn2.get_val_acc()
    loss2 = nn2.get_val_loss()
    print("Accuracy: ", acc2, ", Loss: ", loss2, " with predicted: ", num2)

    # Conv Network Classification
    nn3 = ConvNeuralNetwork()
    num3 = nn3.get_prediction(data=set, id=0)
    acc3 = nn3.get_val_acc()
    loss3 = nn3.get_val_loss()
    print("Accuracy: ", acc3, ", Loss: ", loss3, " with predicted: ", num3)
