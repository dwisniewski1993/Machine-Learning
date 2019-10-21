import time

from ANN.Pytorch.NNClassifier import TorchNeuralNetClassifier
from ANN.Pytorch.NNRegressor import TorchNeuralNetRegression
from ANN.Tensorflow.NNClassifier import TfNeuralNetClassifier
from ANN.Tensorflow.NNRegressor import TfNeuralNetRegression


def main() -> None:
    """
    Main function. Run Artificial Neural Network classification and regression tasks.
    train.csv: iris dataset
    train.tsv: Poznań flats prices
    :return: None
    """
    # Specify files path
    train_file_location_classification = r'train.csv'
    train_file_location_regression = r'train.tsv'

    # Artificial Neural Network Classification - TF Version
    start_time = time.time()
    tf_neural_net_classifier = TfNeuralNetClassifier(train_file=train_file_location_classification)
    tf_neural_net_classifier.train_model()
    elapsed_tf_classification = time.time() - start_time

    # Artificial Neural Network Regression - TF Version
    start_time = time.time()
    tf_neural_net_regression = TfNeuralNetRegression(train_file=train_file_location_regression)
    tf_neural_net_regression.train_model()
    elapsed_tf_regression = time.time() - start_time

    # Artificial Neural Network Classification - PyTorch Version
    start_time = time.time()
    torch_neural_net_classifier = TorchNeuralNetClassifier(train_file=train_file_location_classification)
    torch_neural_net_classifier.train_model()
    elapsed_torch_classification = time.time() - start_time

    # Artificial Neural Network Regression - PyTorch Version
    start_time = time.time()
    torch_neural_network_regression = TorchNeuralNetRegression(train_file=train_file_location_regression)
    torch_neural_network_regression.train_model()
    elapsed_torch_regression = time.time() - start_time

    # Output
    val_loss, val_accuracy = tf_neural_net_classifier.output()
    print(f"Tensorflow classifier accuracy: {val_accuracy}, 100 epochs in {elapsed_tf_classification}")
    val_loss, val_accuracy = tf_neural_net_regression.output()
    print(f"Tensorflow regression loss: {val_loss}, 100 epochs in {elapsed_tf_regression}")
    val_accuracy = torch_neural_net_classifier.output()
    print(f"PyTorch classifier accuracy: {val_accuracy}, 1000 epochs in {elapsed_torch_classification}")
    val_loss = torch_neural_network_regression.output()
    print(f"PyTorch regression loss: {val_loss}, 1000 epochs in {elapsed_torch_regression}")
