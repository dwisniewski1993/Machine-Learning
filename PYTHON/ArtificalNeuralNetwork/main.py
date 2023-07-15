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
    # Specify file paths
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
    f1_val = tf_neural_net_classifier.score()
    print(f"TensorFlow classifier f1 score: {f1_val}, 500 epochs in {elapsed_tf_classification} seconds")

    r2_val = tf_neural_net_regression.score()
    print(f"TensorFlow regression r2 score: {r2_val}, 500 epochs in {elapsed_tf_regression} seconds")

    f1_val = torch_neural_net_classifier.score()
    print(f"PyTorch classifier f1 score: {f1_val}, 500 epochs in {elapsed_torch_classification} seconds")

    r2_val = torch_neural_network_regression.score()
    print(f"PyTorch regression r2 score: {r2_val}, 500 epochs in {elapsed_torch_regression} seconds")
