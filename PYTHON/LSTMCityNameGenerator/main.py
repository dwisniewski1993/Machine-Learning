from LSTM.LongShortTermMemoryNeuralNetwork import LongShortTermMemoryNeuralNetwork


def main():
    """
    Main function. Run City Name Generator
    :return: None
    """
    train_file_location = r'Cities.txt'
    lstm_neural_network = LongShortTermMemoryNeuralNetwork(path=train_file_location)
    lstm_neural_network.train(num_epochs=1, temperature=1.5, num_characters=10)
    score = lstm_neural_network.output(temperature=1.5, num_characters=10)
    print(f"Final Score: {score}")
