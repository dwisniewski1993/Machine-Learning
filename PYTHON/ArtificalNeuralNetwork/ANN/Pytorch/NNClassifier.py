import logging as log

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, num_input: int, num_output: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, 8)
        self.fc2 = nn.Linear(8, num_output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class TorchNeuralNetClassifier:
    """
    Class implementing an artificial neural network for classification using PyTorch
    """

    def __init__(self, train_file: str) -> None:
        """
        Load and prepare the data
        :param train_file: path to the training file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Classifier With PyTorch')

        # Load the dataset
        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)
        train_array = train_data_frame.values

        # Shuffle the data
        np.random.shuffle(train_array)

        # Extract values into numpy arrays
        self.X = train_array[:, 0:4].astype(float)
        self.Y = train_array[:, 4]

        # Map string labels to numeric
        self.Y = np.array(self.map_labels(self.Y)).astype(float)

        # Split into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)

        # Initialize the neural network model
        self.model = Net(num_input=self.X_train.shape[-1], num_output=self.Y_train.shape[-1])

        # Convert data to torch.Tensor variables
        self.X_train = Variable(torch.Tensor(self.X_train).float())
        self.X_test = Variable(torch.Tensor(self.X_test).float())
        self.Y_train = Variable(torch.Tensor(self.Y_train).long())
        self.Y_test = Variable(torch.Tensor(self.Y_test).long())

    @staticmethod
    def map_labels(labels: np.array) -> list:
        """
        Map iris data labels to categorical values
        :param labels: numpy array containing labels
        :return: list of mapped values
        """
        mapped = [0 if x == 'Iris-setosa' else 1 if x == 'Iris-versicolor' else 2 for x in labels]
        return mapped

    def train_model(self) -> None:
        """
        Train the model
        :return: None
        """
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for _ in tqdm(range(1000)):
            optimizer.zero_grad()
            out = self.model(self.X_train)
            loss = loss_function(out, self.Y_train)
            loss.backward()
            optimizer.step()

    def score(self) -> float:
        """
        Calculate the F1 score for the test data
        :return: F1 score value
        """
        predict_out = self.model(self.X_test)
        _, predict_y = torch.max(predict_out, 1)
        y_pred = f1_score(self.Y_test, predict_y, average='weighted')
        return y_pred
