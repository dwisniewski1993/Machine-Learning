import logging as log

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, num_input: int, num_output: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, 8)
        self.fc2 = nn.Linear(8, num_output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class TorchNeuralNetClassifier:
    """
        Artificial Neural Network Classification using PyTorch
        """

    def __init__(self, train_file: str) -> None:
        """
        Loading and preparing data
        :param train_file: path to train file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Classifier With PyTorch')

        # Load set
        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile)
        train_array = train_data_frame.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract value to numpy.Array
        self.X = train_array[:, 0:4].astype(float)
        self.Y = train_array[:, 4]

        # Map string labels to numeric
        self.Y = np.array(self.map_labels(self.Y)).astype(float)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)

        self.model = Net(num_input=self.X_train.shape[-1], num_output=self.Y_train.shape[-1])

        self.X_train = Variable(torch.Tensor(self.X_train).float())
        self.X_test = Variable(torch.Tensor(self.X_test).float())
        self.Y_train = Variable(torch.Tensor(self.Y_train).long())
        self.Y_test = Variable(torch.Tensor(self.Y_test).long())

    @staticmethod
    def map_labels(labels: np.array) -> list:
        """
        Mapping iris data labels to categorical values
        :param labels: numpy.Arrays contains labels
        :return: list of mapped values
        """
        mapped = [0 if x == 'Iris-setosa' else 1 if x == 'Iris-versicolor' else 2 for x in labels]
        return mapped

    def train_model(self) -> None:
        """
        Training model
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

    def output(self) -> float:
        """
        Print Validation loss and accuracy
        :return: None
        """
        predict_out = self.model(self.X_test)
        _, predict_y = torch.max(predict_out, 1)
        val_acc = accuracy_score(self.Y_test, predict_y)

        return val_acc
