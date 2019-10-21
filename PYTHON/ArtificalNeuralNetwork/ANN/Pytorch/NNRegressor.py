import logging as log

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, num_input: int, num_output: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, 8)
        self.fc2 = nn.Linear(8, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class TorchNeuralNetRegression:
    """
    Artificial Neural Network Regression
    """

    def __init__(self, train_file: str) -> None:
        """
        Loading and preparing data
        :param train_file: path to train file
        """
        log.getLogger().setLevel(log.INFO)
        log.info('Neural Network Regression With PyTorch')

        self.trainFile = train_file
        train_data_frame = pd.read_csv(self.trainFile, sep='\t', header=None)

        # Mapping string and bool values to numeric
        self.mapping_string = self.map_columns(train_data_frame, 4)
        self.mapping_bool = self.map_columns(train_data_frame, 1)
        train_data_frame = train_data_frame.applymap(
            lambda x: self.mapping_string.get(x) if x in self.mapping_string else x)
        train_data_frame = train_data_frame.applymap(
            lambda x: self.mapping_bool.get(x) if x in self.mapping_bool else x)
        train_array = train_data_frame.values

        # Shuffle Data
        np.random.shuffle(train_array)

        # Extract values to numpy.Arrays
        self.X = train_array[:, 1:]
        self.Y = train_array[:, 0]

        # Split to train-test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=0)
        self.model = Net(num_input=self.X_train.shape[-1], num_output=1)

        self.X_train = Variable(torch.Tensor(self.X_train).float())
        self.X_test = Variable(torch.Tensor(self.X_test).float())
        self.Y_train = Variable(torch.Tensor(self.Y_train).float())
        self.Y_test = Variable(torch.Tensor(self.Y_test).float())

    @staticmethod
    def map_columns(df: pd.DataFrame, col_number: int) -> dict:
        """
        Mapping non numeric values to numeric
        :param df: pandas dataframe that contain dataset
        :param col_number: number collumn to map
        :return: dictionary with mapped values
        """
        return dict([(y, x + 1) for x, y in enumerate(sorted(set(df[col_number].unique())))])

    def normalize(self) -> None:
        """
        Standardize the data
        :return: None
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    def train_model(self) -> None:
        """
        Training model
        :return: None
        """
        loss_function = nn.MSELoss()
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
        val_loss = mean_squared_error(self.Y_test, predict_out.detach().numpy())

        return val_loss
