"""Converting data from MATLAB .mat file to torch tensors

    Searches for 'shlezingerMat.mat' file, extracts the variables 'trainX'
    'trainS' 'dataX' 'dataS' variables and returning two classes, containing the
    train and test data.

    Returns
    -------
    ShlezDatasetTrain
        A class containing the train data

    ShlezDatasetTest
        A class containing the test data
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class ShlezDatasetTrain(Dataset):
    """
        Data class for the training data set (X and S pairs)
        Creates fields for X data and S data and calculates the mean and
        and variance of the X data (to be used for creation of a codebook)
    """

    def __init__(self, x_data, s_data):
        # Setting input and output dimensions (defined in ProjectConstants.py)
        self.inputDim = x_data.shape[1]
        self.outputDim = s_data.shape[1]
        # Expected value estimation
        self.X_mean = np.mean(x_data)
        self.S_mean = np.mean(s_data)
        # Variance estimation
        self.X_var = np.mean((x_data - self.X_mean) * (x_data - self.X_mean))
        self.S_var = np.mean((s_data - self.S_mean) * (s_data - self.S_mean))
        # Converting numpy arrays to pytorch tensors:
        # noinspection PyUnresolvedReferences
        self.X_data = torch.from_numpy(x_data)
        # noinspection PyUnresolvedReferences
        self.S_data = torch.from_numpy(s_data)

        # Number of X, S couples:
        self.len = s_data.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.S_data[index]

    def __len__(self):
        return self.len


class ShlezDatasetTest(Dataset):
    """ Data class for the testing data set (X and S pairs) """

    def __init__(self, x_data, s_data):

        # Converting numpy arrays to pytorch tensors:
        # noinspection PyUnresolvedReferences
        self.X_data = torch.from_numpy(x_data)
        # noinspection PyUnresolvedReferences
        self.S_data = torch.from_numpy(s_data)

        # Number of X, S couples:
        self.len = s_data.shape[0]

    def __getitem__(self, index):
        return self.X_data[index], self.S_data[index]

    def __len__(self):
        return self.len
