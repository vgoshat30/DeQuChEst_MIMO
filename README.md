## Deep Quantization for Channel Estimation in Massive MIMO Systems

This is the source code implementing the neural networks (NN's) used in the numerical study described in M.Shohat, G.Tsintsadze, N.Shlezinger, and Y.C. Eldar paper, entitled the same.

The repository contains all the essential code to perform the following:
- Create a dedicated dataset which can be used for training and testing the NN (MATLAB).
- Use it in the learning proccess of the NN in one of the two methods described in the paper (PYTHON).
- Obtain the results of tests, including saving, displaying and editing (all results are saved and menaged in MATLAB .mat file, via PYTHON code).


------

### Setup



#### PYTHON Packages

In order to be able to run all PYTHON code in this project you will need the following libraries:

- [PyTorch](https://pytorch.org)
- [NumPy and SciPy](https://scipy.org/scipylib/download.html)
- [SymPy](https://www.sympy.org/en/index.html)
- [Matplotlib](https://matplotlib.org)

Once you installed the packages above, you should download the project .zip and unzip it on your system.




#### Creating Data

Before running the NN, you have to create a MATLAB .mat file which will hold the data needed to train and test the NN and few vectors which will be used to create the theoretical bounds while displaying the NN results.


For this purpose, open the project directory in MATLAB, and run the following function in the command window:
```
generateData('Filename', 'data')
```
This will create the described above file with the name 'data.mat' in the project directory, using deafult parametrs. If you wish to choose the saving location or define your own parameters you can do it using optional Name-Value pairs, for example:
```
generateData('Filename', 'data', 'Autosave', 'off', 'Users', 4, 'Antennas', 10)
```
This will let you choose the saving diretory and create the data with specified numbers of users and antennas.
For further information reffer the function documentation by typing `doc generateData` in the command window.

_**Note!**_
  If a .mat file with the name you chose already exists, you will be notified.
  If you choose to replace it, BE SURE you want raplace it.


------

### Tuning



Generally speaking, the only two file you will interact with (before inspecting the NN results) main.py and the only one you will have to edit: ProjectConstants.py. This file consists of three small parts:

#### Training, Testing and Logging Data

Here you define the names of the .mat file, the code handles. The first one:
```
DATA_MAT_FILE = 'data.mat'
```
Is the file you created in the section [Creating Data](#creating-data) above.

The second one:
```
TEST_LOG_MAT_FILE = 'tempTestLog.mat'
```
Will contain each test performed on the trained NN. At first this file will not exist (as in the example above), in this case, a proper .mat file will be created automatically. If test results already were saved in a .mat file serving as a test logger and you want to continue adding tests to it, set `TEST_LOG_MAT_FILE` to its name.

All the test log manipulation are performed using the testLogger.py module in the project directory.

For information about all possible interaction with a test log .mat file see testLogger.py help using python consule (from the project directory):
```
from testLogger import *
help(testlogger)
```

#### Models to Activate
In this section, uncomment the name of the models you want to activate. There are only two of them - the two described in the paper.

_**Note!**_
  DO NOT change their names. Other parts of code are relying on it.
  
  
#### Neural Network constants
The constants in this section define lists of parameters for the NN to iterate on. The NN will be trained and tested for each possible combination of the parameters values.

Make sure to define all the parameters except `BATCH_SIZE` as lists even if they have only one element (so they can be iterable). For example: `EPOCH_RANGE = [5, ]`


------

### Running


To run the NN to specified in [Models to Activate](#models-to-activate) call the main.py file in your console:
```
python main.py
```
Make sure that you are in the project directory.


------

### Inspect Results
