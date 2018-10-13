## Deep Quantization for Channel Estimation in Massive MIMO Systems

This is the source code implementing the neural networks (NN's) used in the numerical study described in M.Shohat, G.Tsintsadze, N.Shlezinger, and Y.C. Eldar paper, entitled the same.

The repository contains all the essential code to perform the following:
- Create a dedicated dataset which can be used for training and testing the NN (MATLAB).
- Use it in the learning process of the NN in one of the two methods described in the paper (PYTHON).
- Obtain the results of tests, including saving, displaying and editing (all results are saved and managed in MATLAB .mat file, via PYTHON code).


------

### Setup



#### PYTHON Packages

In order to be able to run all PYTHON code in this project you will need the following packages:

- [PyTorch](https://pytorch.org/get-started/locally/)
- [NumPy and SciPy](https://scipy.org/install.html)
- [SymPy](https://docs.sympy.org/latest/install.html) (Install using `conda install sympy`)
- [Matplotlib](https://matplotlib.org) (Should be already installed with PYTHON)

Once you have the packages above installed, you should download the project .zip and unzip it on your system.




#### Creating Data

Before running the NN, you have to create a MATLAB .mat file which will hold the data needed to train and test the NN and few vectors which will be used to create the theoretical bounds while displaying the NN results.


For this purpose, open the project directory in MATLAB, and run the following function in the command window:
```
generateData('Filename', 'data')
```
This will create the described above file with the name 'data.mat' in the project directory, using default parameters. If you wish to choose the saving location or define your own parameters you can do it using optional Name-Value pairs, for example:
```
generateData('Filename', 'data', 'Autosave', 'off', 'Users', 4, 'Antennas', 10)
```
This will let you choose the saving directory and create the data with specified numbers of users and antennas.
For further information refer the function documentation by typing `doc generateData` in the command window.

_**Note!**_
  If a .mat file with the name you chose already exists, you will be notified.
  If you choose to replace it, BE SURE you want replace it.


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

For information about all possible interaction with a test log .mat file see testLogger.py help using python console (from the project directory):
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

Make sure to define all the parameters except `BATCH_SIZE` as lists even if they have only one element (so they can be itterable). For example: `EPOCH_RANGE = [5, ]`


------

### Running


To run the NN to specified in [Models to Activate](#models-to-activate) call the main.py file in your console:
```
python main.py
```
Make sure that you are in the project directory.


------

### Inspect Results


Each time a test is finished, you will see its summary in the console which will resemble this:

```
====================================================================

	Results of 'Soft to Hard Quantization' Testing

_________________________________
|	Training Parameters	|
|				|
| - Epochs number:	10	|
| - Learning Rate:	0.2	|
| - Codebook Size:	6	|
| - MAGIC_C:		5	|
|_______________________________|

Rate:	0.861654166907052
Average Loss:	0.16809970140457153
====================================================================
```

Followed by an indication that a new test was logged to the `TEST_LOG_MAT_FILE` defined in [Training, Testing and Logging Data](#training-testing-and-logging-data) which will be:

```
Logged test number  2 	in testlogger: tempTestLog.mat
```

As you want to check the results and compare them to the theoretical bounds, you will use the file 'ShowResults.py' from the project directory. Here you will interact with lines 81-91 with the function calls described bellow:


Delete one or several tests:

```
log.delete([1, 5, 9])
```

_**Note!**_
  PAY ATTENTION TO THE DELETE FUNCTION! IT DELETES THE TEST LOG! COMMENT IT!
  
Show content of one or several tests:

```
log.content('last')
```

Plot all the loss vs. rate results of the test, in respect to the theoretical bounds (the results should be in the cyan area):

```
log.plot()
```

Plot the soft and hard quantization layer of **one** of the runs (relevant only for Soft to Hard Quantization method):

```
plotTanhFunction(log, 1)
```


For full information about the abilities of the functions delete() content() and plot() of the testlogger, refer test logger documentation (as mentioned earlier):
```
from testLogger import *
help(testlogger)
```
