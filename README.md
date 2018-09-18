## Deep Quantization for Channel Estimation in Massive MIMO Systems

This is the source code implementing the neural networks (NN's) used in numerical study described in M.Shohat, G.Tsintsadze, N.Shlezinger, and Y.C. Eldar paper, entitled the same.

The repository contains all the essential code to perform the following:
- Create a dedicated dataset which can be used for training and testing the NN (MATLAB).
- Use it in the learning proccess of the NN in one of the two methods described in the paper (PYTHON).
- Obtain the results of tests, including saving, displaying and editing (all results are saved and menaged in MATLAB .mat file, via PYTHON code).


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

**Note!**
  If a .mat file with the name you chose already exists, you will be notified.
  If you choose to replace it, BE SURE you want raplace it.



After creating the .mat file above, you are ready to run the NN.



### Running



Generally speaking, the only two file you will interact with (before inspecting the NN results) are:
- main.py
- ProjectConstants.py
And the only one you will have to edit is ProjectConstants.py. This file consists of three small parts:

#### Training, Testing and Logging Data

Here you define the names of the .mat file, the code handles. The first one:
```
DATA_MAT_FILE = 'data.mat'
```
Is the file you created in the section [Creating Data](####creating-cata) above.
