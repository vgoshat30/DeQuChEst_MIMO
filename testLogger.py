import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os.path


class TestLogger:
    """Test Logger

    This module deals with logging simulation test into a MATLAB .mat file and can:
        -   Create .mat file ready to be handled by the testLogger class
        -   Edit a suitable .mat file (including creating, editing and deleting)
        -   Display the content of a certain testLogger, by printing to the command
            prompt or by plotting results on a figure, in respective to theoretical
            bounds.
    An intuition on the usage of this module will be as follows:
        1.  Create a .mat file which will be the log. All information on your
            simulation results will be saved there. Be sure to use createMatFile().
        2.  Once you have a .mat file with correct specifications to serve as a test
            log, "connect" it to a python variable, using TestLogger().
            This will create a testLogger class object.
        3.  Modify the testLogger class object using its available properties.
            It will automatically update the .mat file.
    """

    def __init__(self, name):
        """Connect between a .mat file and a python variable

        When called, the __init__ function reads the specified .mat file and
        creates all the fields relevant for manipulating this .mat file.
        According to the loggerType (specified in a variable in the .mat
        file, which is set using the createMatFile() function) additional
        fields are created (for example, aCoefs, bCoefs, cCoefs for a testlogger
        of type 'tanh')

        Parameters
        ----------
        name : str
            Name of the .mat file to be manipulated

        Returns
        -------
        class 'TestLogger.TestLogger'
            A variable containing all the fields created.

        Example
        -------
        from testLogger import *
        myLog = TestLogger('tanhLog.mat')
        myLog
        """
        # mat. file name
        self.filename = name

        try:
            # Load data from file
            python_mat_file = sio.loadmat(self.filename)
        except IOError:
            raise IOError("File '" + self.filename + "' does not exist.")

        # Resulted rate of the test
        self.rate = python_mat_file['rateResults']
        # Resulted loss of the test
        self.loss = python_mat_file['lossResults']
        # Number of codewords
        self.codewordNum = python_mat_file['codewordNum']
        # Learning rate multiplier in the learning algorithm
        self.learningRate = python_mat_file['learningRate']
        # Dimension rations of the NN layers
        self.layersDim = python_mat_file['layersDim']
        # Training runtime
        self.runtime = python_mat_file['runTime']
        # The time the logging was performed
        self.logtime = python_mat_file['time']
        # Name of the used algorithm
        self.algorithm = python_mat_file['algorithmName']
        # Number of training epochs
        self.epochs = python_mat_file['trainEpochs']
        # A note about the test
        self.note = python_mat_file['notes']
        # Amplitudes of sum of tanh function (a coefficients)
        self.aCoefs = python_mat_file['aCoefs']
        # Shifts of sum of tanh function (b coefficients)
        self.bCoefs = python_mat_file['bCoefs']
        # "Slopes" of sum of tanh function (c coefficients)
        self.cCoefs = python_mat_file['cCoefs']
        # Multiplier of the tanh argument
        self.magic_c = python_mat_file['magic_c']

        # Training and testing data parameters:

        # Test set size
        self.s_fD = python_mat_file['s_fD']
        # Number of antennas
        self.s_fNt = python_mat_file['s_fNt']
        # Number of users
        self.s_fNu = python_mat_file['s_fNu']
        # Ratio
        self.s_fRatio = python_mat_file['s_fRatio']
        # Train set size
        self.s_fT = python_mat_file['s_fT']
        # Test power
        self.s_fTestPower = python_mat_file['s_fTestPower']
        # Train powers
        self.v_fTrainPower = python_mat_file['v_fTrainPower']

        # If the test log is not empty, reducing the dimensions of all
        # parameters to one
        if self.rate.shape[0]:
            self.rate = self.rate[0]
            self.loss = self.loss[0]
            self.codewordNum = self.codewordNum[0]
            self.learningRate = self.learningRate[0]
            self.layersDim = self.layersDim[0]
            self.runtime = self.runtime[0]
            self.logtime = self.logtime[0]
            self.algorithm = self.algorithm[0]
            self.epochs = self.epochs[0]
            self.note = self.note[0]
            self.aCoefs = self.aCoefs[0]
            self.bCoefs = self.bCoefs[0]
            self.cCoefs = self.cCoefs[0]
            self.magic_c = self.magic_c[0]
            self.s_fD = self.s_fD[0]
            self.s_fNt = self.s_fNt[0]
            self.s_fNu = self.s_fNu[0]
            self.s_fRatio = self.s_fRatio[0]
            self.s_fT = self.s_fT[0]
            self.s_fTestPower = self.s_fTestPower[0]
            self.v_fTrainPower = self.v_fTrainPower[0]

    def add_empty_test(self):
        """Append empty elements to all log fields

        The function appends an empty element to each field of the testLogger,
        and to special fields if needed (according to self.loggerType).
        If there is already empty test at the end of the testlogger, the
        function will not add additional one.

        Example
        -------
        (Additional function __init__() and content() were used in the example)

        from testLogger import *
        myLog = TestLogger('tanhLog.mat')
        myLog.content()
        myLog.add_empty_test()
        myLog.content('all')

        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

        Content of testlogger 'tempTestLog.mat'

         _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_



        Test 1 Info

        Rate:		    0.0
        Loss:		    0.0
        Codewords Num.: ________________
        Algorithm:	    ________________
        Learning rate:	________________
        Layers Dim.:	________________
        Tanh a coeffs:	________________
        Tanh b coeffs:	________________
        Tanh c coeffs:	________________
        MAGIC_C:	    ________________
        Train Runtime: 	________________
        Train Epochs:	________________
        Logging Time: 	2018-10-13 16:28:35.505715
        Note:		    ________________

        Train and test data parameters:

        Train set size (s_fT):		    ________________
        Test set size (s_fD):		    ________________
        Antennas num. (s_fNt):		    ________________
        Users num. (s_fNu):		        ________________
        Ratio (s_fRatio):	          	________________
        Test power (s_fTestPower):  	________________
        Train power (v_fTrainPower):	________________

        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
        """
        # Append only if there is no empty test at the end of the log already
        if self.rate.all():
            # Append 0 to rate results
            self.rate = np.append(self.rate, 0)
            # Append empty ndarray of float to loss
            self.loss = np.append(self.loss, np.empty((1, 1), float))
            # Append empty ndarray of float to codewordNum
            self.codewordNum = np.append(self.codewordNum,
                                         np.empty((1, 1), float))
            # Append empty ndarray of float to learningRate
            self.learningRate = np.append(self.learningRate,
                                          np.empty((1, 1), float))
            # Append empty ndarray of float to layersDim
            self.layersDim = np.append(self.layersDim, np.empty((1, 1), float))
            # Append empty string to runtime
            self.runtime = np.append(self.runtime, '')
            # Set the logging time to current time
            self.logtime = np.append(self.logtime, str(datetime.now()))
            # Append empty string to algorithm name
            self.algorithm = np.append(self.algorithm, '')
            # Append empty ndarray of float to number of training epochs
            self.epochs = np.append(self.epochs, np.empty((1, 1), float))
            # Append empty string to note
            self.note = np.append(self.note, '')
            # Append empty ndarray of float
            self.aCoefs = np.append(self.aCoefs, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.bCoefs = np.append(self.bCoefs, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.cCoefs = np.append(self.cCoefs, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.magic_c = np.append(self.magic_c, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.s_fD = np.append(self.s_fD, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.s_fNt = np.append(self.s_fNt, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.s_fNu = np.append(self.s_fNu, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.s_fRatio = np.append(self.s_fRatio, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.s_fT = np.append(self.s_fT, np.empty((1, 1), float))
            # Append empty ndarray of float
            self.s_fTestPower = np.append(self.s_fTestPower,
                                          np.empty((1, 1), float))
            # Append empty ndarray of float
            self.v_fTrainPower = np.append(self.v_fTrainPower,
                                           np.empty((1, 1), float))

    def log(self, test=None, **kwargs):
        """Manipulate content of a log element

        Use this function to add a new test or edit an existing one. Also can be
        used with the add_empty_test() function to create an empty test and edit
        it later on.

        Parameters
        ----------
            test : int OR str , optional
                if int:
                    Number of test to edit
                if str:
                    'last' - to edit last test in the testLogger (empty or not)

            **kwargs (for all testlogger types)
                rate : float
                    The rate used in the test
                loss : float
                    Resulting loss of the test
                codewordNum : int
                    Number of quantization codewords
                learningRate : float
                    The learning rate of the learning algorithm
                layersDim: list
                    Dimension ratios between layers of the NN
                runtime : datetime.timedelta
                    Time spent on the learning proccess
                algorithm : str
                    Name of the algorithm used
                epochs : int
                    Number of train epochs
                note : str
                    A note to the test. Contains anything you think is important
                a : list
                    The amplitude coefficients of the sum of tanh function.
                    Pass as a list even if there is only one element.
                    For example: mylog.log(a=[1, ])
                b : list
                    The shifts of the sum of tanh function.
                    Pass as a list even if there is only one element
                    (see aCoefs above)
                c : list
                    The "slopes" of the sum of tanh function.
                    Pass as a list even if there is only one element
                    (see aCoefs above)
                magic_c : float
                    Multiplier of the argument of the sum of tanh function.
                dataFile : str
                    Name of the data .mat file used for the training and testing
                    Specify with .mat extention, for example: 'data.mat'

        Example
        -------
        from testLogger import *
        from datetime import datetime
        before = datetime.now()
        after = datetime.now() - before
        myLog = TestLogger('tanhLog.mat')
        myLog.log(rate=0.3, loss=0.05, runtime=after)
        myLog.log(rate=0.4, loss=0.03, codewordNum=8, epochs=3)
        myLog.log('last', runtime=after, a=[-1.53, 0.24, 2.42], b=[-2.12, -0.97, 3.01])
        myLog.log(1, note='Didnt perform quantization correctly')
        """

        # If no parameters to log were specified
        if not kwargs:
            self.exception('log', 'ValueError')
            return

        # If test number not specified, creating new empty test
        if not test:
            # Add empty slot if needed
            self.add_empty_test()
            # Setting test index to the last index in the test log
            test_index = len(self.rate) - 1
            print("Created test number", test_index+1,
                  "\tin testlogger:", self.filename)
        # If requested to log to the last test
        elif test is 'last':
            test_index = len(self.rate) - 1
        # If specified test number
        elif type(test) is int:
            # Setting test index to the index specified by 'test'
            test_index = test - 1

            # If test number is larger than existing number of tests by exactly
            # one, creating new empty test and logging to it
            if test_index is len(self.rate):
                self.add_empty_test()
                print("Created test number", test_index+1,
                      "\tin testlogger:", self.filename)
            # If test number is larger than existing number of tests by more
            # than one, returning.
            elif test_index > len(self.rate):
                self.exception('log', 'IndexError', test)
                return
        # If test is not a known codeword or not an integer, returning
        else:
            self.exception('log', 'TypeError', test)
            return

        # Check what optional arguments were passed
        for key in kwargs:
            # Check if rate provided
            if key is 'rate':
                self.rate[test_index] = kwargs[key]
            if key is 'loss':
                self.loss[test_index] = kwargs[key]
            if key is 'codewordNum':
                self.codewordNum[test_index] = kwargs[key]
            if key is 'learningRate':
                self.learningRate[test_index] = kwargs[key]
            if key is 'layersDim':
                self.layersDim[test_index] = kwargs[key]
            if key is 'runtime':
                self.runtime[test_index] = str(kwargs[key])
            if key is 'algorithm':
                self.algorithm[test_index] = kwargs[key]
            if key is 'epochs':
                self.epochs[test_index] = kwargs[key]
            if key is 'note':
                self.note[test_index] = kwargs[key]
            if key is 'a':
                self.aCoefs[test_index] = kwargs[key]
            if key is 'b':
                self.bCoefs[test_index] = kwargs[key]
            if key is 'c':
                self.cCoefs[test_index] = kwargs[key]
            if key is 'magic_c':
                self.magic_c[test_index] = kwargs[key]
            if key is 'dataFile':
                # Getting train and test data parameters from the data .mat file
                pythonDataFile = sio.loadmat(kwargs[key])

                self.s_fD[test_index] = pythonDataFile['s_fD']
                self.s_fNt[test_index] = pythonDataFile['s_fNt']
                self.s_fNu[test_index] = pythonDataFile['s_fNu']
                self.s_fRatio[test_index] = pythonDataFile['s_fRatio']
                self.s_fT[test_index] = pythonDataFile['s_fT']
                self.s_fTestPower[test_index] = pythonDataFile['s_fTestPower']
                self.v_fTrainPower[test_index] = pythonDataFile['v_fTrainPower']

        self.save()
        print("Logged test number ", test_index+1,
              "\tin testlogger:", self.filename)

    def delete(self, test=None, indexing=None):
        '''Delete specific test or clear all test log

        Parameters
        ----------
            test : int OR list OR str , optional
                if int:
                    Deleting specified test
                if list:
                    Deleting all test specified in the list
                if str:
                    'last' - deleting last test
                if empty:
                    Clearing all test log

            indexing : str
                'except':
                    Delete all tests in the testLogger except specified
                'from'
                    Delete all tests from specified test (including)
                    In this case, test MUST be an int.
                'to'
                    Delete all tests to specified test (including)
                    In this case, test MUST be an int.
                if empty:
                    Deleteing test specified by the parameter test

        Example
        -------
        >>> from testLogger import *
        >>> myLog = TestLogger('tanhLog.mat')
        >>> for ii in range(1, 11):
        ...     myLog.log(rate=0.1*ii, loss=0.01*ii)
        ...
        Created test number 1 	in testlogger: tanhLog.mat
        Logged test number  1 	in testlogger: tanhLog.mat

        NOT PART OF THE PROMT: same printings until test 10:

        Created test number 10 	in testlogger: tanhLog.mat
        Logged test number  10 	in testlogger: tanhLog.mat
        >>> myLog.delete(5)
        Deleted test(s): 5
        From testlogger: tanhLog.mat
        >>> myLog.delete(7, 'from')
        Deleted test(s): [7 8 9]
        From testlogger: tanhLog.mat
        >>> myLog.delete(3, 'to')
        Deleted test(s): [1 2 3]
        From testlogger: tanhLog.mat
        >>> myLog.delete([1, 3], 'except')
        Deleted test(s): [2]
        From testlogger: tanhLog.mat
        >>> myLog.delete()
        Cleared testlogger: tanhLog.mat
        '''

        # If specified number of test(s)
        if not(test is None):
            # If specified few tests
            if type(test) is list:
                testIndex = []
                for ii in test:
                    # If test number is too large
                    if ii > len(self.rate):
                        self.exception('delete', 'IndexError', ii)
                        return
                    testIndex.append(ii-1)
            # If specified only one test
            elif type(test) is int:
                # If test number is too large
                if test > len(self.rate):
                    self.exception('delete', 'IndexError', test)
                    return
                testIndex = test - 1
            elif type(test) is str:
                if test is 'last':
                    testIndex = len(self.rate)-1
                else:
                    self.exception('delete', 'TypeErrorBadStr', test)
                    return
            else:
                self.exception('delete', 'TypeErrorNotInt', test)
                return

            # If no special kind of indexing specified
            if indexing is None:
                deleteIndex = testIndex
            # If the indexing is 'except', deleteIndex are all the indecies in
            # self.rate except the specified ones
            elif indexing is 'except':
                deleteIndex = list(range(0, len(self.rate)))
                if type(testIndex) is int:
                    deleteIndex.remove(testIndex)
                else:
                    for ii in testIndex:
                        print('DEBUGGING 1:', ii)
                        deleteIndex.remove(ii)
            # If the indexing is 'from', deleteIndex are all the indecies from
            # specified number to the end
            elif indexing is 'from':
                if type(test) is not int:
                    self.exception('delete', 'TypeErrorFrom', test)
                    return
                deleteIndex = range(test-1, len(self.rate))
            elif indexing is 'to':
                if type(test) is not int:
                    self.exception('delete', 'TypeErrorTo', test)
                    return
                deleteIndex = range(0, test)

            self.rate = np.delete(self.rate, deleteIndex, 0)
            self.loss = np.delete(self.loss, deleteIndex, 0)
            self.codewordNum = np.delete(self.codewordNum, deleteIndex, 0)
            self.learningRate = np.delete(self.learningRate, deleteIndex, 0)
            self.layersDim = np.delete(self.layersDim, deleteIndex, 0)
            self.runtime = np.delete(self.runtime, deleteIndex, 0)
            self.logtime = np.delete(self.logtime, deleteIndex, 0)
            self.algorithm = np.delete(self.algorithm, deleteIndex, 0)
            self.epochs = np.delete(self.epochs, deleteIndex, 0)
            self.note = np.delete(self.note, deleteIndex, 0)
            self.aCoefs = np.delete(self.aCoefs, deleteIndex, 0)
            self.bCoefs = np.delete(self.bCoefs, deleteIndex, 0)
            self.cCoefs = np.delete(self.cCoefs, deleteIndex, 0)
            self.magic_c = np.delete(self.magic_c, deleteIndex, 0)
            self.s_fD = np.delete(self.s_fD, deleteIndex, 0)
            self.s_fNt = np.delete(self.s_fNt, deleteIndex, 0)
            self.s_fNu = np.delete(self.s_fNu, deleteIndex, 0)
            self.s_fRatio = np.delete(self.s_fRatio, deleteIndex, 0)
            self.s_fT = np.delete(self.s_fT, deleteIndex, 0)
            self.s_fTestPower = np.delete(self.s_fTestPower, deleteIndex, 0)
            self.v_fTrainPower = np.delete(self.v_fTrainPower, deleteIndex, 0)

            print('Deleted test(s):', np.array(deleteIndex) + 1,
                  "\nFrom testlogger:", self.filename)
        else:
            self.rate = np.empty((0, 1), float)  # MATLAB Array of doubles
            self.loss = np.empty((0, 1), float)  # MATLAB Array of doubles
            self.codewordNum = np.empty((0, 1), float)  # MATLAB Array
            self.learningRate = np.empty((0, 1), float)  # MATLAB Array
            self.layersDim = np.empty((0, 1), object)  # MATLAB Cell
            self.runtime = np.empty((0, 1), object)  # MATLAB Cell
            self.logtime = np.empty((0, 1), object)  # MATLAB Cell
            self.algorithm = np.empty((0, 1), object)  # MATLAB Cell
            self.epochs = np.empty((0, 1), float)  # MATLAB Array of doubles
            self.note = np.empty((0, 1), object)  # MATLAB Cell
            self.aCoefs = np.empty((0, 1), object)  # MATLAB Cell
            self.bCoefs = np.empty((0, 1), object)  # MATLAB Cell
            self.cCoefs = np.empty((0, 1), object)  # MATLAB Cell
            self.magic_c = np.empty((0, 1), float)  # MATLAB Array
            self.s_fD = np.empty((0, 1), float)  # MATLAB Array
            self.s_fNt = np.empty((0, 1), float)  # MATLAB Array
            self.s_fNu = np.empty((0, 1), float)  # MATLAB Array
            self.s_fRatio = np.empty((0, 1), float)  # MATLAB Array
            self.s_fT = np.empty((0, 1), float)  # MATLAB Array
            self.s_fTestPower = np.empty((0, 1), float)  # MATLAB Array
            self.v_fTrainPower = np.empty((0, 1), object)  # MATLAB Cell

            print('Cleared testlogger:', self.filename)

        self.save()

    def content(self, test=None):
        '''Show the content of the testlogger

        Prints all information available for a certain test, multiple tests or
        all the tests in a testlogger.

        Parameters
        ----------
            test : int OR list OR str, optional
                if int:
                    Printing content of specified test
                if list:
                    Printing content of specified tests
                if str:
                    'all'  - Printing content of all tests
                    'last' - Printing content of last logged test
                if empty:
                    Printing short info about the testlogger itself
                    (filename, number of logged tests and testlogger type)

        Example
        -------
        >>> from testLogger import *
        >>> from datetime import datetime
        >>> before = datetime.now()
        >>> after = datetime.now() - before
        >>> myLog = TestLogger('tanhLog.mat')
        >>> myLog.log(rate=0.3, loss=0.05, runtime=after)
        Created test number 1 	in testlogger: tanhLog.mat
        Logged test number  1 	in testlogger: tanhLog.mat
        >>> myLog.log(runtime=after, a=[-1.53, 0.24, 2.42],
        ...           b=[-2.12, -0.97, 3.01])
        Created test number 2 	in testlogger: tanhLog.mat
        Logged test number  2 	in testlogger: tanhLog.mat
        >>> myLog.log('last', rate=0.2, loss=0.03, note='Content example')
        Logged test number  2 	in testlogger: tanhLog.mat
        >>> myLog.content()
        The testlogger 'tanhLog.mat' contains 2 tests.
        >>> myLog.content(2)
        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

        Content of testlogger 'tempTestLog.mat'

         _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_



        Test 2 Info

        Rate:		    0.2
        Loss:		    0.03
        Codewords Num.: 0.05
        Algorithm:	    ________________
        Learning rate:	________________
        Layers Dim.:	________________
        Tanh a coeffs:	________________
        Tanh b coeffs:	________________
        Tanh c coeffs:	________________
        MAGIC_C:	    ________________
        Train Runtime: 	0:00:00.000011
        Train Epochs:	________________
        Logging Time: 	2018-10-13 16:38:40.383540
        Note:		    Content example

        Train and test data parameters:

        Train set size (s_fT):		    ________________
        Test set size (s_fD):		    ________________
        Antennas num. (s_fNt):		    ________________
        Users num. (s_fNu):		        ________________
        Ratio (s_fRatio):		        ________________
        Test power (s_fTestPower):	    ________________
        Train power (v_fTrainPower):	________________

        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
        '''

        dontExistMessage = '________________'

        if test is None:
            print("The testlogger '" + self.filename +
                  "' contains", len(self.rate), "tests.")
        else:
            if type(test) is int:
                if test > len(self.rate):
                    self.exception('content', 'IndexError', test)
                    return
                testNum = [test, ]
            elif type(test) is list:
                for ii in test:
                    if ii > len(self.rate):
                        self.exception('content', 'IndexError', ii)
                        return
                testNum = test
            elif type(test) is str:
                if test is 'all':
                    if len(self.rate):
                        testNum = range(1, len(self.rate)+1)
                    else:
                        print("The testlogger", self.filename, "is empty.")
                        return
                elif test is 'last':
                    if len(self.rate):
                        testNum = [len(self.rate), ]
                    else:
                        print("The testlogger", self.filename, "is empty.")
                        return
                else:
                    self.exception('content', 'TypeError', test)
                    return
            else:
                self.exception('content', 'TypeError', test)
                return

            print("\n\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/"
                  "\n\nContent of testlogger '" + self.filename + "'\n\n",
                  "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n")
            for ii, currTest in enumerate(testNum):
                if not self.codewordNum[currTest-1]:
                    codewordNumToPrint = dontExistMessage
                else:
                    codewordNumToPrint = self.codewordNum[currTest-1]

                if not self.algorithm[currTest-1]:
                    algToPrint = dontExistMessage
                else:
                    algToPrint = removeCellFormat(self.algorithm[currTest-1])

                if not self.learningRate[currTest-1]:
                    learningRateToPrint = dontExistMessage
                else:
                    learningRateToPrint = self.learningRate[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (len(self.layersDim[currTest-1]) == 1 and
                        self.layersDim[currTest-1][0, 0] == 0)):
                    layersDimToPrint = dontExistMessage
                else:
                    layersDimToPrint = self.layersDim[currTest-1]

                if not self.runtime[currTest-1]:
                    runtimeToPrint = dontExistMessage
                else:
                    runtimeToPrint = removeCellFormat(self.runtime[currTest-1])

                if not self.epochs[currTest-1]:
                    epochToPrint = dontExistMessage
                else:
                    epochToPrint = self.epochs[currTest-1]

                if not self.note[currTest-1]:
                    noteToPrint = dontExistMessage
                else:
                    noteToPrint = removeCellFormat(self.note[currTest-1])

                if (type(self.layersDim[currTest-1]) is float or
                    (self.aCoefs[currTest-1].size == 1 and
                        self.aCoefs[currTest-1][0, 0] == 0)):
                    aCoefsToPrint = dontExistMessage
                else:
                    aCoefsToPrint = self.aCoefs[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (self.bCoefs[currTest-1].size == 1 and
                        self.bCoefs[currTest-1][0, 0] == 0)):
                    bCoefsToPrint = dontExistMessage
                else:
                    bCoefsToPrint = self.bCoefs[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (self.cCoefs[currTest-1].size == 1 and
                        self.cCoefs[currTest-1][0, 0] == 0)):
                    cCoefsToPrint = dontExistMessage
                else:
                    cCoefsToPrint = self.cCoefs[currTest-1]

                if not self.magic_c[currTest-1]:
                    magicCToPrint = dontExistMessage
                else:
                    magicCToPrint = self.magic_c[currTest-1]

                if not self.s_fD[currTest-1]:
                    s_fDToPrint = dontExistMessage
                else:
                    s_fDToPrint = self.s_fD[currTest-1]

                if not self.s_fNt[currTest-1]:
                    s_fNtToPrint = dontExistMessage
                else:
                    s_fNtToPrint = self.s_fNt[currTest-1]

                if not self.s_fNu[currTest-1]:
                    s_fNuToPrint = dontExistMessage
                else:
                    s_fNuToPrint = self.s_fNu[currTest-1]

                if not self.s_fRatio[currTest-1]:
                    s_fRatioToPrint = dontExistMessage
                else:
                    s_fRatioToPrint = self.s_fRatio[currTest-1]

                if not self.s_fT[currTest-1]:
                    s_fTToPrint = dontExistMessage
                else:
                    s_fTToPrint = self.s_fT[currTest-1]

                if not self.s_fTestPower[currTest-1]:
                    s_fTestPowerToPrint = dontExistMessage
                else:
                    s_fTestPowerToPrint = self.s_fTestPower[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (self.v_fTrainPower[currTest-1].size == 1 and
                        self.v_fTrainPower[currTest-1][0, 0] == 0)):
                    v_fTrainPowerToPrint = dontExistMessage
                else:
                    v_fTrainPowerToPrint = self.v_fTrainPower[currTest-1]

                print("\n\nTest {} Info\n\n"
                      "Rate:\t\t{}\n"
                      "Loss:\t\t{}\n"
                      "Codewords Num.: {}\n"
                      "Algorithm:\t{}\n"
                      "Learning rate:\t{}\n"
                      "Layers Dim.:\t{}\n"
                      "Tanh a coeffs:\t{}\n"
                      "Tanh b coeffs:\t{}\n"
                      "Tanh c coeffs:\t{}\n"
                      "MAGIC_C:\t{}\n"
                      "Train Runtime: \t{}\n"
                      "Train Epochs:\t{}\n"
                      "Logging Time: \t{}\n"
                      "Note:\t\t{}\n\n"
                      "Train and test data parameters:\n\n"
                      "Train set size (s_fT):\t\t{}\n"
                      "Test set size (s_fD):\t\t{}\n"
                      "Antennas num. (s_fNt):\t\t{}\n"
                      "Users num. (s_fNu):\t\t{}\n"
                      "Ratio (s_fRatio):\t\t{}\n"
                      "Test power (s_fTestPower):\t{}\n"
                      "Train power (v_fTrainPower):\t{}\n"
                      .format(currTest, self.rate[currTest-1],
                              self.loss[currTest-1], codewordNumToPrint,
                              algToPrint, learningRateToPrint,
                              layersDimToPrint, aCoefsToPrint,
                              bCoefsToPrint, cCoefsToPrint, magicCToPrint,
                              runtimeToPrint, epochToPrint,
                              removeCellFormat(self.logtime[currTest-1]),
                              noteToPrint, s_fTToPrint, s_fDToPrint,
                              s_fNtToPrint, s_fNuToPrint, s_fRatioToPrint,
                              s_fTestPowerToPrint, v_fTrainPowerToPrint))

                if ii < len(testNum)-1:
                    print("---------------------------------------------------")

            print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")

    def plot(self, theoryDataFile=None):
        '''Plot all results in a testlogger

        IMPORTANT!!!
            The function opens a figure which will pause all your code until
            closed!

        Opens a new figure, plots all the theoretical bounds and all
        previously logged tests.
        The points are enumerated according to their logging order.
        Click any datapoint to open a tooltip with additional imformation.
        To see all imformation available for a test, use the function
        content()

        Parameters
        ----------
            theoryDataFile : str
                Name of MATLAB .mat file containing the variables 'v_fRate' and
                'm_fCurves' which provide theoretical bounds for the results.
                This is a file creted using the generateData.m function.
                Specify including .mat extention, for example: 'data.mat'

        Example
        -------
        The file 'data.mat' is generated using generateData.m

        >>> from testLogger import *
        >>> myLog = TestLogger('tanhLog.mat')
        >>> myLog.log(rate=0.3, loss=0.05)
        Created test number 1 	in testlogger: tanhLog.mat
        Logged test number  1 	in testlogger: tanhLog.mat
        >>> myLog.plot('data.mat')
        '''

        chosenMarkersize = 2
        regMarkerSize = 7
        indexAlpha = 0.5
        datatipAlpha = 1
        dataTipFontsize = 6
        legendFontsize = 7
        textboxAlpha = 0.8
        textOffset = 0.0005
        tickOffset = 0.001
        sizeOfFigure = (8, 5)  # in inches

        # Define which lines to plot
        whichToPlot = [1,  # No quantization
                       1,  # Asymptotic optimal task-based
                       1,  # Asymptotic optimal task-ignorant
                       1]  # Hardware limited upper bound

        # Set the legend labels
        labels = ['No quantization',
                  'Asymptotic optimal task-based',
                  'Asymptotic optimal task-ignorant',
                  'Hardware limited upper bound']

        markers = ['', '', '', '']
        lineStyles = [':', '--', '--', '--']
        linecolors = ['black', 'red', 'blue', 'lime']
        lineWidths = [1, 1, 1, 1.5]
        markerSizes = [4, 1, 1, 1]
        markerLinewidths = [1, 1, 1, 1]
        pointMarker = 'x'
        pointsColor = 'orange'
        chosenMarker = 'o'
        chosenColor = 'red'
        tooltipBoxStyle = 'round'
        tooltipBoxColor = 'wheat'
        fillColor = 'c'

        enumerateTests = True

        def pick_handler(event):
            '''Handles the choosing of plotted results

            Note
            ----
            This function is not intended for the user and is used only by other
            function in the class testLogger.
            '''
            # Get the pressed artist
            artist = event.artist

            # If the clicked point was already chosen, clear all tooltips and return
            if artist.get_marker() is chosenMarker:
                clearDatatips(event, -1)
                return

            # Mark the chosen point
            artist.set(marker=chosenMarker, markersize=chosenMarkersize,
                       color=chosenColor)

            # Get the index of the clicked point in the test log
            chosenIndex = resList.index(artist)
            # Hide current result index
            indexText[chosenIndex].set(alpha=0)
            # Show chosen texbox
            textBoxes[chosenIndex] = dict(boxstyle=tooltipBoxStyle,
                                          facecolor=tooltipBoxColor,
                                          alpha=textboxAlpha)
            # Show chosen tooltip
            tooltips[chosenIndex].set(alpha=datatipAlpha,
                                      bbox=textBoxes[chosenIndex])

            # Clear other tooltips
            clearDatatips(event, chosenIndex)

            # Update figure
            fig.canvas.draw()
            fig.canvas.flush_events()

        def clearDatatips(event, exeptInex):
            '''Clears tooltips

            Parameters
            ----------
            event : matplotlib.backend_bases.PickEvent
                Callback pick event
            exeptInex : int
                Dont clear the tooltip of the point at the specified index
                Pass -1 to clear all tooltips

            Note
            ----
            This function is not intended for the user and is used only by other
            function in the class testLogger.
            '''

            for ii in range(0, len(resList)):
                if ii is exeptInex:
                    continue
                # Unmark all other points
                resList[ii].set(marker=pointMarker, markersize=regMarkerSize,
                                color=pointsColor)
                # Show all result indecies
                indexText[ii].set(alpha=indexAlpha)
                # Hide all textBoxes
                textBoxes[ii] = dict(boxstyle=tooltipBoxStyle,
                                     facecolor=tooltipBoxColor,
                                     alpha=0)
                # Hide all result tooltips
                tooltips[ii].set(alpha=0, bbox=textBoxes[ii])

            # Update figure
            fig.canvas.draw()
            fig.canvas.flush_events()

        def getTextAlignment(x, y, textOffset):
            '''Set tooltip textbox alignment and its offset from the
            corresponding point on the graph

            Note
            ----
            This function is not intended for the user and is used only by other
            function in the class testLogger.

            Parameters
            ----------
            x : numpy.ndarray
                X position of a test log (rate)
            y : numpy.ndarray
                Y position of a test log (average loss)
            textOffset : float
                The absolute value of the offset of the tooltip from the point

            Returns
            -------
            (ha, va, xOffset, yOffset) : tuple
                ha : str
                    Horizontal alignment as accepted by the text property:
                    horizontalalignment of the matplotlib.
                    Possible values: [ 'right' | 'left' ]
                ha : str
                    Vertical alignment as accepted by the text property:
                    verticalalignment of the matplotlib.
                    Possible values: [ 'top' | 'bottom' ]
                xOffset : float
                    The X axis offset of the textbox
                yOffset : float
                    The Y axis offset of the textbox
            '''
            axs = plt.gca()
            xlim = axs.get_xlim()
            ylim = axs.get_ylim()

            if x < xlim[1] / 2:
                ha = 'left'
                xOffset = textOffset
            else:
                ha = 'right'
                xOffset = -textOffset

            if y < ylim[1]/2:
                va = 'bottom'
                yOffset = textOffset
            else:
                va = 'top'
                yOffset = -textOffset

            return (ha, va, xOffset, yOffset)

        plt.close('all')

        # Create figure and get axes handle
        fig = plt.figure(figsize=sizeOfFigure)
        ax = fig.add_subplot(111)

        # Connect figure to callback
        fig.canvas.mpl_connect('pick_event', pick_handler)
        fig.canvas.mpl_connect('figure_leave_event', lambda event,
                               temp=-1: clearDatatips(event, temp))

        if theoryDataFile:
            theory = sio.loadmat(theoryDataFile)

            # Extracting theory data vectors
            theoryRate = theory['v_fRate']
            theoryLoss = theory['m_fCurves']

            # Create fill vectors
            xFill = np.concatenate((theoryRate[0],
                                    np.flip(theoryRate[0], 0)), axis=0)
            yFill = np.concatenate((theoryLoss[3, :],
                                    np.flip(theoryLoss[1, :], 0)), axis=0)

            # Plot all theoretical bounds
            for ii in range(0, theoryLoss.shape[0]):
                if whichToPlot[ii]:
                    ax.plot(theoryRate[0], theoryLoss[ii, :],
                            label=labels[ii],
                            marker=markers[ii], color=linecolors[ii],
                            linestyle=lineStyles[ii], linewidth=lineWidths[ii],
                            markersize=markerSizes[ii])

            # Plot fill
            ax.fill(xFill, yFill, c=fillColor, alpha=0.3)

        # Plot previous results (the loop is for plotting as separate artists)
        resList = []
        for ii in range(0, len(self.rate)-1):
            if self.rate[ii]:
                resList += ax.plot(self.rate[ii], self.loss[ii], marker='x',
                                   markersize=regMarkerSize,
                                   color=pointsColor, picker=5)

        # Plot result (not plotting in the for loop above only to set the label
        # to 'Results' without affecting all result points)
        try:
            if self.rate[-1]:
                resList += ax.plot(self.rate[-1], self.loss[-1], marker='x',
                                   markersize=regMarkerSize, color=pointsColor,
                                   label='Results', picker=5)
        except IndexError:
            self.exception('plot', 'EmptyLog')

        indexText = []
        tooltips = []
        textBoxes = []
        for ii in range(0, len(self.rate)):
            # If the current iterated test is not empty
            if self.rate[ii]:
                # Enumerate result points in the figure
                if enumerateTests:
                    indexText.append(ax.text(self.rate[ii] + tickOffset,
                                             self.loss[ii] + tickOffset,
                                             ii+1, fontsize=8, alpha=indexAlpha,
                                             verticalalignment='center',
                                             horizontalalignment='center'))
                # Create all textboxes and dont display them
                textBoxes.append(dict(boxstyle='round', facecolor='wheat',
                                      alpha=0))

                # Create all data tooltip and dont display them
                currIterDateTime = datetime.strptime(
                    removeCellFormat(self.logtime[ii]), '%Y-%m-%d %H:%M:%S.%f')
                if self.runtime[ii]:
                    currRuntime = datetime.strptime(removeCellFormat(
                        self.runtime[ii]),
                        '%H:%M:%S.%f')
                if self.algorithm[ii] and self.runtime[ii]:
                    textToDisplay = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nAlgorithm:\n' + \
                        removeCellFormat(self.algorithm[ii]) + \
                        '\nRuntime: ' + currRuntime.strftime('%H:%M:%S.%f') + \
                        '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                        '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
                elif self.algorithm[ii] and not(self.runtime[ii]):
                    textToDisplay = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nAlgorithm:\n' + \
                        removeCellFormat(self.algorithm[ii]) + \
                        '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                        '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
                elif not(self.algorithm[ii]) and self.runtime[ii]:
                    textToDisplay = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nRuntime: ' + currRuntime.strftime('%H:%M:%S.%f') + \
                        '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                        '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
                else:
                    textToDisplay = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nDate: ' + currIterDateTime.strftime('%d/%m/%y') + \
                        '\nTime: ' + currIterDateTime.strftime('%H:%M:%S')
                textAlign = getTextAlignment(self.rate[ii],
                                             self.loss[ii],
                                             textOffset)
                tooltips.append(ax.text(self.rate[ii]+textAlign[2],
                                        self.loss[ii]+textAlign[3],
                                        textToDisplay,
                                        alpha=0, fontsize=dataTipFontsize,
                                        bbox=textBoxes[ii],
                                        ha=textAlign[0], va=textAlign[1]))

        # Labeling and graph appearance
        plt.xlabel('Rate', fontsize=18, fontname='Times New Roman')
        plt.ylabel('Average Distortion', fontsize=18,
                   fontname='Times New Roman')
        ax.legend(fontsize=legendFontsize)
        ax.autoscale(enable=True, axis='x', tight=True)
        # Show figure
        plt.show()

    def save(self):
        '''Save a testLogger

        Note
        ----
        This function is not intended for the user and is used only by other
        function in the class testLogger.
        '''

        saveVars = {'rateResults': self.rate,
                    'lossResults': self.loss,
                    'codewordNum': self.codewordNum,
                    'learningRate': self.learningRate,
                    'layersDim': self.layersDim,
                    'time': self.logtime,
                    'algorithmName': self.algorithm,
                    'runTime': self.runtime,
                    'trainEpochs': self.epochs,
                    'notes': self.note,
                    'aCoefs': self.aCoefs,
                    'bCoefs': self.bCoefs,
                    'cCoefs': self.cCoefs,
                    'magic_c': self.magic_c,
                    's_fD': self.s_fD,
                    's_fNt': self.s_fNt,
                    's_fNu': self.s_fNu,
                    's_fRatio': self.s_fRatio,
                    's_fT': self.s_fT,
                    's_fTestPower': self.s_fTestPower,
                    'v_fTrainPower': self.v_fTrainPower}

        sio.savemat(self.filename, saveVars)

    def exception(self, property, excepType, *test):
        '''Print an exception

        Because the testlogger class is used in long simulations, you dont want
        any little mistake to throw an error and stop ypur code.
        Exceptions in this function are more informative than python exceptions
        and were easier to use in this particullar code.

        Note
        ----
        This function is not intended for the user and is used only by other
        function in the class testLogger.
        '''

        if test:
            test = test[0]

        if property is 'log':
            if excepType is 'IndexError':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tIndexError in the property log()",
                      "\n\tCause:\tTest numer", test,
                      "does not exist,",  "there are only",
                      len(self.rate), "tests.")
            if excepType is 'TypeError':
                if type(test) is str:
                    print("Error in testlogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property log()",
                          "\n\tCause:\tTest argument can only be an integer",
                          "or the string 'last', but was the string",
                          "'" + test + "'")
                else:
                    print("Error in testlogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property log()",
                          "\n\tCause:\tTest argument can only be an integer",
                          "or the string 'last', but was of type", type(test))
            if excepType is 'ValueError':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tValueError in the property log()",
                      "\n\tCause:\tThere are no input arguments to log()")

        if property is 'delete':
            if excepType is 'IndexError':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tIndexError in the property delete()",
                      "\n\tCause:\tTest numer", test, "does not exist,",
                      "there are only", len(self.rate), "tests.")
            if excepType is 'TypeErrorBadStr':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTest argument can only be an integer,",
                      "a list or the string 'last', but was the string '" +
                      test + "'")
            if excepType is 'TypeErrorNotInt':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTest argument can only be an integer",
                      "or a list, but was of type", type(test))
            if excepType is 'TypeErrorFrom':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTo delete tests 'from' certain test,",
                      "it needs to be an integer.",
                      "Instead was of type", type(test))
            if excepType is 'TypeErrorTo':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTo delete tests 'to' certain test,",
                      "it needs to be an integer.",
                      "Instead was of type", type(test))

        if property is 'content':
            if excepType is 'TypeError':
                if type(test) is str:
                    print("Error in testlogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property content()",
                          "\n\tCause:\tTest argument can only be an integer,",
                          "a list or the strings 'all' or 'last', but was the",
                          "string '" + test + "'")
                else:
                    print("Error in testlogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property content()",
                          "\n\tCause:\tTest argument can only be an integer,",
                          "a list or the string 'all', but was of type",
                          type(test))
            if excepType is 'IndexError':
                print("Error in testlogger", self.filename + ":",
                      "\n\tType:\tIndexError in the property content()",
                      "\n\tCause:\tTest numer", test,
                      "does not exist,",  "there are only",
                      len(self.rate), "tests.")

        if property is 'plot':
            if excepType is 'EmptyLog':
                print("Warning in testlogger", self.filename + ":",
                      "\n\tType:\tEmptyLog in the property plot()",
                      "\n\tCause:\tTestlogger:", self.filename,
                      "is empty. Nothing to plot...")


def createMatFile(name):
    '''Create a new .mat file that can be handled by the testlogger class

    Creates a MATLAB .mat file with all the variables required so it can be
    properly "connected" to the python class testlogger.
    Spetial variables are added, if needed, according to the type parameter.

    Parameters
    ----------
        name : str
            Name of the mat file, with extention! For example: "testLog.mat"

    Returns
    -------
        testLogger.testLogger
            The created file is connected to the testLogger class using
            testLogger() at the end of the function createMatFile().
            One can alternatively call createMatFile() without using its output
            and then call testLogger() later

    Example
    -------
    >>> from testLogger import *
    >>> myLog = createMatFile("testLog.mat")
    Created new test log 'testLog.mat'
    >>> myLog
    <testLogger.testlogger object at 0x10890de10>
    '''

    rate = np.empty((0, 1), float)  # MATLAB Array of doubles
    loss = np.empty((0, 1), float)  # MATLAB Array of doubles
    codewordNum = np.empty((0, 1), float)  # MATLAB Array of doubles
    learningRate = np.empty((0, 1), float)  # MATLAB Array of doubles
    layersDim = np.empty((0, 1), object)  # MATLAB Cell
    runtime = np.empty((0, 1), object)  # MATLAB Cell
    logtime = np.empty((0, 1), object)  # MATLAB Cell
    algorithm = np.empty((0, 1), object)  # MATLAB Cell
    epochs = np.empty((0, 1), float)  # MATLAB Array of doubles
    note = np.empty((0, 1), object)  # MATLAB Cell
    aCoefs = np.empty((0, 1), object)  # MATLAB Cell
    bCoefs = np.empty((0, 1), object)  # MATLAB Cell
    cCoefs = np.empty((0, 1), object)  # MATLAB Cell
    magic_c = np.empty((0, 1), float)  # MATLAB Array
    s_fD = np.empty((0, 1), float)  # MATLAB Array
    s_fNt = np.empty((0, 1), float)  # MATLAB Array
    s_fNu = np.empty((0, 1), float)  # MATLAB Array
    s_fRatio = np.empty((0, 1), float)  # MATLAB Array
    s_fT = np.empty((0, 1), float)  # MATLAB Array
    s_fTestPower = np.empty((0, 1), float)  # MATLAB Array
    v_fTrainPower = np.empty((0, 1), object)  # MATLAB Cell

    variables = {'rateResults': rate,
                 'lossResults': loss,
                 'codewordNum': codewordNum,
                 'learningRate': learningRate,
                 'layersDim': layersDim,
                 'time': logtime,
                 'algorithmName': algorithm,
                 'runTime': runtime,
                 'trainEpochs': epochs,
                 'notes': note,
                 'aCoefs': aCoefs,
                 'bCoefs': bCoefs,
                 'cCoefs': cCoefs,
                 'magic_c': magic_c,
                 's_fD': s_fD,
                 's_fNt': s_fNt,
                 's_fNu': s_fNu,
                 's_fRatio': s_fRatio,
                 's_fT': s_fT,
                 's_fTestPower': s_fTestPower,
                 'v_fTrainPower': v_fTrainPower}

    if os.path.exists(name):
        print("Test logger '" + name + "' already exists...")
    else:
        sio.savemat(name, variables)
        print("Created new test log '" + name + "'")
    return TestLogger(name)


def removeCellFormat(formatted):
    '''Clear a string formatting caused by saving as cell in mat file

    Note
    ----
    This function is not intended for the user and is used only by other
    function in the class testLogger.
    '''
    unformatted = str(formatted).replace("[", "")
    unformatted = unformatted.replace("]", "")
    unformatted = unformatted.replace("'", "")
    return unformatted
