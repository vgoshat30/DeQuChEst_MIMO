import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os.path


class TestLogger:
    """Test Logger

    This module deals with logging simulation test into a MATLAB .mat file and can:
        -   Create .mat file ready to be handled by the TestLogger class
        -   Edit a suitable .mat file (including creating, editing and deleting)
        -   Display the content of a certain TestLogger, by printing to the command
            prompt or by plotting results on a figure, in respective to theoretical
            bounds.
    An intuition on the usage of this module will be as follows:
        1.  Create a .mat file which will be the log. All information on your
            simulation results will be saved there. Be sure to use create_mat_file().
        2.  Once you have a .mat file with correct specifications to serve as a test
            log, "connect" it to a python variable, using TestLogger().
            This will create a TestLogger class object.
        3.  Modify the TestLogger class object using its available properties.
            It will automatically update the .mat file.
    """

    # Keys are names used in this code and values are names of the variables
    # in the MATLAB code. DO NOT CHANGE PYTHON VARIABLE NAMES!
    #                   PYTHON      :     MATLAB
    field_names = {'rate_results'   : 'rateResults',
                   'loss_results'   : 'lossResults',
                   'codeword_num'   : 'codewordNum',
                   'learning_rate'  : 'learningRate',
                   'layers_dim'     : 'layersDim',
                   'run_time'       : 'runTime',
                   'time'           : 'time',
                   'algorithm_name' : 'algorithmName',
                   'train_epochs'   : 'trainEpochs',
                   'notes'          : 'testNotes',
                   'a_coefs'        : 'aCoefs',
                   'b_coefs'        : 'bCoefs',
                   'c_coefs'        : 'cCoefs',
                   'c_bounds'       : 'cBounds',
                   's_fD'           : 's_fD',
                   's_fNt'          : 's_fNt',
                   's_fNu'          : 's_fNu',
                   's_fRatio'       : 's_fRatio',
                   's_fT'           : 's_fT',
                   's_fTestPower'   : 's_fTestPower',
                   'v_fTrainPower'  : 'v_fTrainPower'}

    def __init__(self, name):
        """Connect between a .mat file and a python variable

        When called, the __init__ function reads the specified .mat file and
        creates all the fields relevant for manipulating this .mat file.

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
        from TestLogger import *
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
        self.rate = python_mat_file[TestLogger.field_names['rate_results']]
        # Resulted loss of the test
        self.loss = python_mat_file[TestLogger.field_names['loss_results']]
        # Number of codewords
        self.codewordNum = python_mat_file[TestLogger.field_names['codeword_num']]
        # Learning rate multiplier in the learning algorithm
        self.learningRate = python_mat_file[TestLogger.field_names['learning_rate']]
        # Dimension rations of the NN layers
        self.layersDim = python_mat_file[TestLogger.field_names['layers_dim']]
        # Training runtime
        self.runtime = python_mat_file[TestLogger.field_names['run_time']]
        # The time the logging was performed
        self.logtime = python_mat_file[TestLogger.field_names['time']]
        # Name of the used algorithm
        self.algorithm = python_mat_file[TestLogger.field_names['algorithm_name']]
        # Number of training epochs
        self.epochs = python_mat_file[TestLogger.field_names['train_epochs']]
        # A note about the test
        self.note = python_mat_file[TestLogger.field_names['notes']]
        # Amplitudes of sum of tanh function (a coefficients)
        self.aCoefs = python_mat_file[TestLogger.field_names['a_coefs']]
        # Shifts of sum of tanh function (b coefficients)
        self.bCoefs = python_mat_file[TestLogger.field_names['b_coefs']]
        # "Slopes" of sum of tanh function (c coefficients)
        self.cCoefs = python_mat_file[TestLogger.field_names['c_coefs']]
        # Multiplier of the tanh argument
        self.magic_c = python_mat_file[TestLogger.field_names['c_bounds']]

        # Training and testing data parameters:

        # Test set size
        self.s_fD = python_mat_file[TestLogger.field_names['s_fD']]
        # Number of antennas
        self.s_fNt = python_mat_file[TestLogger.field_names['s_fNt']]
        # Number of users
        self.s_fNu = python_mat_file[TestLogger.field_names['s_fNu']]
        # Ratio
        self.s_fRatio = python_mat_file[TestLogger.field_names['s_fRatio']]
        # Train set size
        self.s_fT = python_mat_file[TestLogger.field_names['s_fT']]
        # Test power
        self.s_fTestPower = python_mat_file[TestLogger.field_names['s_fTestPower']]
        # Train powers
        self.v_fTrainPower = python_mat_file[TestLogger.field_names['v_fTrainPower']]

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

        The function appends an empty element to each field of the TestLogger,
        and to special fields if needed (according to self.loggerType).
        If there is already empty test at the end of the TestLogger, the
        function will not add additional one.

        Example
        -------
        (Additional function __init__() and content() were used in the example)

        from TestLogger import *
        myLog = TestLogger('tanhLog.mat')
        myLog.content()
        myLog.add_empty_test()
        myLog.content('all')

        \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

        Content of TestLogger 'tempTestLog.mat'

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
                    'last' - to edit last test in the TestLogger (empty or not)

            **kwargs (for all TestLogger types)
                'rate' : float
                    The rate used in the test
                'loss' : float
                    Resulting loss of the test
                'codewordNum' : int
                    Number of quantization codewords
                'learningRate' : float
                    The learning rate of the learning algorithm
                'layersDim' : list
                    Dimension ratios between layers of the NN
                'runtime' : datetime.timedelta
                    Time spent on the learning process
                'algorithm' : str
                    Name of the algorithm used
                'epochs' : int
                    Number of train epochs
                'note' : str
                    A note to the test. Contains anything you think is important
                'a' : list
                    The amplitude coefficients of the sum of tanh function.
                    Pass as a list even if there is only one element.
                    For example: mylog.log(a=[1, ])
                'b' : list
                    The shifts of the sum of tanh function.
                    Pass as a list even if there is only one element
                    (see aCoefs above)
                'c' : list
                    The "slopes" of the sum of tanh function.
                    Pass as a list even if there is only one element
                    (see aCoefs above)
                'magic_c' : float
                    Multiplier of the argument of the sum of tanh function.
                'dataFile' : str
                    Name of the data .mat file used for the training and testing
                    Specify with .mat extension, for example: 'data.mat'

        Example
        -------
        from TestLogger import *
        from datetime import datetime
        before = datetime.now()
        after = datetime.now() - before
        myLog = TestLogger('tanhLog.mat')
        myLog.log(rate=0.3, loss=0.05, runtime=after)
        myLog.log(rate=0.4, loss=0.03, codewordNum=8, epochs=3)
        myLog.log('last', runtime=after, a=[-1.53, 0.24, 2.42], b=[-2.12, -0.97, 3.01])
        myLog.log(1, note='Did not perform quantization correctly')
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
                  "\tin TestLogger:", self.filename)
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
                      "\tin TestLogger:", self.filename)
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
                python_data_file = sio.loadmat(kwargs[key])

                self.s_fD[test_index] = python_data_file[TestLogger.field_names['s_fD']]
                self.s_fNt[test_index] = python_data_file[TestLogger.field_names['s_fNt']]
                self.s_fNu[test_index] = python_data_file[TestLogger.field_names['s_fNu']]
                self.s_fRatio[test_index] = python_data_file[TestLogger.field_names['s_fRatio']]
                self.s_fT[test_index] = python_data_file[TestLogger.field_names['s_fT']]
                self.s_fTestPower[test_index] = python_data_file[TestLogger.field_names['s_fTestPower']]
                self.v_fTrainPower[test_index] = python_data_file[TestLogger.field_names['v_fTrainPower']]

        self.save()
        print("Logged test number ", test_index+1,
              "\tin TestLogger:", self.filename)

    def delete(self, test=None, indexing=None):
        """Delete specific test or clear all test log

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
                    Delete all tests in the TestLogger except specified
                'from'
                    Delete all tests from specified test (including)
                    In this case, test MUST be an int.
                'to'
                    Delete all tests to specified test (including)
                    In this case, test MUST be an int.
                if empty:
                    Deleting test specified by the parameter test

        Example
        -------
        from TestLogger import *
        myLog = TestLogger('tanhLog.mat')
        for ii in range(1, 11):
            myLog.log(rate=0.1*ii, loss=0.01*ii)
        myLog.delete(5)
        myLog.delete(7, 'from')
        myLog.delete(3, 'to')
        myLog.delete([1, 3], 'except')
        myLog.delete()
        """

        # If specified number of test(s)
        if not(test is None):
            # If specified few tests
            if type(test) is list:
                test_index = []
                for ii in test:
                    # If test number is too large
                    if ii > len(self.rate):
                        self.exception('delete', 'IndexError', ii)
                        return
                    test_index.append(ii-1)
            # If specified only one test
            elif type(test) is int:
                # If test number is too large
                if test > len(self.rate):
                    self.exception('delete', 'IndexError', test)
                    return
                test_index = test - 1
            elif type(test) is str:
                if test is 'last':
                    test_index = len(self.rate)-1
                else:
                    self.exception('delete', 'TypeErrorBadStr', test)
                    return
            else:
                self.exception('delete', 'TypeErrorNotInt', test)
                return

            # If no special kind of indexing specified
            delete_index = test_index
            # If the indexing is 'except', delete_index are all the indices in
            # self.rate except the specified ones
            if indexing is 'except':
                delete_index = list(range(0, len(self.rate)))
                if type(test_index) is int:
                    delete_index.remove(test_index)
                else:
                    for ii in test_index:
                        delete_index.remove(ii)
            # If the indexing is 'from', delete_index are all the indices from
            # specified number to the end
            elif indexing is 'from':
                if type(test) is not int:
                    self.exception('delete', 'TypeErrorFrom', test)
                    return
                delete_index = range(test-1, len(self.rate))
            elif indexing is 'to':
                if type(test) is not int:
                    self.exception('delete', 'TypeErrorTo', test)
                    return
                delete_index = range(0, test)

            self.rate = np.delete(self.rate, delete_index, 0)
            self.loss = np.delete(self.loss, delete_index, 0)
            self.codewordNum = np.delete(self.codewordNum, delete_index, 0)
            self.learningRate = np.delete(self.learningRate, delete_index, 0)
            self.layersDim = np.delete(self.layersDim, delete_index, 0)
            self.runtime = np.delete(self.runtime, delete_index, 0)
            self.logtime = np.delete(self.logtime, delete_index, 0)
            self.algorithm = np.delete(self.algorithm, delete_index, 0)
            self.epochs = np.delete(self.epochs, delete_index, 0)
            self.note = np.delete(self.note, delete_index, 0)
            self.aCoefs = np.delete(self.aCoefs, delete_index, 0)
            self.bCoefs = np.delete(self.bCoefs, delete_index, 0)
            self.cCoefs = np.delete(self.cCoefs, delete_index, 0)
            self.magic_c = np.delete(self.magic_c, delete_index, 0)
            self.s_fD = np.delete(self.s_fD, delete_index, 0)
            self.s_fNt = np.delete(self.s_fNt, delete_index, 0)
            self.s_fNu = np.delete(self.s_fNu, delete_index, 0)
            self.s_fRatio = np.delete(self.s_fRatio, delete_index, 0)
            self.s_fT = np.delete(self.s_fT, delete_index, 0)
            self.s_fTestPower = np.delete(self.s_fTestPower, delete_index, 0)
            self.v_fTrainPower = np.delete(self.v_fTrainPower, delete_index, 0)

            print('Deleted test(s):', np.array(delete_index) + 1,
                  "\nFrom TestLogger:", self.filename)
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

            print('Cleared TestLogger:', self.filename)

        self.save()

    def content(self, test=None):
        """Show the content of the TestLogger

        Prints all information available for a certain test, multiple tests or
        all the tests in a TestLogger.

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
                    Printing short info about the TestLogger itself
                    (filename, number of logged tests and TestLogger type)

        Example
        -------
        from TestLogger import *
        from datetime import datetime
        before = datetime.now()
        after = datetime.now() - before
        myLog = TestLogger('tanhLog.mat')
        myLog.log(rate=0.3, loss=0.05, runtime=after)
        myLog.log(runtime=after, a=[-1.53, 0.24, 2.42], b=[-2.12, -0.97, 3.01])
        myLog.log('last', rate=0.2, loss=0.03, note='Content example')
        myLog.content()
        myLog.content(2)
        """

        do_not_exist_message = '________________'

        if test is None:
            print("The TestLogger '" + self.filename +
                  "' contains", len(self.rate), "tests.")
        else:
            if type(test) is int:
                if test > len(self.rate):
                    self.exception('content', 'IndexError', test)
                    return
                test_num = [test, ]
            elif type(test) is list:
                for ii in test:
                    if ii > len(self.rate):
                        self.exception('content', 'IndexError', ii)
                        return
                test_num = test
            elif type(test) is str:
                if test is 'all':
                    if len(self.rate):
                        test_num = range(1, len(self.rate)+1)
                    else:
                        print("The TestLogger", self.filename, "is empty.")
                        return
                elif test is 'last':
                    if len(self.rate):
                        test_num = [len(self.rate), ]
                    else:
                        print("The TestLogger", self.filename, "is empty.")
                        return
                else:
                    self.exception('content', 'TypeError', test)
                    return
            else:
                self.exception('content', 'TypeError', test)
                return

            print("\n\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/"
                  "\n\nContent of TestLogger '" + self.filename + "'\n\n",
                  "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n")
            for ii, currTest in enumerate(test_num):
                if not self.codewordNum[currTest-1]:
                    codeword_num_to_print = do_not_exist_message
                else:
                    codeword_num_to_print = self.codewordNum[currTest-1]

                if not self.algorithm[currTest-1]:
                    alg_to_print = do_not_exist_message
                else:
                    alg_to_print = remove_cell_format(self.algorithm[currTest - 1])

                if not self.learningRate[currTest-1]:
                    learning_rate_to_print = do_not_exist_message
                else:
                    learning_rate_to_print = self.learningRate[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (len(self.layersDim[currTest-1]) == 1 and
                        self.layersDim[currTest-1][0, 0] == 0)):
                    layers_dim_to_print = do_not_exist_message
                else:
                    layers_dim_to_print = self.layersDim[currTest-1]

                if not self.runtime[currTest-1]:
                    runtime_to_print = do_not_exist_message
                else:
                    runtime_to_print = remove_cell_format(self.runtime[currTest - 1])

                if not self.epochs[currTest-1]:
                    epoch_to_print = do_not_exist_message
                else:
                    epoch_to_print = self.epochs[currTest-1]

                if not self.note[currTest-1]:
                    note_to_print = do_not_exist_message
                else:
                    note_to_print = remove_cell_format(self.note[currTest - 1])

                if (type(self.layersDim[currTest-1]) is float or
                    (self.aCoefs[currTest-1].size == 1 and
                        self.aCoefs[currTest-1][0, 0] == 0)):
                    a_coefs_to_print = do_not_exist_message
                else:
                    a_coefs_to_print = self.aCoefs[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (self.bCoefs[currTest-1].size == 1 and
                        self.bCoefs[currTest-1][0, 0] == 0)):
                    b_coefs_to_print = do_not_exist_message
                else:
                    b_coefs_to_print = self.bCoefs[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (self.cCoefs[currTest-1].size == 1 and
                        self.cCoefs[currTest-1][0, 0] == 0)):
                    c_coefs_to_print = do_not_exist_message
                else:
                    c_coefs_to_print = self.cCoefs[currTest-1]

                if not self.magic_c[currTest-1]:
                    magic_c_to_print = do_not_exist_message
                else:
                    magic_c_to_print = self.magic_c[currTest-1]

                if not self.s_fD[currTest-1]:
                    s_f_d_to_print = do_not_exist_message
                else:
                    s_f_d_to_print = self.s_fD[currTest-1]

                if not self.s_fNt[currTest-1]:
                    s_f_nt_to_print = do_not_exist_message
                else:
                    s_f_nt_to_print = self.s_fNt[currTest-1]

                if not self.s_fNu[currTest-1]:
                    s_f_nu_to_print = do_not_exist_message
                else:
                    s_f_nu_to_print = self.s_fNu[currTest-1]

                if not self.s_fRatio[currTest-1]:
                    s_f_ratio_to_print = do_not_exist_message
                else:
                    s_f_ratio_to_print = self.s_fRatio[currTest-1]

                if not self.s_fT[currTest-1]:
                    s_f_t_to_print = do_not_exist_message
                else:
                    s_f_t_to_print = self.s_fT[currTest-1]

                if not self.s_fTestPower[currTest-1]:
                    s_f_test_power_to_print = do_not_exist_message
                else:
                    s_f_test_power_to_print = self.s_fTestPower[currTest-1]

                if (type(self.layersDim[currTest-1]) is float or
                    (self.v_fTrainPower[currTest-1].size == 1 and
                        self.v_fTrainPower[currTest-1][0, 0] == 0)):
                    v_f_train_power_to_print = do_not_exist_message
                else:
                    v_f_train_power_to_print = self.v_fTrainPower[currTest-1]

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
                              self.loss[currTest-1], codeword_num_to_print,
                              alg_to_print, learning_rate_to_print,
                              layers_dim_to_print, a_coefs_to_print,
                              b_coefs_to_print, c_coefs_to_print, magic_c_to_print,
                              runtime_to_print, epoch_to_print,
                              remove_cell_format(self.logtime[currTest - 1]),
                              note_to_print, s_f_t_to_print, s_f_d_to_print,
                              s_f_nt_to_print, s_f_nu_to_print, s_f_ratio_to_print,
                              s_f_test_power_to_print, v_f_train_power_to_print))

                if ii < len(test_num)-1:
                    print("---------------------------------------------------")

            print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")

    def plot(self, theory_data_file=None):
        """Plot all results in a TestLogger

        IMPORTANT!!!
            The function opens a figure which will pause all your code until
            closed!

        Opens a new figure, plots all the theoretical bounds and all
        previously logged tests.
        The points are enumerated according to their logging order.
        Click any data point to open a tooltip with additional information.
        To see all information available for a test, use the function
        content()

        Parameters
        ----------
            theory_data_file : str
                Name of MATLAB .mat file containing the variables 'v_fRate' and
                'm_fCurves' which provide theoretical bounds for the results.
                This is a file created using the generateData.m function.
                Specify including .mat extension, for example: 'data.mat'

        Example
        -------
        The file 'data.mat' is generated using generateData.m

        from TestLogger import *
        myLog = TestLogger('tanhLog.mat')
        myLog.log(rate=0.3, loss=0.05)
        myLog.plot('data.mat')
        """

        chosen_marker_size = 2
        reg_marker_size = 7
        index_alpha = 0.5
        data_tip_alpha = 1
        data_tip_font_size = 6
        legend_font_size = 7
        textbox_alpha = 0.8
        text_offset = 0.0005
        tick_offset = 0.001
        size_of_figure = (8, 5)  # in inches

        # Define which lines to plot
        which_to_plot = [
                         1,  # No quantization
                         1,  # Asymptotic optimal task-based
                         1,  # Asymptotic optimal task-ignorant
                         1   # Hardware limited upper bound
                         ]

        # Set the legend labels
        labels = [
                  'No quantization',
                  'Asymptotic optimal task-based',
                  'Asymptotic optimal task-ignorant',
                  'Hardware limited upper bound'
                  ]

        markers = ['', '', '', '']
        line_styles = [':', '--', '--', '--']
        line_colors = ['black', 'red', 'blue', 'lime']
        line_widths = [1, 1, 1, 1.5]
        marker_sizes = [4, 1, 1, 1]
        point_marker = 'x'
        points_color = 'orange'
        chosen_marker = 'o'
        chosen_color = 'red'
        tooltip_box_style = 'round'
        tooltip_box_color = 'wheat'
        fill_color = 'c'

        enumerate_tests = True

        def pick_handler(event):
            """Handles the choosing of plotted results

            Note
            ----
            This function is not intended for the user and is used only by other
            function in the class TestLogger.
            """
            # Get the pressed artist
            artist = event.artist

            # If the clicked point was already chosen, clear all tooltips and return
            if artist.get_marker() is chosen_marker:
                clear_data_tips(-1)
                return

            # Mark the chosen point
            artist.set(marker=chosen_marker, markersize=chosen_marker_size,
                       color=chosen_color)

            # Get the index of the clicked point in the test log
            chosen_index = res_list.index(artist)
            # Hide current result index
            index_text[chosen_index].set(alpha=0)
            # Show chosen text box
            text_boxes[chosen_index] = dict(boxstyle=tooltip_box_style,
                                            facecolor=tooltip_box_color,
                                            alpha=textbox_alpha)
            # Show chosen tooltip
            tooltips[chosen_index].set(alpha=data_tip_alpha,
                                       bbox=text_boxes[chosen_index])

            # Clear other tooltips
            clear_data_tips(chosen_index)

            # Update figure
            fig.canvas.draw()
            fig.canvas.flush_events()

        def clear_data_tips(except_index):
            """Clears tooltips

            Parameters
            ----------
            except_index : int
                Do not clear the tooltip of the point at the specified index
                Pass -1 to clear all tooltips

            Note
            ----
            This function is not intended for the user and is used only by other
            function in the class TestLogger.
            """

            for i in range(0, len(res_list)):
                if i is except_index:
                    continue
                # Unmark all other points
                res_list[i].set(marker=point_marker, markersize=reg_marker_size,
                                color=points_color)
                # Show all result indices
                index_text[i].set(alpha=index_alpha)
                # Hide all text_boxes
                text_boxes[i] = dict(boxstyle=tooltip_box_style,
                                     facecolor=tooltip_box_color,
                                     alpha=0)
                # Hide all result tooltips
                tooltips[i].set(alpha=0, bbox=text_boxes[i])

            # Update figure
            fig.canvas.draw()
            fig.canvas.flush_events()

        def get_text_alignment(x, y, text_shift):
            """Set tooltip textbox alignment and its offset from the
            corresponding point on the graph

            Note
            ----
            This function is not intended for the user and is used only by other
            function in the class TestLogger.

            Parameters
            ----------
            x : numpy.ndarray
                X position of a test log (rate)
            y : numpy.ndarray
                Y position of a test log (average loss)
            text_shift : float
                The absolute value of the offset of the tooltip from the point

            Returns
            -------
            (ha, va, x_offset, y_offset) : tuple
                ha : str
                    Horizontal alignment as accepted by the text property:
                    horizontalalignment of the matplotlib.
                    Possible values: [ 'right' | 'left' ]
                ha : str
                    Vertical alignment as accepted by the text property:
                    verticalalignment of the matplotlib.
                    Possible values: [ 'top' | 'bottom' ]
                x_offset : float
                    The X axis offset of the textbox
                y_offset : float
                    The Y axis offset of the textbox
            """
            axs = plt.gca()
            x_lim = axs.get_xlim()
            y_lim = axs.get_ylim()

            if x < x_lim[1] / 2:
                ha = 'left'
                x_offset = text_shift
            else:
                ha = 'right'
                x_offset = -text_shift

            if y < y_lim[1]/2:
                va = 'bottom'
                y_offset = text_shift
            else:
                va = 'top'
                y_offset = -text_shift

            return ha, va, x_offset, y_offset

        plt.close('all')

        # Create figure and get axes handle
        fig = plt.figure(figsize=size_of_figure)
        ax = fig.add_subplot(111)

        # Connect figure to callback
        fig.canvas.mpl_connect('pick_event', pick_handler)
        fig.canvas.mpl_connect('figure_leave_event', lambda event,
                               temp=-1: clear_data_tips(temp))

        if theory_data_file:
            theory = sio.loadmat(theory_data_file)

            # Extracting theory data vectors
            theory_rate = theory['v_fRate']
            theory_loss = theory['m_fCurves']



            # Plot all theoretical bounds
            for ii in range(0, theory_loss.shape[0]):
                if which_to_plot[ii]:
                    ax.plot(theory_rate[0], theory_loss[ii, :],
                            label=labels[ii],
                            marker=markers[ii], color=line_colors[ii],
                            linestyle=line_styles[ii], linewidth=line_widths[ii],
                            markersize=marker_sizes[ii])

            if theory_loss.shape[0] > 2:
                # Create fill vectors
                x_fill = np.concatenate((theory_rate[0],
                                         np.flip(theory_rate[0], 0)), axis=0)
                y_fill = np.concatenate((theory_loss[3, :],
                                         np.flip(theory_loss[1, :], 0)), axis=0)

                # Plot fill
                ax.fill(x_fill, y_fill, c=fill_color, alpha=0.3)

        # Plot previous results (the loop is for plotting as separate artists)
        res_list = []
        for ii in range(0, len(self.rate)-1):
            if self.rate[ii]:
                res_list += ax.plot(self.rate[ii], self.loss[ii], marker='x',
                                    markersize=reg_marker_size,
                                    color=points_color, picker=5)

        # Plot result (not plotting in the for loop above only to set the label
        # to 'Results' without affecting all result points)
        try:
            if self.rate[-1]:
                res_list += ax.plot(self.rate[-1], self.loss[-1], marker='x',
                                    markersize=reg_marker_size, color=points_color,
                                    label='Results', picker=5)
        except IndexError:
            self.exception('plot', 'EmptyLog')

        index_text = []
        tooltips = []
        text_boxes = []
        for ii in range(0, len(self.rate)):
            # If the current iterated test is not empty
            if self.rate[ii]:
                # Enumerate result points in the figure
                if enumerate_tests:
                    index_text.append(ax.text(self.rate[ii] + tick_offset,
                                              self.loss[ii] + tick_offset,
                                              str(ii+1), fontsize=8, alpha=index_alpha,
                                              verticalalignment='center',
                                              horizontalalignment='center'))
                # Create all text boxes and do not display them
                text_boxes.append(dict(boxstyle='round', facecolor='wheat',
                                       alpha=0))

                # Create all data tooltip and do not display them
                curr_iter_date_time = datetime.strptime(
                    remove_cell_format(self.logtime[ii]), '%Y-%m-%d %H:%M:%S.%f')
                if self.runtime[ii]:
                    curr_runtime = datetime.strptime(remove_cell_format(self.runtime[ii]),
                                                     '%H:%M:%S.%f')
                else:
                    curr_runtime = ''
                if self.algorithm[ii] and self.runtime[ii]:
                    text_to_display = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nAlgorithm:\n' + \
                                      remove_cell_format(self.algorithm[ii]) + \
                        '\nRuntime: ' + curr_runtime.strftime('%H:%M:%S.%f') + \
                        '\nDate: ' + curr_iter_date_time.strftime('%d/%m/%y') + \
                        '\nTime: ' + curr_iter_date_time.strftime('%H:%M:%S')
                elif self.algorithm[ii] and not(self.runtime[ii]):
                    text_to_display = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nAlgorithm:\n' + \
                                      remove_cell_format(self.algorithm[ii]) + \
                        '\nDate: ' + curr_iter_date_time.strftime('%d/%m/%y') + \
                        '\nTime: ' + curr_iter_date_time.strftime('%H:%M:%S')
                elif not(self.algorithm[ii]) and self.runtime[ii]:
                    text_to_display = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nRuntime: ' + curr_runtime.strftime('%H:%M:%S.%f') + \
                        '\nDate: ' + curr_iter_date_time.strftime('%d/%m/%y') + \
                        '\nTime: ' + curr_iter_date_time.strftime('%H:%M:%S')
                else:
                    text_to_display = 'Rate: ' + str(self.rate[ii]) + \
                        '\nAvg. Distortion: ' + str(self.loss[ii]) + \
                        '\nDate: ' + curr_iter_date_time.strftime('%d/%m/%y') + \
                        '\nTime: ' + curr_iter_date_time.strftime('%H:%M:%S')
                text_align = get_text_alignment(self.rate[ii],
                                                self.loss[ii],
                                                text_offset)
                tooltips.append(ax.text(self.rate[ii]+text_align[2],
                                        self.loss[ii]+text_align[3],
                                        text_to_display,
                                        alpha=0, fontsize=data_tip_font_size,
                                        bbox=text_boxes[ii],
                                        ha=text_align[0], va=text_align[1]))

        # Labeling and graph appearance
        plt.xlabel('Rate', fontsize=18, fontname='Times New Roman')
        plt.ylabel('Average Distortion', fontsize=18,
                   fontname='Times New Roman')
        ax.legend(fontsize=legend_font_size)
        ax.autoscale(enable=True, axis='x', tight=True)
        # Show figure
        plt.show()

    def save(self):
        """Save a TestLogger

        Note
        ----
        This function is not intended for the user and is used only by other
        function in the class TestLogger.
        """

        save_vars = {TestLogger.field_names['rate_results']: self.rate,
                     TestLogger.field_names['loss_results']: self.loss,
                     TestLogger.field_names['codeword_num']: self.codewordNum,
                     TestLogger.field_names['learning_rate']: self.learningRate,
                     TestLogger.field_names['layers_dim']: self.layersDim,
                     TestLogger.field_names['run_time']: self.runtime,
                     TestLogger.field_names['time']: self.logtime,
                     TestLogger.field_names['algorithm_name']: self.algorithm,
                     TestLogger.field_names['train_epochs']: self.epochs,
                     TestLogger.field_names['notes']: self.note,
                     TestLogger.field_names['a_coefs']: self.aCoefs,
                     TestLogger.field_names['b_coefs']: self.bCoefs,
                     TestLogger.field_names['c_coefs']: self.cCoefs,
                     TestLogger.field_names['c_bounds']: self.magic_c,
                     TestLogger.field_names['s_fD']: self.s_fD,
                     TestLogger.field_names['s_fNt']: self.s_fNt,
                     TestLogger.field_names['s_fNu']: self.s_fNu,
                     TestLogger.field_names['s_fRatio']: self.s_fRatio,
                     TestLogger.field_names['s_fT']: self.s_fT,
                     TestLogger.field_names['s_fTestPower']: self.s_fTestPower,
                     TestLogger.field_names['v_fTrainPower']: self.v_fTrainPower}

        sio.savemat(self.filename, save_vars)

    def exception(self, property_name, except_type, *test):
        """Print an exception

        Because the TestLogger class is used in long simulations, you do not want
        any little mistake to throw an error and stop your code.
        Exceptions in this function are more informative than python exceptions
        and were easier to use in this particular code.

        Note
        ----
        This function is not intended for the user and is used only by other
        function in the class TestLogger.
        """

        if test:
            test = test[0]

        if property_name is 'log':
            if except_type is 'IndexError':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tIndexError in the property log()",
                      "\n\tCause:\tTest number", test,
                      "does not exist,",  "there are only",
                      len(self.rate), "tests.")
            if except_type is 'TypeError':
                if type(test) is str:
                    print("Error in TestLogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property log()",
                          "\n\tCause:\tTest argument can only be an integer",
                          "or the string 'last', but was the string",
                          "'" + test + "'")
                else:
                    print("Error in TestLogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property log()",
                          "\n\tCause:\tTest argument can only be an integer",
                          "or the string 'last', but was of type", type(test))
            if except_type is 'ValueError':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tValueError in the property log()",
                      "\n\tCause:\tThere are no input arguments to log()")

        if property_name is 'delete':
            if except_type is 'IndexError':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tIndexError in the property delete()",
                      "\n\tCause:\tTest number", test, "does not exist,",
                      "there are only", len(self.rate), "tests.")
            if except_type is 'TypeErrorBadStr':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTest argument can only be an integer,",
                      "a list or the string 'last', but was the string '" +
                      test + "'")
            if except_type is 'TypeErrorNotInt':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTest argument can only be an integer",
                      "or a list, but was of type", type(test))
            if except_type is 'TypeErrorFrom':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTo delete tests 'from' certain test,",
                      "it needs to be an integer.",
                      "Instead was of type", type(test))
            if except_type is 'TypeErrorTo':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tTypeError in the property delete()",
                      "\n\tCause:\tTo delete tests 'to' certain test,",
                      "it needs to be an integer.",
                      "Instead was of type", type(test))

        if property_name is 'content':
            if except_type is 'TypeError':
                if type(test) is str:
                    print("Error in TestLogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property content()",
                          "\n\tCause:\tTest argument can only be an integer,",
                          "a list or the strings 'all' or 'last', but was the",
                          "string '" + test + "'")
                else:
                    print("Error in TestLogger", self.filename + ":",
                          "\n\tType:\tTypeError in the property content()",
                          "\n\tCause:\tTest argument can only be an integer,",
                          "a list or the string 'all', but was of type",
                          type(test))
            if except_type is 'IndexError':
                print("Error in TestLogger", self.filename + ":",
                      "\n\tType:\tIndexError in the property content()",
                      "\n\tCause:\tTest number", test,
                      "does not exist,",  "there are only",
                      len(self.rate), "tests.")

        if property_name is 'plot':
            if except_type is 'EmptyLog':
                print("Warning in TestLogger", self.filename + ":",
                      "\n\tType:\tEmptyLog in the property plot()",
                      "\n\tCause:\tTestLogger:", self.filename,
                      "is empty. Nothing to plot...")


def create_mat_file(name):
    """Create a new .mat file that can be handled by the TestLogger class

    Creates a MATLAB .mat file with all the variables required so it can be
    properly "connected" to the python class TestLogger.
    Special variables are added, if needed, according to the type parameter.

    Parameters
    ----------
        name : str
            Name of the mat file, with extension! For example: "testLog.mat"

    Returns
    -------
        TestLogger.TestLogger
            The created file is connected to the TestLogger class using
            TestLogger() at the end of the function createMatFile().
            One can alternatively call createMatFile() without using its output
            and then call TestLogger() later

    Example
    -------
    from TestLogger import *
    myLog = create_mat_file("testLog.mat")
    myLog
    """

    rate = np.empty((0, 1), float)  # MATLAB Array of doubles
    loss = np.empty((0, 1), float)  # MATLAB Array of doubles
    codeword_num = np.empty((0, 1), float)  # MATLAB Array of doubles
    learning_rate = np.empty((0, 1), float)  # MATLAB Array of doubles
    layers_dim = np.empty((0, 1), object)  # MATLAB Cell
    runtime = np.empty((0, 1), object)  # MATLAB Cell
    logtime = np.empty((0, 1), object)  # MATLAB Cell
    algorithm = np.empty((0, 1), object)  # MATLAB Cell
    epochs = np.empty((0, 1), float)  # MATLAB Array of doubles
    note = np.empty((0, 1), object)  # MATLAB Cell
    a_coefs = np.empty((0, 1), object)  # MATLAB Cell
    b_coefs = np.empty((0, 1), object)  # MATLAB Cell
    c_coefs = np.empty((0, 1), object)  # MATLAB Cell
    magic_c = np.empty((0, 1), float)  # MATLAB Array
    s_fd = np.empty((0, 1), float)  # MATLAB Array
    s_fnt = np.empty((0, 1), float)  # MATLAB Array
    s_fnu = np.empty((0, 1), float)  # MATLAB Array
    s_f_ratio = np.empty((0, 1), float)  # MATLAB Array
    s_ft = np.empty((0, 1), float)  # MATLAB Array
    s_f_test_power = np.empty((0, 1), float)  # MATLAB Array
    v_f_train_power = np.empty((0, 1), object)  # MATLAB Cell

    variables = {TestLogger.field_names['rate_results']: rate,
                 TestLogger.field_names['loss_results']: loss,
                 TestLogger.field_names['codeword_num']: codeword_num,
                 TestLogger.field_names['learning_rate']: learning_rate,
                 TestLogger.field_names['layers_dim']: layers_dim,
                 TestLogger.field_names['run_time']: runtime,
                 TestLogger.field_names['time']: logtime,
                 TestLogger.field_names['algorithm_name']: algorithm,
                 TestLogger.field_names['train_epochs']: epochs,
                 TestLogger.field_names['notes']: note,
                 TestLogger.field_names['a_coefs']: a_coefs,
                 TestLogger.field_names['b_coefs']: b_coefs,
                 TestLogger.field_names['c_coefs']: c_coefs,
                 TestLogger.field_names['c_bounds']: magic_c,
                 TestLogger.field_names['s_fD']: s_fd,
                 TestLogger.field_names['s_fNt']: s_fnt,
                 TestLogger.field_names['s_fNu']: s_fnu,
                 TestLogger.field_names['s_fRatio']: s_f_ratio,
                 TestLogger.field_names['s_fT']: s_ft,
                 TestLogger.field_names['s_fTestPower']: s_f_test_power,
                 TestLogger.field_names['v_fTrainPower']: v_f_train_power}

    if os.path.exists(name):
        print("Test logger '" + name + "' already exists...")
    else:
        sio.savemat(name, variables)
        print("Created new test log '" + name + "'")
    return TestLogger(name)


def remove_cell_format(formatted):
    """Clear a string formatting caused by saving as cell in mat file

    Note
    ----
    This function is not intended for the user and is used only by other
    function in the class TestLogger.
    """
    unformatted = str(formatted).replace("[", "")
    unformatted = unformatted.replace("]", "")
    unformatted = unformatted.replace("'", "")
    return unformatted
