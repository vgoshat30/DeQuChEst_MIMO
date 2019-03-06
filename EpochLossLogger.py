import scipy.io as sio
import os.path
import numpy as np
import matplotlib.pyplot as plt

# TODO: Make an option to create multiple graphs (one for each test)


class EpochLossLogger:

    # Keys are names used in this code and values are names of the variables
    # in the MATLAB code. DO NOT CHANGE PYTHON VARIABLE NAMES!
    #                   PYTHON      :     MATLAB
    field_names = {'epoch': 'epoch',
                   'loss': 'loss'}

    def __init__(self, name):
        """Connect between a .mat file and a python variable

        When called, the __init__ function reads the specified .mat file and
        creates all the fields relevant for manipulating this .mat file.
        According to the loggerType (specified in a variable in the .mat
        file, which is set using the create_mat_file() function) additional
        fields are created (for example, aCoefs, bCoefs, cCoefs for a TestLogger
        of type 'tanh')

        Parameters
        ----------
        name : str
            Name of the .mat file to be manipulated

        Returns
        -------
        class 'EpochLossLogger.EpochLossLogger'
            A variable containing all the fields created.

        Example
        -------
        from EpochLossLogger import *
        myLog = EpochLossLogger("epochLog.mat")
        myLog
        """
        # mat. file name
        self.filename = name

        try:
            # Load data from file
            python_mat_file = sio.loadmat(self.filename)
        except IOError:
            raise IOError("File '" + self.filename + "' does not exist.")

        # Number of training epochs
        self.epoch = python_mat_file[EpochLossLogger.field_names['epoch']]
        # Resulted loss fo the relevant number of training epochs
        self.loss = python_mat_file[EpochLossLogger.field_names['loss']]

        # If the test log is not empty, reducing the dimensions of all
        # parameters to one
        if self.epoch.shape[0]:
            self.epoch = self.epoch[0]
            self.loss = self.loss[0]

    def add_empty_test(self):
        """Append empty elements to all log fields

        The function appends an empty element to each field of the EpochLossLogger.
        If there is already empty test at the end of the EpochLossLogger, the
        function will not add additional one.

        Example
        -------
        (Additional function __init__() and content() were used in the example)

        from EpochLossLogger import *
        myLog = EpochLossLogger('epochLog.mat')
        myLog.add_empty_test()
        """
        # Append only if there is no empty test at the end of the log already
        if self.epoch.all():
            self.epoch = np.append(self.epoch, 0)
            self.loss = np.append(self.loss, 0)

    def log(self, index=None, **kwargs):
        """Manipulate content of a log element

        Use this function to add a new test or edit an existing one.

        Parameters
        ----------
            index : int OR str , optional
                if int:
                    Number of test to edit
                if str:
                    'last' - to edit last test in the TestLogger (empty or not)

            **kwargs (for all TestLogger types)
                'epoch' : float
                    Current epoch
                'loss' : float
                    Resulting loss of the epoch

        Example
        -------
        from EpochLossLogger import *
        myLog = EpochLossLogger('epochLog.mat')
        """

        # If no parameters to log were specified
        if not kwargs:
            self.exception('log', 'ValueError')
            return

        # If test number not specified, creating new empty test
        if not index:
            # Add empty slot if needed
            self.add_empty_test()
            # Setting test index to the last index in the test log
            test_index = len(self.epoch) - 1
            print("Created log number", test_index+1,
                  "\tin EpochLossLogger:", self.filename)
        # If requested to log to the last test
        elif index is 'last':
            test_index = len(self.epoch) - 1
        # If specified test number
        elif type(index) is int:
            # Setting test index to the index specified by 'test'
            test_index = index - 1

            # If test number is larger than existing number of tests by exactly
            # one, creating new empty test and logging to it
            if test_index is len(self.epoch):
                self.add_empty_test()
                print("Created log number", test_index+1,
                      "\tin EpochLossLogger:", self.filename)
            # If test number is larger than existing number of tests by more
            # than one, returning.
            elif test_index > len(self.epoch):
                self.exception('log', 'IndexError', index)
                return
        # If test is not a known codeword or not an integer, returning
        else:
            self.exception('log', 'TypeError', index)
            return

        # Check what optional arguments were passed
        for key in kwargs:
            # Check if epoch provided
            if key is 'epoch':
                self.epoch[test_index] = kwargs[key]
            if key is 'loss':
                self.loss[test_index] = kwargs[key]

        self.save()
        print("Logged log number ", test_index+1,
              "\tin EpochLossLogger:", self.filename)

    def plot(self):
        """Plot all results in a EpochLossLogger

        IMPORTANT!!!
            The function opens a figure which will pause all your code until
            closed!

        Opens a new figure and plots loss versus epoch.

        Example
        -------
        The file 'data.mat' is generated using generateData.m

        from EpochLossLogger import *
        myLog = EpochLossLogger('epochLog.mat')
        myLog.log(epoch=10, loss=0.05)
        myLog.plot()
        """

        marker = ''
        line_style = '-'
        line_color = 'red'
        line_width = 1
        marker_size = 4
        size_of_figure = (8, 5)  # in inches

        plt.close('all')

        # Create figure and get axes handle
        fig = plt.figure(figsize=size_of_figure)
        ax = fig.add_subplot(111)

        ax.plot(self.epoch, self.loss, marker=marker,
                markersize=marker_size, color=line_color,
                linestyle=line_style, linewidth=line_width)

        # Labeling and graph appearance
        plt.xlabel('Epoch', fontsize=18, fontname='Times New Roman')
        plt.ylabel('Average Distortion', fontsize=18,
                   fontname='Times New Roman')
        ax.autoscale(enable=True, axis='x', tight=True)
        # Show figure
        plt.show()

    def save(self):
        """Save a EpochLossLogger

        Note
        ----
        This function is not intended for the user and is used only by other
        function in the class TestLogger.
        """

        save_vars = {EpochLossLogger.field_names['epoch']: self.epoch,
                     EpochLossLogger.field_names['loss']: self.loss}

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
                      "\n\tCause:\tLog number", test,
                      "does not exist,",  "there are only",
                      len(self.epoch), "logs.")
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


        if property_name is 'plot':
            if except_type is 'EmptyLog':
                print("Warning in TestLogger", self.filename + ":",
                      "\n\tType:\tEmptyLog in the property plot()",
                      "\n\tCause:\tTestLogger:", self.filename,
                      "is empty. Nothing to plot...")


def create_epoch_loss_log(name):
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
            EpochLossLogger.EpochLossLogger
                The created file is connected to the EpochLossLogger class using
                EpochLossLogger() at the end of the function createMatFile().
                One can alternatively call createMatFile() without using its output
                and then call EpochLossLogger() later

        Example
        -------
        from EpochLossLogger import *
        myLog = create_epoch_loss_log("epochLog.mat")
        myLog
        """

        epoch = np.empty((0, 1), float)  # MATLAB Array of doubles
        loss = np.empty((0, 1), float)  # MATLAB Array of doubles

        variables = {EpochLossLogger.field_names['epoch']: epoch,
                     EpochLossLogger.field_names['loss']: loss}

        if os.path.exists(name):
            print("Test logger '" + name + "' already exists...")
        else:
            sio.savemat(name, variables)
            print("Created new test log '" + name + "'")
        return EpochLossLogger(name)