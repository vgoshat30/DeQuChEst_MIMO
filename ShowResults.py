import scipy.io as sio
import scipy.optimize as optim
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from random import random
from datetime import datetime
from torch.nn.parameter import Parameter
import torch
import sys
import time

from testLogger import *
import UserInterface as UI
from ProjectConstants import *


def plotTanhFunction(testLog, testNumber):
    def quantize(input, a, b, q):
        """Get the result of the hard quantization function derived from the
        soft one (the meaning of the name 'SoftToHardQuantization') and the
        codeword the input was sorted to

        Parameters
        ----------
            input : float
                The quantizer input
            a : numpy.ndarray
                ai coefficients
            b : numpy.ndarray
                bi coefficients

        Returns
        -------
            quantizedValue : int
                The output of the quantizer
            codeword : int
                The index of the coresponding codeword
        """
        output = np.empty([input.size, 2])
        for inputIndex in range(input.size):
            if input[inputIndex] <= b[0]:
                output[inputIndex, 0] = - sum(a)
                output[inputIndex, 1] = 0
            if input[inputIndex] > b[-1]:
                output[inputIndex, 0] = sum(a)
                output[inputIndex, 1] = len(b)
            for ii in range(len(b)-1):
                if b[ii] < input[inputIndex] and input[inputIndex] <= b[ii + 1]:
                    output[inputIndex, 0] = q((b[ii] + b[ii+1])/2)
                    output[inputIndex, 1] = ii + 1
        return output

    a = testLog.aCoefs[testNumber-1][0]
    b = testLog.bCoefs[testNumber-1][0]
    magic_c = testLog.magic_c[testNumber-1]

    # Create symbolic variable x
    symX = sym.symbols('x')

    sym_tanh = a[0] * sym.tanh(magic_c*(symX + b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(magic_c*(symX + b[ii]))
    # Convert the symbolic functions to numpy friendly (for substitution)
    q = sym.lambdify(symX, sym_tanh, "numpy")

    space = 5

    x = np.linspace(b[0]-space, b[-1]+space, 10000)

    quantized = quantize(x, a, b, q)

    plt.plot(x, q(x))
    plt.plot(x, quantized[:, 0])
    plt.show()


# Set the test log
log = testlogger(TEST_LOG_MAT_FILE)

# # Delete tests
# log.delete(100000)

# Show content of tests
# log.content('all')

# Plot test log
log.plot()

# Plot soft and hard quantization functions of specific test
# plotTanhFunction(log, 8)
