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


def plotTanhFunction(a, b, c):
    def quantize(input, a, b, c, q):
        """Get the result of the hard quantization function derived from the soft
        one (the meaning of the name 'SoftToHardQuantization') and the codeword the
        input was sorted to

        Parameters
        ----------
            input : float
                The quantizer input
            a : numpy.ndarray
                ai coefficients
            b : numpy.ndarray
                bi coefficients
            c : numpy.ndarray
                ci coefficients

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
                output[inputIndex, 0] = q(-1000000)  # - sum(a)
                output[inputIndex, 1] = 0
            if input[inputIndex] > b[-1]:
                output[inputIndex, 0] = q(1000000)  # sum(a)
                output[inputIndex, 1] = len(b)
            for ii in range(len(b)-1):
                if b[ii] < input[inputIndex] and input[inputIndex] <= b[ii + 1]:
                    output[inputIndex, 0] = q((b[ii] + b[ii+1])/2)
                    output[inputIndex, 1] = ii + 1
        return output

    # Create symbolic variable x
    symX = sym.symbols('x')

    sym_tanh = a[0] * sym.tanh(MAGIC_C*(symX + b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(MAGIC_C*(symX + b[ii]))
    # Convert the symbolic functions to numpy friendly (for substitution)
    q = sym.lambdify(symX, sym_tanh, "numpy")

    space = 5

    x = np.linspace(b[0]-space, b[-1]+space, 10000)

    quantized = quantize(x, a, b, 0, q)

    plt.plot(x, q(x))
    plt.plot(x, quantized[:, 0])
    plt.show()


# theoryMatFile = sio.loadmat('theoreticalBounds.mat')
# theoryLoss = theoryMatFile['m_fCurves']
# theoryRate = theoryMatFile['v_fRate']
#
# log = createMatFile('testLog.mat', 'tanh', theoryRate, theoryLoss)


log = testlogger('testLog.mat')
# log.delete([19, 17, 13, 14])
# log.content(10)
log.plot()


a = np.array([0.40584144, 0.30556887, 0.27043727, 0.2697681,  0.30160347, 0.4007607])
b = np.array([-5.061066,   -2.7510405,  -0.88095266,  0.8922799,   2.7661927,   5.051867])


c = np.array([4.5177526, -20.526585,   12.683011,    9.798179,   14.7053995,  -8.947173,
              3.0234663])

# plotTanhFunction(a, b, 0)
