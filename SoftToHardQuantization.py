import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy
import sympy as sym
from ProjectConstants import *


class network(nn.Module):

    def __init__(self, codebookSize):
        super(network, self).__init__()
        self.l1 = nn.Linear(INPUT_DIMENSION, 520)
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(520, OUTPUT_DIMENSION)
        self.l3 = nn.Linear(OUTPUT_DIMENSION, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, OUTPUT_DIMENSION)
        self.q1 = quantizationLayer(
            OUTPUT_DIMENSION, OUTPUT_DIMENSION, codebookSize)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class quantizationLayer(Module):
    def __init__(self, in_features, out_features, M):
        super(quantizationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebookSize = M
        # There are three group of parameters: {ai, bi, ci}, as described in
        # the paper
        self.weight = Parameter(torch.Tensor(self.codebookSize - 1, 2))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.numel())
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ret = torch.zeros(input.size())
        # The first two for loops run on all the input values
        # for ii in range(0, input.size(0)):
        #     for jj in range(0, input.size(1)):
        #         for kk in range(0, self.codebookSize - 1):
        #             ret[ii, jj] += self.weight[kk, 0] * \
        #                 torch.tanh(self.weight[kk, 2] *
        #                            (input[ii, jj] - self.weight[kk, 1]))

        for kk in range(0, self.codebookSize - 1):
            # addVal = torch.mul(input, self.weight[kk, 2])
            addVal = torch.add(input, self.weight[kk, 1])
            addVal = torch.mul(addVal, MAGIC_C)
            addVal = torch.tanh(addVal)
            addVal = torch.mul(addVal, self.weight[kk, 0])
            ret = torch.add(ret, addVal)  # out=None ?
        return ret


def getParameters(model):
    """Extract the {ai, bi, ci} coefficients of the soft quantization function
    from the network model, return them sorted by ascending order of the
    bi coefficients, and create a symbolic function which will be used in the
    hard quantization process (the 'quantize' funtion below)

    Parameters
    ----------
        model : network (from this module)
            The network instance of the 'Soft to Hard Quantization' model

    Returns
    -------
        a : list
            The ai coefficients of the sum of tanh soft qunatization function
        b : list
            The bi coefficients
        c : list
            The ci coefficients

        q : sympy function
            The soft qunatization symbolic function (sum of tanh)
    """

    parameters = model.q1.weight.data.numpy()

    # Coefficients of the tanh
    a = parameters[:, 0]
    b = parameters[:, 1]
    # c = parameters[:, 2]

    # Sort the coefficients by ascending order of the bi-s
    sortedIndecies = b.argsort()
    a = a[sortedIndecies]
    b = b[sortedIndecies]
    c = None  # c[sortedIndecies]

    # Create symbolic variable x
    symX = sym.symbols('x')

    sym_tanh = a[0] * sym.tanh(MAGIC_C*(symX + b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(MAGIC_C*(symX + b[ii]))
    # Convert the symbolic functions to numpy friendly (for substitution)
    q = sym.lambdify(symX, sym_tanh, "numpy")

    return a, b, c, q


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

    if input <= b[0]:
        return -sum(a), 0
    if input > b[-1]:
        return sum(a), len(b)
    for ii in range(0, len(b)):
        if b[ii] < input and input <= b[ii + 1]:
            return q((b[ii] + b[ii+1])/2), ii + 1
