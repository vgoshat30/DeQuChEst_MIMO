import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import math

import numpy
import sympy as sym
from ProjectConstants import *


class network(nn.Module):

    def __init__(self, codebookSize, inputDimention, outputDimention, magic_c,
                 layersDimentions):
        super(network, self).__init__()
        self.l1 = nn.Linear(inputDimention, math.floor(layersDimentions[0] *
                                                       inputDimention))
        # See Hardware-Limited Task-Based Quantization Proposion 3. for the
        # choice of output features
        self.l2 = nn.Linear(math.floor(layersDimentions[0] * inputDimention),
                            outputDimention)
        self.l3 = nn.Linear(outputDimention, math.floor(layersDimentions[1] *
                                                        outputDimention))
        self.l4 = nn.Linear(math.floor(layersDimentions[1] * outputDimention),
                            math.floor(layersDimentions[2] * outputDimention))
        self.l5 = nn.Linear(math.floor(layersDimentions[2] * outputDimention),
                            outputDimention)
        self.q1 = quantizationLayer(
            outputDimention, outputDimention, codebookSize, magic_c)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


class quantizationLayer(Module):
    def __init__(self, in_features, out_features, M, magic_c):
        super(quantizationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebookSize = M
        self.magic_c = magic_c
        # There are two group of parameters: {ai, bi}, as described in the paper
        self.weight = Parameter(torch.Tensor(self.codebookSize - 1, 2))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.numel())
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ret = Variable(torch.zeros(input.size()), requires_grad=False)

        for kk in range(0, self.codebookSize - 1):
            tempVal = torch.add(input, self.weight[kk, 1])
            tempVal = torch.mul(tempVal, self.magic_c)
            tempVal = torch.tanh(tempVal)
            tempVal = torch.mul(tempVal, self.weight[kk, 0])
            ret = torch.add(ret, tempVal)
        return ret


def getParameters(model, magic_c):
    """Extract the {ai, bi} coefficients of the soft quantization function
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

        q : sympy function
            The soft qunatization symbolic function (sum of tanh)
    """

    parameters = model.q1.weight.data.numpy()

    # Coefficients of the tanh
    a = parameters[:, 0]
    b = parameters[:, 1]

    # Sort the coefficients by ascending order of the bi-s
    sortedIndecies = b.argsort()
    a = a[sortedIndecies]
    b = b[sortedIndecies]

    # Create symbolic variable x
    symX = sym.symbols('x')

    sym_tanh = a[0] * sym.tanh(magic_c*(symX + b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(magic_c*(symX + b[ii]))
    # Convert the symbolic functions to numpy friendly (for substitution)
    q = sym.lambdify(symX, sym_tanh, "numpy")

    return a, b, q


def quantize(input, a, b, q):
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
