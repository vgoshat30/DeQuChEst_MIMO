import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F

import math
import numpy
import sympy as sym

from ProjectConstants import *
import UserInterface as UI


class network(nn.Module):

    def __init__(self, modelname, codebookSize, inputDimention, outputDimention,
                 magic_c, architecture):
        super(network, self).__init__()

        self.modelname = modelname

        self.layers = []
        previousInputDimention = inputDimention
        waitingForLinearLayerValue = False
        layerNameIndex = 1
        # Find last occurance of integer (linear layer dimention mutiplier)
        # in the architecture list
        lastDimentionEntry = 0
        for index, arcParser in enumerate(architecture):
            if isinstance(arcParser, int) or isinstance(arcParser, float):
                lastDimentionEntry = index
        # Construct the array of layesr.
        for archIndex, arcParser in enumerate(architecture):
            # If a linear layer specified, its dimentions multiplier should
            # follow
            if waitingForLinearLayerValue:
                # If its the last linear layer, making shure that its output
                # dimention is the outout dimention of the module
                if archIndex == lastDimentionEntry:
                    layerOutDim = outputDimention
                # If the next layer is the quantization layer, forcing the
                # output of the linear layer to be outputDimention
                elif architecture[archIndex + 1] is 'quantization':
                    layerOutDim = outputDimention
                else:
                    layerOutDim = math.floor(arcParser *
                                             previousInputDimention)
                # Adding layer to layers array (for forward implementation)
                self.layers.append(
                    nn.Linear(previousInputDimention, layerOutDim))
                # Asining layer to module
                self.add_module('l' + str(layerNameIndex), self.layers[-1])
                layerNameIndex += 1
                previousInputDimention = layerOutDim
                waitingForLinearLayerValue = False
            # If requested linear layer, expecting its dimention multiplier in
            # next loop iteration
            elif arcParser is 'linear':
                waitingForLinearLayerValue = True
            # If requested non linear layer
            elif arcParser is 'relu':
                # Adding layer to layers array (for forward implementation)
                self.layers.append(F.relu)
            elif arcParser is 'quantization':
                # Set the input dimention of the nex linear layer to be as the
                # ouptput of the quantization layer
                previousInputDimention = outputDimention
                # Adding layer to layers array (for forward implementation)
                self.layers.append(quantizationLayer(outputDimention,
                                                     outputDimention,
                                                     codebookSize, magic_c))
                # Asining layer to module
                self.add_module('l' + str(layerNameIndex), self.layers[-1])
                self.quantizationLayerNameIndex = layerNameIndex
                self.quantizationLayerIndex = len(self.layers) - 1
                layerNameIndex += 1
            else:
                raise ValueError('Invalid layer type: ' + arcParser)
                return

    def forward(self, x):
        for correntLayer in self.layers:
            x = correntLayer(x)
        return(x)


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
    quantLayer = getattr(model, 'l' + str(model.quantizationLayerNameIndex))
    parameters = quantLayer.weight.data.numpy()

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
