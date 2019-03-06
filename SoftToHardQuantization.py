import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as f

import math
import numpy
import sympy as sym

import numpy as np


class Network(nn.Module):

    def __init__(self, modelname, codebook_size, train_data, a,
                 initial_c, architecture, force_quantization_dim=None):
        super(Network, self).__init__()

        self.modelname = modelname

        self.layers = []
        self.quantizationLayerIndex = []

        input_dimension = train_data.inputDim
        output_dimension = train_data.outputDim

        quantization_layer_exists = False
        next_input_dimension = input_dimension
        waiting_for_linear_layer_ratio = False
        layer_name_index = 1
        # Find last occurrence of integer (layer dimension multiplier)
        # in the architecture list
        last_dimension_entry = 0
        for index, arcParser in enumerate(architecture):
            if isinstance(arcParser, int) or isinstance(arcParser, float):
                last_dimension_entry = index
        # Construct the array of layers.
        for archIndex, arcParser in enumerate(architecture):
            # If a scalable layer specified, its dimensions
            # multiplier should follow
            if waiting_for_linear_layer_ratio:
                # If its the last scalable layer, make sure that its output
                # dimension is the output dimension of the module
                if archIndex == last_dimension_entry:
                    layer_out_dim = output_dimension
                # If the next layer is the quantization layer, forcing the
                # output of the scalable layer to be outputDimension
                elif (architecture[archIndex + 1] is 'quantization' and
                      force_quantization_dim is not None):
                    layer_out_dim = force_quantization_dim
                else:
                    layer_out_dim = math.floor(arcParser * next_input_dimension)
                # Adding layer to layers array (for forward implementation)
                if waiting_for_linear_layer_ratio:
                    self.layers.append(nn.Linear(next_input_dimension, layer_out_dim))
                # Assigning layer to module
                self.add_module('l' + str(layer_name_index), self.layers[-1])
                layer_name_index += 1
                next_input_dimension = layer_out_dim
                waiting_for_linear_layer_ratio = False
            # If requested linear layer, expecting its dimension multiplier in
            # next loop iteration
            elif arcParser is 'linear':
                waiting_for_linear_layer_ratio = True
            # If requested non linear layer
            elif arcParser is 'relu':
                # Adding layer to layers array (for forward implementation)
                self.layers.append(f.relu)
            elif arcParser is 'quantization':
                if quantization_layer_exists:
                    raise ValueError('Cant have two quantization layers')
                quantization_layer_exists = True
                # Set the input dimension of the nex linear layer to be as the
                # output of the quantization layer
                if force_quantization_dim is not None:
                    next_input_dimension = force_quantization_dim
                # Creating quantization layer
                self.quantization_layer = QuantizationLayer(next_input_dimension,
                                                            next_input_dimension,
                                                            codebook_size, a, initial_c,
                                                            np.sqrt(train_data.S_var))
                # Adding layer to layers array (for forward implementation)
                self.layers.append(self.quantization_layer)
                # Assigning layer to module
                self.add_module('l' + str(layer_name_index), self.layers[-1])
                # self.quantizationLayerNameIndex = layer_name_index
                self.quantizationLayerIndex = len(self.layers) - 1
                layer_name_index += 1
            else:
                raise ValueError('Unknown layer type: \'{}\''.format(arcParser))

    def forward(self, x):
        for current_layer in self.layers:
            x = current_layer(x)
        return x


class QuantizationLayer(Module):
    def __init__(self, in_features, out_features, m, ai, ci, train_std):
        super(QuantizationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebookSize = m
        self.a = ai
        self.b = np.linspace(-2, 2, self.codebookSize)
        # Parameter(torch.Tensor(self.codebookSize - 1))
        self.c = ci
        self.reset_parameters(train_std/2)

    def reset_parameters(self, reset_bounds):
        return
        # self.b.data = Parameter(torch.Tensor(np.linspace(-reset_bounds, reset_bounds, self.codebookSize)))

    def forward(self, x):
        # noinspection PyUnresolvedReferences
        ret = Variable(torch.zeros(x.size()), requires_grad=False)

        for kk in range(0, self.codebookSize - 1):
            # noinspection PyUnresolvedReferences
            temp_val = torch.sub(x, self.b[kk])
            # noinspection PyUnresolvedReferences
            temp_val = torch.mul(temp_val, self.c)
            # noinspection PyUnresolvedReferences
            temp_val = torch.tanh(temp_val)
            # noinspection PyUnresolvedReferences
            temp_val = torch.mul(temp_val, self.a)
            # noinspection PyUnresolvedReferences
            ret = torch.add(ret, temp_val)
        return ret


def get_parameters(model):
    """Extract the {ai, bi} coefficients of the soft quantization function
    from the network model, return them sorted by ascending order of the
    bi coefficients, and create a symbolic function which will be used in the
    hard quantization process (the 'quantize' function below)

    Parameters
    ----------
        model : network (from this module)
            The network instance of the 'Soft to Hard Quantization' model

    Returns
    -------
        a : list
            The ai coefficients of the sum of tanh soft quantization function
        b : list
            The bi coefficients

        q : sympy function
            The soft quantization symbolic function (sum of tanh)
    """

    # Coefficients of the tanh
    a = model.quantization_layer.a
    b = model.quantization_layer.b #.data.numpy()
    c = model.quantization_layer.c

    # # Sort the coefficients by ascending order of the bi-s
    # sorted_indexes = b.argsort()
    # # a = a[sorted_indexes]
    # b = b[sorted_indexes]

    # Create symbolic variable x
    sym_x = sym.symbols('x')

    # sym_tanh = np.square(a[0]) * sym.tanh(c*(sym_x - b[0]))
    # for ii in range(1, len(b)):
    #     sym_tanh = sym_tanh + np.square(a[ii]) * sym.tanh(c*(sym_x - b[ii]))
    # # Convert the symbolic functions to numpy friendly (for substitution)
    # q = sym.lambdify(sym_x, sym_tanh, "numpy")

    sym_tanh = a * sym.tanh(c * (sym_x - b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a * sym.tanh(c * (sym_x - b[ii]))
    # Convert the symbolic functions to numpy friendly (for substitution)
    q = sym.lambdify(sym_x, sym_tanh, "numpy")

    return a, b, c, q


def quantize(x, a, b, q):
    """Get the result of the hard quantization function derived from the soft
    one (the meaning of the name 'SoftToHardQuantization') and the codeword the
    input was sorted to

    Parameters
    ----------
        x : float
            The quantizer input
        a : numpy.ndarray
            ai coefficients
        b : numpy.ndarray
            bi coefficients
        q : sympy.FunctionClass
            Symbolic lambdified quantization function

    Returns
    -------
        quantizedValue : int
            The output of the quantizer
        codeword : int
            The index of the corresponding codeword
    """

    # if x <= b[0]:
    #     return -sum(np.square(a)), 0
    # if x > b[-1]:
    #     return sum(np.square(a)), len(b)
    # for ii in range(0, len(b)):
    #     if b[ii] < x <= b[ii + 1]:
    #         return q((b[ii] + b[ii+1])/2), ii + 1

    if x <= b[0]:
        return -a*b.size, 0
    if x > b[-1]:
        return a*b.size, len(b)
    for ii in range(0, len(b)):
        if b[ii] < x <= b[ii + 1]:
            return q((b[ii] + b[ii+1])/2), ii + 1
