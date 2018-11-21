import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as f

import math
import numpy
import sympy as sym


class Network(nn.Module):

    def __init__(self, modelname, codebook_size, input_dimension, output_dimension,
                 initial_c, architecture):
        super(Network, self).__init__()

        self.modelname = modelname

        self.layers = []
        self.quantization_layer = QuantizationLayer(output_dimension,
                                                    output_dimension,
                                                    codebook_size, initial_c)

        previous_input_dimension = input_dimension
        waiting_for_linear_layer_value = False
        layer_name_index = 1
        # Find last occurrence of integer (linear layer dimension multiplier)
        # in the architecture list
        last_dimension_entry = 0
        for index, arcParser in enumerate(architecture):
            if isinstance(arcParser, int) or isinstance(arcParser, float):
                last_dimension_entry = index
        # Construct the array of layers.
        for archIndex, arcParser in enumerate(architecture):
            # If a linear layer specified, its dimensions multiplier should
            # follow
            if waiting_for_linear_layer_value:
                # If its the last linear layer, making sure that its output
                # dimension is the output dimension of the module
                if archIndex == last_dimension_entry:
                    layer_out_dim = output_dimension
                # If the next layer is the quantization layer, forcing the
                # output of the linear layer to be outputDimension
                elif architecture[archIndex + 1] is 'quantization':
                    layer_out_dim = output_dimension
                else:
                    layer_out_dim = math.floor(arcParser * previous_input_dimension)
                # Adding layer to layers array (for forward implementation)
                self.layers.append(
                    nn.Linear(previous_input_dimension, layer_out_dim))
                # Assigning layer to module
                self.add_module('l' + str(layer_name_index), self.layers[-1])
                layer_name_index += 1
                previous_input_dimension = layer_out_dim
                waiting_for_linear_layer_value = False
            # If requested linear layer, expecting its dimension multiplier in
            # next loop iteration
            elif arcParser is 'linear':
                waiting_for_linear_layer_value = True
            # If requested non linear layer
            elif arcParser is 'relu':
                # Adding layer to layers array (for forward implementation)
                self.layers.append(f.relu)
            elif arcParser is 'quantization':
                # Set the input dimension of the nex linear layer to be as the
                # output of the quantization layer
                previous_input_dimension = output_dimension
                # Adding layer to layers array (for forward implementation)
                self.layers.append(self.quantization_layer)
                # Assigning layer to module
                self.add_module('l' + str(layer_name_index), self.layers[-1])
                # self.quantizationLayerNameIndex = layer_name_index
                self.quantizationLayerIndex = len(self.layers) - 1
                layer_name_index += 1
            else:
                raise ValueError('Invalid layer type: ' + arcParser)

    def forward(self, x):
        for current_layer in self.layers:
            x = current_layer(x)
        return x


class QuantizationLayer(Module):
    def __init__(self, in_features, out_features, m, ci):
        super(QuantizationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebookSize = m
        self.c = ci
        # There are two group of parameters: {ai, bi}, as described in the paper
        self.weight = Parameter(torch.Tensor(self.codebookSize - 1, 2))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.numel())
        self.weight.data.uniform_(-std, std)

    def forward(self, x):
        # noinspection PyUnresolvedReferences
        ret = Variable(torch.zeros(x.size()), requires_grad=False)

        for kk in range(0, self.codebookSize - 1):
            # noinspection PyUnresolvedReferences
            temp_val = torch.add(x, self.weight[kk, 1])
            # noinspection PyUnresolvedReferences
            temp_val = torch.mul(temp_val, self.c)
            # noinspection PyUnresolvedReferences
            temp_val = torch.tanh(temp_val)
            # noinspection PyUnresolvedReferences
            temp_val = torch.mul(temp_val, self.weight[kk, 0])
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
        magic_c : float
            The slope of the hyperbolic tangents

    Returns
    -------
        a : list
            The ai coefficients of the sum of tanh soft quantization function
        b : list
            The bi coefficients

        q : sympy function
            The soft quantization symbolic function (sum of tanh)
    """
    # quantization_layer = getattr(model, 'l' + str(model.quantizationLayerNameIndex))

    parameters = model.quantization_layer.weight.data.numpy()

    # Coefficients of the tanh
    a = parameters[:, 0]
    b = parameters[:, 1]
    c = model.quantization_layer.c

    # Sort the coefficients by ascending order of the bi-s
    sorted_indexes = b.argsort()
    a = a[sorted_indexes]
    b = b[sorted_indexes]

    # Create symbolic variable x
    sym_x = sym.symbols('x')

    sym_tanh = a[0] * sym.tanh(c*(sym_x + b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(c*(sym_x + b[ii]))
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

    if x <= b[0]:
        return -sum(a), 0
    if x > b[-1]:
        return sum(a), len(b)
    for ii in range(0, len(b)):
        if b[ii] < x <= b[ii + 1]:
            return q((b[ii] + b[ii+1])/2), ii + 1
