import torch
import torch.nn as nn
import torch.nn.functional as f

import math

import UniformQuantizer


class Network(nn.Module):

    def __init__(self, modelname, codebook, input_dimension, output_dimension,
                 architecture):
        super(Network, self).__init__()

        self.modelname = modelname
        self.codebook = codebook

        self.layers = []
        previous_input_dimension = input_dimension
        waiting_for_linear_layer_value = False
        # Find last occurrence of integer (linear layer dimension multiplier)
        # in the architecture list
        last_dimension_entry = 0
        layer_index = 1
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
                self.add_module('l' + str(layer_index), self.layers[-1])
                layer_index += 1
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
                self.layers.append(QuantizationLayer.apply)
                self.quantizationLayerIndex = layer_index - 1
                layer_index += 1
            else:
                raise ValueError('Invalid layer type: ' + arcParser)

    def forward(self, x):
        for currentLayerIndex, currentLayer in enumerate(self.layers):
            if currentLayerIndex == self.quantizationLayerIndex:
                x = currentLayer(x, self.codebook)
            else:
                x = currentLayer(x)
        return x


class QuantizationLayer(torch.autograd.Function):
    """Applies a quantization process to the incoming data.
        Can be integrated as activation layer of NN, therefore useful for cases
        we want to check the performance while keeping the whole system as one
        neural network. This function is "hidden" to the backpropegation, i.e.,
        it moves backward its input gradient.

    Shape
    -----
        Input
            To be filled...
        Output
            quantized_input
                the quantized input formed using codebook. The quantized input
                is the closest codeword avaliable in codeword.

    Attributes
    ----------
        codebook
            the fixed codebook of the module of shape `(M x 1)`
            which will construct the quantized input

    """
    # Note that both forward and backward are @staticmethod
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, codebook):
        ctx.save_for_backward(x)
        input_data = x.data
        input_numpy = input_data.numpy()
        quantized_input = torch.zeros(x.size())
        for ii in range(0, input_data.size(0)):
            for jj in range(0, input_data.size(1)):
                quantized_input[ii, jj], __ = UniformQuantizer.get_optimal_word(
                    input_numpy[ii, jj], codebook)
        return quantized_input

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None
