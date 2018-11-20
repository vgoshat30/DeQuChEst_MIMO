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
        # Find last occurrence of integer (linear layer dimension mutiplier)
        # in the architecture list
        lastDimentionEntry = 0
        layerIndex = 1
        for index, arcParser in enumerate(architecture):
            if isinstance(arcParser, int) or isinstance(arcParser, float):
                lastDimentionEntry = index
        # Construct the array of layesr.
        for archIndex, arcParser in enumerate(architecture):
            # If a linear layer specified, its dimentions multiplier should
            # follow
            if waiting_for_linear_layer_value:
                # If its the last linear layer, making shure that its output
                # dimension is the outout dimension of the module
                if archIndex == lastDimentionEntry:
                    layerOutDim = output_dimension
                # If the next layer is the quantization layer, forcing the
                # output of the linear layer to be outputDimention
                elif architecture[archIndex + 1] is 'quantization':
                    layerOutDim = output_dimension
                else:
                    layerOutDim = math.floor(arcParser *
                                             previous_input_dimension)
                # Adding layer to layers array (for forward implementation)
                self.layers.append(
                    nn.Linear(previous_input_dimension, layerOutDim))
                # Asining layer to module
                self.add_module('l' + str(layerIndex), self.layers[-1])
                layerIndex += 1
                previous_input_dimension = layerOutDim
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
                # ouptput of the quantization layer
                previous_input_dimension = output_dimension
                # Adding layer to layers array (for forward implementation)
                self.layers.append(quantizationLayer.apply)
                self.quantizationLayerIndex = layerIndex - 1
                layerIndex += 1
            else:
                raise ValueError('Invalid layer type: ' + arcParser)
                return

    def forward(self, x):
        for correntLayerIndex, correntLayer in enumerate(self.layers):
            if correntLayerIndex == self.quantizationLayerIndex:
                x = correntLayer(x, self.codebook)
            else:
                x = correntLayer(x)
        return(x)


class quantizationLayer(torch.autograd.Function):
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
            qunatized_input
                the quantized input formed using codebook. The quantized input
                is the closest codeword avaliable in codeword.

    Attributes
    ----------
        codebook
            the fixed codebook of the module of shape `(M x 1)`
            which will construct the quantized input

    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, codebook):
        ctx.save_for_backward(input)
        input_data = input.data
        input_numpy = input_data.numpy()
        qunatized_input = torch.zeros(input.size())
        for ii in range(0, input_data.size(0)):
            for jj in range(0, input_data.size(1)):
                qunatized_input[ii, jj], __ = UniformQuantizer.get_optimal_word(
                    input_numpy[ii, jj], codebook)
        return qunatized_input

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None
