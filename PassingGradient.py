import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import math

from ProjectConstants import *
import UniformQuantizer


class network(nn.Module):

    def __init__(self, codebook, inputDimention, outputDimention,
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
        self.q1 = quantizationLayer.apply
        self.codebook = codebook

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.q1(x, self.codebook)
        x = self.l3(x)
        x = self.l4(x)
        return self.l5(x)


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
