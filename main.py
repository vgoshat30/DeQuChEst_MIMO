"""Main file of projectA

    Trains and tests:
    -- Passing Gradient
    -- Soft to Hard Quantization
"""

import math

import PassingGradient
import SoftToHardQuantization

from ProjectConstants import *
import UniformQuantizer
from testLogger import *
import UserInterface as Ui

from DataLoader import *

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train(model, optimizer, epoch, train_loader, scheduler=None, c_confines=None):
    model.train()

    if c_confines is not None:
        c_step_size = (c_confines[1] / c_confines[0]) ** (1.0 / C_STEPS_AMOUNT)
        when_to_step = math.floor(len(train_loader) / C_STEPS_AMOUNT)

    for corrEpoch in range(0, epoch):
        model.quantization_layer.c = c_confines[0]
        if scheduler is not None:
            scheduler.step()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.float()), Variable(target.float())

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, 1), target.view(-1, 1))
            loss.backward()
            optimizer.step()

            if c_confines is not None and batch_idx % when_to_step == 0:
                if model.quantization_layer.c * c_step_size < c_confines[1]:
                    model.quantization_layer.c *= c_step_size

            if batch_idx % 10 == 0:
                    Ui.train_iteration(model, corrEpoch, epoch, batch_idx,  data,
                                       train_loader, loss, c_confines)


def test_passing_gradient(model):
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))
        Ui.test_iteration(model.modelname, batch_idx, data, testLoader)

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)

    return test_loss.detach().numpy()


def test_soft_to_hard_quantization(model):
    a, b, c, q = SoftToHardQuantization.get_parameters(model)
    log.log(a=a, b=b, c=c)

    test_loss = 0
    for batch_idx, (data, target) in enumerate(testLoader):
        output, target = Variable(data.float()), Variable(target.float())

        for layerIndex, layer in enumerate(model.layers):
            if layerIndex is model.quantizationLayerIndex:
                output_data = output.data
                output_numpy = output_data.numpy()
                for ii in range(0, output_data.size(0)):
                    for jj in range(0, output_data.size(1)):
                        output[ii, jj], kk = SoftToHardQuantization.quantize(
                            output_numpy[ii, jj], a, b, q)
            else:
                output = layer(output)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))
        Ui.test_iteration(model.modelname, batch_idx, data, testLoader)

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)
    return test_loss.detach().numpy()


# Trying to create a new test log mat file for the case that such one does not exist
log = create_mat_file(TEST_LOG_MAT_FILE)


constantPermutations = [(dataFile, lr, architecture, epoch, codebookSize, c_range)
                        for dataFile in DATA_MAT_FILE
                        for lr in LR_RANGE
                        for architecture in ARCHITECTURE
                        for epoch in EPOCH_RANGE
                        for codebookSize in M_RANGE
                        for c_range in C_INCREMENT_RANGE]

# Iterating over all possible permutations of train epochs, learning rate, and
# codebookSize arrays defined in the ProjectConstants module
for constPerm in constantPermutations:
    dataFile = constPerm[0]
    lr = constPerm[1]
    architecture = constPerm[2]
    corrTopEpoch = constPerm[3]
    codebookSize = constPerm[4]
    c_bounds = constPerm[5]

    loadedDataFile = sio.loadmat(dataFile)

    # Extracting train and test data
    trainX = loadedDataFile['trainX']
    trainS = loadedDataFile['trainS']
    testX = loadedDataFile['dataX']
    testS = loadedDataFile['dataS']

    # Get the class containing the train data from DataLoader.py
    trainData = ShlezDatasetTrain(trainX, trainS)
    # define training dataloader
    trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE,
                             shuffle=True)
    # Do the same for the test data
    testData = ShlezDatasetTest(testX, testS)
    testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE,
                            shuffle=True)

    QUANTIZATION_RATE = math.log2(codebookSize) *\
        trainData.outputDim/trainData.inputDim

    # Generate uniform codebooks for the train sets
    S_codebook = UniformQuantizer.codebook_uniform(trainData.S_var,
                                                   codebookSize)

    # device = torch.device('cpu')
    criterion = nn.MSELoss()

    ########################################################################
    #                  Training and testing all networks                   #
    ########################################################################

    # ------------------------------------------------
    # ---            'Passing Gradient'            ---
    # ------------------------------------------------

    if 'Passing Gradient' in modelsToActivate:
        model_name = 'Passing Gradient'

        # Defining the 'Passing Gradient' model, as described in the paper.
        passingGradient_model = PassingGradient.Network(model_name, S_codebook,
                                                        trainData.inputDim,
                                                        trainData.outputDim,
                                                        architecture)

        passingGradient_optimizer = optim.SGD(
            passingGradient_model.parameters(),
            lr=lr, momentum=0.5)

        passingGradient_scheduler = optim.lr_scheduler.ExponentialLR(
            passingGradient_optimizer, gamma=0.7, last_epoch=-1)

        # Training 'Passing Gradient':
        Ui.train_message(passingGradient_model, dataFile, corrTopEpoch, lr,
                         codebookSize, architecture)
        model_linUniformQunat_runtime = datetime.now()
        train(passingGradient_model, passingGradient_optimizer, corrTopEpoch,
              trainLoader, passingGradient_scheduler)
        model_linUniformQunat_runtime = datetime.now() - \
            model_linUniformQunat_runtime

        # Testing 'Passing Gradient':
        Ui.test_message(model_name)
        model_linUniformQunat_loss = test_passing_gradient(passingGradient_model)
        Ui.test_results(model_name, corrTopEpoch, lr, codebookSize,
                        QUANTIZATION_RATE,
                        model_linUniformQunat_loss)
        log.log(rate=QUANTIZATION_RATE, loss=model_linUniformQunat_loss,
                algorithm=model_name,
                codewordNum=codebookSize,
                learningRate=lr,
                layersDim=architecture,
                epochs=corrTopEpoch,
                runtime=model_linUniformQunat_runtime,
                dataFile=dataFile)

    # ------------------------------------------------
    # ---        'Soft to Hard Quantization'       ---
    # ------------------------------------------------

    if 'Soft to Hard Quantization' in modelsToActivate:
        model_name = 'Soft to Hard Quantization'

        # Defining the 'Soft to Hard Quantization' model, as described in the
        # paper.
        if FORCE_QUANTIZATION_DIM:
            quantization_dimention = trainData.outputDim
        else:
            quantization_dimention = None
        softToHardQuantization_model = SoftToHardQuantization.Network(
            model_name, codebookSize, trainData.inputDim, trainData.outputDim,
            c_bounds[0], architecture, quantization_dimention)

        softToHardQuantization_optimizer = optim.SGD(
            softToHardQuantization_model.parameters(), lr=lr, momentum=0.5)

        softToHardQuantization_scheduler = optim.lr_scheduler.ExponentialLR(
            softToHardQuantization_optimizer, gamma=0.7, last_epoch=-1)

        # Training 'Soft to Hard Quantization':
        Ui.train_message(softToHardQuantization_model, dataFile, corrTopEpoch,
                         lr, codebookSize, architecture, c_bounds)
        model_tanhQuantize_runtime = datetime.now()
        train(softToHardQuantization_model, softToHardQuantization_optimizer,
              corrTopEpoch, trainLoader, softToHardQuantization_scheduler, c_bounds)
        model_tanhQuantize_runtime = datetime.now() - model_tanhQuantize_runtime

        # Testing 'Soft to Hard Quantization':
        Ui.test_message(model_name)
        model_tanhQuantize_loss = \
            test_soft_to_hard_quantization(softToHardQuantization_model)

        Ui.test_results(model_name, corrTopEpoch, lr, codebookSize,
                        QUANTIZATION_RATE, model_tanhQuantize_loss, c_bounds)
        log.log('last', rate=QUANTIZATION_RATE, loss=model_tanhQuantize_loss,
                algorithm=model_name,
                codewordNum=codebookSize,
                learningRate=lr,
                layersDim=architecture,
                epochs=corrTopEpoch,
                runtime=model_tanhQuantize_runtime,
                dataFile=dataFile)
