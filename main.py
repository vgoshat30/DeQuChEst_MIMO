"""Main file of projectA

    Trains and tests:
    -- Passing Gradient
    -- Soft to Hard Quantization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
import math
from datetime import datetime

import PassingGradient
import SoftToHardQuantization

from dataLoader import *
from ProjectConstants import *
import UniformQuantizer
from testLogger import *
import UserInterface as UI


def train(modelname, epoch, model, optimizer, scheduler=None, InitIncFactorC=1, DecayC=1):
    model.train()
    incFactor = InitIncFactorC
    for corrEpoch in range(0, epoch):
        incFactor = incFactor ** DecayC
        if scheduler != None:
            scheduler.step()
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = Variable(data.float()), Variable(target.float())

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, 1), target.view(-1, 1))
            loss.backward(retain_graph=True)
            optimizer.step()
            if modelname == 'Soft to Hard Quantization'
                model.q1.weight[:, 2] = torch.mul(model.q1.weight[:, 2], incFactor)
            if batch_idx % 10 == 0:
                UI.trainIteration(modelname, corrEpoch, epoch, batch_idx, data,
                                  trainLoader, loss)


def testPassingGradient(modelname, model):
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))
        UI.testIteration(modelname, batch_idx, data, testLoader)

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)

    return test_loss.detach().numpy()


def testSoftToHardQuantization(modelname, model, codebookSize):
    a, b, c, q = SoftToHardQuantization.getParameters(model)
    log.log(a=a, b=b, c=c)

    classificationCounter = np.zeros(codebookSize)
    test_loss = 0
    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model.l1(data)
        output = model.l2(output)
        output_data = output.data
        output_numpy = output_data.numpy()
        for ii in range(0, output_data.size(0)):
            for jj in range(0, output_data.size(1)):
                output[ii, jj], kk = SoftToHardQuantization.quantize(
                    output_numpy[ii, jj], a, b, c, q)
                classificationCounter[kk] += 1
        output = model.l3(output)
        output = model.l4(output)
        output = model.l5(output)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))
        UI.testIteration(modelname, batch_idx, data, testLoader)

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)
    return test_loss.detach().numpy(), classificationCounter


#########################################################################
###                 Extracting Train and Test Data                    ###
#########################################################################



# Get the class containing the train data from dataLoader.py
trainData = ShlezDatasetTrain()
# define training dataloader
trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# Do the same for the test data
testData = ShlezDatasetTest()
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=True)


constantPermutationns = [(epoch, lr, codebookSize)
                         for epoch in EPOCH_RANGE
                         for lr in LR_RANGE
                         for codebookSize in M_RANGE]

# Iterating on all possible remutations of train epochs, learning rate, and
# codebookSize arrays defined in the ProjectConstants module
for constPerm in constantPermutationns:
    corrTopEpoch = constPerm[0]
    lr = constPerm[1]
    codebookSize = constPerm[2]
    QUANTIZATION_RATE = math.log2(codebookSize)*OUTPUT_DIMENSION/INPUT_DIMENSION

    # Generate uniform codebooks for the train sets
    S_codebook = UniformQuantizer.codebook_uniform(trainData.S_var,
                                                   codebookSize)

    #########################################################################
    ###                    Initializing both models                       ###
    #########################################################################

    # The 'Passing Gradinet' model, as described in the paper.
    passingGradient_model = PassingGradient.network(S_codebook)
    # The 'Soft to Hard Quantization' model, as described in the paper.
    softToHardQuantization_model = SoftToHardQuantization.network(
        codebookSize=codebookSize)

    criterion = nn.MSELoss()

    passingGradient_optimizer = optim.SGD(passingGradient_model.parameters(),
                                          lr=lr, momentum=0.5)
    softToHardQuantization_optimizer = optim.SGD(
        softToHardQuantization_model.parameters(), lr=lr, momentum=0.5)

    passingGradient_scheduler = optim.lr_scheduler.ExponentialLR(
        passingGradient_optimizer, gamma=0.7, last_epoch=-1)
    softToHardQuantization_scheduler = optim.lr_scheduler.ExponentialLR(
        softToHardQuantization_optimizer, gamma=0.7, last_epoch=-1)

    ########################################################################
    ###               Training and testing all networks                  ###
    ########################################################################

    # Assigning the log mat file to a testlogger class
    theory = sio.loadmat('theoreticalBounds.mat')
    theoryRate = theory['v_fRate']
    theoryLoss = theory['m_fCurves']
    # Trying ti create a new test log mat file for the case that such one does
    # not exist
    log = createMatFile('testLog.mat', 'tanh', theoryRate, theoryLoss)

    # ------------------------------------------------
    # ---            'Passing Gradient'            ---
    # ------------------------------------------------

    if 'Passing Gradient' in modelsToActivate:
        modelname = 'Passing Gradient'

        # Training 'Passing Gradient':

        UI.trainMessage(modelname, corrTopEpoch, lr, codebookSize)
        model_linUniformQunat_runtime = datetime.now()
        train(modelname, corrTopEpoch, passingGradient_model,
              passingGradient_optimizer, passingGradient_scheduler)
        model_linUniformQunat_runtime = datetime.now() - \
            model_linUniformQunat_runtime

        # Testing 'Passing Gradient':

        UI.testMessage(modelname)
        model_linUniformQunat_loss = testPassingGradient(modelname,
                                                         passingGradient_model)
        UI.testResultsPassingGradient(modelname, corrTopEpoch, lr, codebookSize,
                                      QUANTIZATION_RATE,
                                      model_linUniformQunat_loss)
        log.log(rate=QUANTIZATION_RATE, loss=model_linUniformQunat_loss,
                algorithm=modelname,
                codewordNum=codebookSize,
                learningRate=lr,
                epochs=corrTopEpoch,
                runtime=model_linUniformQunat_runtime)

    # ------------------------------------------------
    # ---        'Soft to Hard Quantization'       ---
    # ------------------------------------------------

    if 'Soft to Hard Quantization' in modelsToActivate:
        modelname = 'Soft to Hard Quantization'

        # Training 'Soft to Hard Quantization':

        UI.trainMessage(modelname, corrTopEpoch, lr, codebookSize)
        model_tanhQuantize_runtime = datetime.now()
        train(modelname, corrTopEpoch, softToHardQuantization_model,
              softToHardQuantization_optimizer,
              softToHardQuantization_scheduler)
        model_tanhQuantize_runtime = datetime.now() - model_tanhQuantize_runtime

        # Testing 'Soft to Hard Quantization':

        UI.testMessage(modelname)
        model_tanhQuantize_loss, classificationByWord = \
            testSoftToHardQuantization(modelname, softToHardQuantization_model,
                                       codebookSize)

        UI.testResultsSoftToHard(modelname, corrTopEpoch, lr, codebookSize,
                                 QUANTIZATION_RATE, model_tanhQuantize_loss,
                                 classificationByWord)
        log.log('last', rate=QUANTIZATION_RATE, loss=model_tanhQuantize_loss,
                algorithm=modelname,
                codewordNum=codebookSize,
                learningRate=lr,
                epochs=corrTopEpoch,
                runtime=model_tanhQuantize_runtime)
