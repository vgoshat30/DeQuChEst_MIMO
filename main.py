"""Main file of projectA

    Trains and tests:
    -- Passing Gradient
    -- Soft to Hard Quantization
"""

import math

import PassingGradient
import SoftToHardQuantization

from dataLoader import *
from ProjectConstants import *
import UniformQuantizer
from testLogger import *
import UserInterface as Ui


def train(model, optimizer, epoch, scheduler=None):
    model.train()
    for corrEpoch in range(0, epoch):
        if scheduler is not None:
            scheduler.step()
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = Variable(data.float()), Variable(target.float())

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, 1), target.view(-1, 1))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                Ui.trainIteration(model.modelname, corrEpoch, epoch, batch_idx,
                                  data, trainLoader, loss)


def test_passing_gradient(model):
    model.eval()
    test_loss = 0

    for batch_idx, (data, target) in enumerate(testLoader):
        data, target = Variable(data.float()), Variable(target.float())
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output.view(-1, 1), target.view(-1, 1))
        Ui.testIteration(model.modelname, batch_idx, data, testLoader)

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)

    return test_loss.detach().numpy()


def test_soft_to_hard_quantization(model, magic_c):
    a, b, q = SoftToHardQuantization.get_parameters(model, magic_c)
    log.log(a=a, b=b)

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
        Ui.testIteration(model.modelname, batch_idx, data, testLoader)

    test_loss /= (len(testLoader.dataset)/BATCH_SIZE)
    return test_loss.detach().numpy()


# Trying to create a new test log mat file for the case that such one does not exist
log = createMatFile(TEST_LOG_MAT_FILE)


constantPermutations = [(dataFile, lr, magic_c,
                         architecture, epoch, codebookSize)
                        for dataFile in DATA_MAT_FILE
                        for lr in LR_RANGE
                        for magic_c in MAGIC_C_RANGE
                        for architecture in ARCHITECTURE
                        for epoch in EPOCH_RANGE
                        for codebookSize in M_RANGE]

# Iterating on all possible permutations of train epochs, learning rate, and
# codebookSize arrays defined in the ProjectConstants module
for constPerm in constantPermutations:
    dataFile = constPerm[0]
    lr = constPerm[1]
    magic_c = constPerm[2]
    architecture = constPerm[3]
    corrTopEpoch = constPerm[4]
    codebookSize = constPerm[5]

    loadedDataFile = sio.loadmat(dataFile)

    # Extracting train and test data
    trainX = loadedDataFile['trainX']
    trainS = loadedDataFile['trainS']
    testX = loadedDataFile['dataX']
    testS = loadedDataFile['dataS']

    # Get the class containing the train data from dataLoader.py
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
        modelname = 'Passing Gradient'

        # Defining the 'Passing Gradient' model, as described in the paper.
        passingGradient_model = PassingGradient.Network(modelname, S_codebook,
                                                        trainData.inputDim,
                                                        trainData.outputDim,
                                                        architecture)

        passingGradient_optimizer = optim.SGD(
            passingGradient_model.parameters(),
            lr=lr, momentum=0.5)

        passingGradient_scheduler = optim.lr_scheduler.ExponentialLR(
            passingGradient_optimizer, gamma=0.7, last_epoch=-1)

        # Training 'Passing Gradient':
        Ui.trainMessage(passingGradient_model, dataFile, corrTopEpoch, lr,
                        codebookSize, architecture)
        model_linUniformQunat_runtime = datetime.now()
        train(passingGradient_model, passingGradient_optimizer, corrTopEpoch,
              passingGradient_scheduler)
        model_linUniformQunat_runtime = datetime.now() - \
            model_linUniformQunat_runtime

        # Testing 'Passing Gradient':
        Ui.testMessage(modelname)
        model_linUniformQunat_loss = test_passing_gradient(passingGradient_model)
        Ui.testResults(modelname, corrTopEpoch, lr, codebookSize,
                       QUANTIZATION_RATE,
                       model_linUniformQunat_loss)
        log.log(rate=QUANTIZATION_RATE, loss=model_linUniformQunat_loss,
                algorithm=modelname,
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
        modelname = 'Soft to Hard Quantization'

        # Defining the 'Soft to Hard Quantization' model, as described in the
        # paper.
        softToHardQuantization_model = SoftToHardQuantization.Network(
            modelname, codebookSize, trainData.inputDim, trainData.outputDim,
            magic_c, architecture)

        softToHardQuantization_optimizer = optim.SGD(
            softToHardQuantization_model.parameters(), lr=lr, momentum=0.5)

        softToHardQuantization_scheduler = optim.lr_scheduler.ExponentialLR(
            softToHardQuantization_optimizer, gamma=0.7, last_epoch=-1)

        # Training 'Soft to Hard Quantization':
        Ui.trainMessage(softToHardQuantization_model, dataFile, corrTopEpoch,
                        lr, codebookSize, architecture, magic_c)
        model_tanhQuantize_runtime = datetime.now()
        train(softToHardQuantization_model, softToHardQuantization_optimizer,
              corrTopEpoch, softToHardQuantization_scheduler)
        model_tanhQuantize_runtime = datetime.now() - model_tanhQuantize_runtime

        # Testing 'Soft to Hard Quantization':
        Ui.testMessage(modelname)
        model_tanhQuantize_loss = \
            test_soft_to_hard_quantization(softToHardQuantization_model, magic_c)

        Ui.testResults(modelname, corrTopEpoch, lr, codebookSize,
                       QUANTIZATION_RATE, model_tanhQuantize_loss, magic_c)
        log.log('last', rate=QUANTIZATION_RATE, loss=model_tanhQuantize_loss,
                algorithm=modelname,
                codewordNum=codebookSize,
                learningRate=lr,
                layersDim=architecture,
                epochs=corrTopEpoch,
                magic_c=magic_c,
                runtime=model_tanhQuantize_runtime,
                dataFile=dataFile)
