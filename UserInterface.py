from ProjectConstants import *

########################################################################
###                        Training Messages                         ###
########################################################################


def trainMessage(model, dataFile, epoch, lr, codebookSize, architecture,
                 magic_c=None):
    print('\n\n'
          '==================================================================='
          '\n\n\tTraining \'{}\' Model\n\n'
          'Data file:\t{}\n'
          'Epochs number:\t{}\n'
          'Learning rate:\t{}\n'
          'Codebook Size:\t{}\n'
          'MAGIC_C:\t{}\n\n'
          'Architecture:\n'
          .format(model.modelname, dataFile, epoch, lr, codebookSize, magic_c))

    printIndex = 0
    for param in model.parameters():
        dimToPrint = list(param.size())
        dimToPrint.reverse()
        if len(dimToPrint) > 1:
            while not(type(architecture[printIndex]) is str):
                printIndex += 1
                print('DEBUGGING 1:', printIndex)
                if architecture[printIndex] is 'relu':
                    print('relu')
                    printIndex += 1
                elif architecture[printIndex] is 'quantization':
                    print('quantization')
                    printIndex += 1
            else:
                print(architecture[printIndex], dimToPrint)
            printIndex += 1

    print('\n\n'
          '===================================================================')


def trainIteration(modelname, corrEpoch, epoch, batch_idx, data, trainLoader,
                   loss):
    print('Training \'{}\' Model:\tEpoch: {}/{} [{}/{} ({:.0f}%)]\t'
          'Linear Loss: {:.6f}'
          .format(modelname, corrEpoch+1, epoch, batch_idx * len(data),
                  len(trainLoader.dataset),
                  100. * batch_idx / len(trainLoader), loss))


########################################################################
###                        Testing Messages                          ###
########################################################################


def testMessage(modelName):
    print('\n\n'
          '===================================================================='
          '\n\tTesting \'{}\' Model\n'
          '===================================================================='
          '\n\n'
          .format(modelName))


def testIteration(modelname, batch_idx, data, testLoader):
    print('Testing \'{}\' Model:\t[{}/{} ({:.0f}%)]'
          .format(modelname, batch_idx * len(data),
                  len(testLoader.dataset),
                  100. * batch_idx / len(testLoader)))


def testResults(modelname, epoch, lr, codebookSize, rate, loss, magic_c=None):
    print('\n\n'
          '===================================================================='
          '\n\n\tResults of \'{}\' Testing\n\n'
          '_________________________________\n'
          '|\tTraining Parameters\t|\n'
          '|\t\t\t\t|\n'
          '| - Epochs number:\t{}\t|\n'
          '| - Learning Rate:\t{}\t|\n'
          '| - Codebook Size:\t{}\t|\n'
          '| - MAGIC_C:\t\t{}\t|\n'
          '|_______________________________|\n\n'
          'Rate:\t{}\n'
          'Average Loss:\t{}\n'
          '===================================================================='
          '\n\n'
          .format(modelname, epoch, lr, codebookSize,
                  magic_c, rate, loss))
