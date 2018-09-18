########################################################################
###                        Training Messages                         ###
########################################################################


def trainMessage(model, epoch, lr, codebookSize, magic_c=None):
    print('\n\n'
          '===================================================================='
          '\n\n\tTraining \'{}\' Model\n\n'
          'Epochs number:\t{}\n'
          'Learning Rate:\t{}\n'
          'Codebook Size:\t{}\n'
          'MAGIC_C:\t{}\n\n'
          '===================================================================='
          '\n\n'
          .format(model, epoch, lr, codebookSize, magic_c))


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


def testMessage(model):
    print('\n\n'
          '===================================================================='
          '\n\tTesting \'{}\' Model\n'
          '===================================================================='
          '\n\n'
          .format(model))


def testIteration(modelname, batch_idx, data, testLoader):
    print('Testing \'{}\' Model:\t[{}/{} ({:.0f}%)]'
          .format(modelname, batch_idx * len(data),
                  len(testLoader.dataset),
                  100. * batch_idx / len(testLoader)))


def testResults(modelname, epoch, lr, codebookSize, rate, loss,
                classificationByWord=None, magic_c=None):
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
          'Num. of clasifications by word:\t{}\n\n'
          '===================================================================='
          '\n\n'
          .format(modelname, epoch, lr, codebookSize, magic_c, rate, loss,
                  classificationByWord))
