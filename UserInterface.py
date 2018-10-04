
def readingDataFile(dataFile):
    print('\nReading data file \'{}\'...\n'.format(dataFile))


########################################################################
###                        Training Messages                         ###
########################################################################


def trainMessage(model, epoch, lr, codebookSize, layersDimentions,
                 magic_c=None):
    print('\n\n'
          '===================================================================='
          '\n\n\tTraining \'{}\' Model\n\n'
          'Epochs number:\t{}\n'
          'Learning rate:\t{}\n'
          'Codebook rize:\t{}\n'
          'MAGIC_C:\t{}\n'
          'Layer dim mlt:\t{}\n\n'
          'Resulting NN structure:\n\n'
          '-------------------------\n'
          '|Layer|   In    |  Out  |\n'
          '-------------------------\n'
          '|  1  | {}\t| {}\t|\n'
          '|  2  | {}\t| {}\t|\n'
          '|-------- RELU ---------|\n'
          '|  3  | {}\t| {}\t|\n'
          '|  4  | {}\t| {}\t|\n'
          '|----- Quantization ----|\n'
          '|  5  | {}\t| {}\t|\n'
          '|  6  | {}\t| {}\t|\n'
          '|-------- RELU ---------|\n'
          '|  7  | {}\t| {}\t|\n'
          '|  8  | {}\t| {}\t|\n'
          '-------------------------\n\n'
          '===================================================================='
          '\n\n'
          .format(model.modelname, epoch, lr, codebookSize, magic_c,
                  layersDimentions,
                  model.l1.weight.shape[1],
                  model.l1.weight.shape[0],
                  model.l2.weight.shape[1],
                  model.l2.weight.shape[0],
                  model.l3.weight.shape[1],
                  model.l3.weight.shape[0],
                  model.l4.weight.shape[1],
                  model.l4.weight.shape[0],
                  model.l5.weight.shape[1],
                  model.l5.weight.shape[0],
                  model.l6.weight.shape[1],
                  model.l6.weight.shape[0],
                  model.l7.weight.shape[1],
                  model.l7.weight.shape[0],
                  model.l8.weight.shape[1],
                  model.l8.weight.shape[0],))


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
                layersDimentions, classificationByWord=None, magic_c=None):
    print('\n\n'
          '===================================================================='
          '\n\n\tResults of \'{}\' Testing\n\n'
          '_________________________________\n'
          '|\tTraining Parameters\t|\n'
          '|\t\t\t\t|\n'
          '| - Epochs number:\t{}\t|\n'
          '| - Learning Rate:\t{}\t|\n'
          '| - Codebook Size:\t{}\t|\n'
          '| - Layers Dim:\t{}\t|\n'
          '| - MAGIC_C:\t\t{}\t|\n'
          '|_______________________________|\n\n'
          'Rate:\t{}\n'
          'Average Loss:\t{}\n'
          'Num. of classifications by word:\t{}\n\n'
          '===================================================================='
          '\n\n'
          .format(modelname, epoch, lr, codebookSize, layersDimentions,
                  magic_c, rate, loss, classificationByWord))
