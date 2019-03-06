########################################################################
#                          Training Messages                           #
########################################################################


def train_message(model, data_file, epoch, lr, codebook_size,
                  rate, c_range=None):
    print('\n\n'
          '==================================================================='
          '\n\n\tTraining \'{}\' Model\n\n'
          'Data file:\t{}\n'
          'Epochs number:\t{}\n'
          'Learning rate:\t{}\n'
          'Codebook Size:\t{}\n'
          'Quantization Rate: {}\n'
          'Ci range:\t{}\n\n'
          'Architecture:\n'
          .format(model.modelname, data_file, epoch, lr, codebook_size,
                  rate, c_range))

    for layer in model.layers:
        print(layer)

    print('\n\n'
          '===================================================================')


def train_iteration(model, corr_epoch, epoch, batch_idx, data, train_loader, loss,
                    c_scope=None):
    if c_scope is None:
        print('Training \'{}\' Model:\tEpoch: {}/{} [{}/{} ({:.0f}%)]\t'
              'Linear Loss: {:.6f}'
              .format(model.modelname, corr_epoch + 1, epoch, batch_idx * len(data),
                      len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss))
    else:
        print('Training \'{}\' Model:\tEpoch: {}/{} [{}/{} ({:.0f}%)]\t'
              'Linear Loss: {:.6f}\tCi: {}'
              .format(model.modelname, corr_epoch + 1, epoch, batch_idx * len(data),
                      len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss,
                      1))
                      # model.quantization_layer.c))


########################################################################
#                          Testing Messages                            #
########################################################################


def test_message(model_name):
    print('\n\n'
          '===================================================================='
          '\n\tTesting \'{}\' Model\n'
          '===================================================================='
          '\n\n'
          .format(model_name))


def test_iteration(modelname, batch_idx, data, test_loader):
    print('Testing \'{}\' Model:\t[{}/{} ({:.0f}%)]'
          .format(modelname, batch_idx * len(data),
                  len(test_loader.dataset),
                  100. * batch_idx / len(test_loader)))


def test_results(modelname, epoch, lr, codebook_size, rate, loss, magic_c=None):
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
          .format(modelname, epoch, lr, codebook_size,
                  magic_c, rate, loss))
