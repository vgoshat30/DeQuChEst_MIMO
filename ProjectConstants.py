################################################################################
#                     Training, Testing and Logging Data                       #
################################################################################

DATA_MAT_FILE = ['data_SNR4.mat', 'data_SNR_1_10.mat']
TEST_LOG_MAT_FILE = 'testLog_SNR4_bestScenario.mat'#'testLog_bestScenario_hanukkah.mat'

################################################################################
#                            Models to Activate                                #
################################################################################

# Only uncommented models will be trained and tested
modelsToActivate = [
    # 'Passing Gradient',
    'Soft to Hard Quantization'
]

################################################################################
#                         Neural Network constants                             #
################################################################################

BATCH_SIZE = 8
EPOCH_RANGE = [10, ]
LR_RANGE = [0.2, ]
M_RANGE = range(49, 100)
C_INCREMENT_RANGE = [[5, 5],
                     [8, 8]]
C_STEPS_AMOUNT = 1

ARCHITECTURE = [
    ['linear', 2,
     'linear', 0.5,
     'linear', 0.5,
     'quantization',
     'linear', 2,
     'linear', 0.5],

    # ['linear', 2,
    #  'linear', 2,
    #  'quantization',
    #  'linear', 0.5,
    #  'linear', 0.5],

    # ['linear', 2,
    #  'linear', 2,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'quantization',
    #  'linear', 2,
    #  'linear', 2,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5],

    # ['linear', 4,
    #  'linear', 4,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'quantization',
    #  'linear', 4,
    #  'linear', 4,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5],
    # ['linear', 4,
    #  'linear', 4,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'quantization',
    #  'linear', 4,
    #  'linear', 4,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5],

    # ['linear', 10,
    #  'linear', 10,
    #  'relu',
    #  'linear', 0.1,
    #  'linear', 0.1,
    #  'quantization',
    #  'linear', 10,
    #  'linear', 10,
    #  'relu',
    #  'linear', 0.1,
    #  'linear', 0.1]
]

'''Instructions for using ARCHITECTURE constant:
Specify different combinations of NN architecture using the following rules:

-   The ARCHITECTURE must be specified as a list of lists, each list
    representing one architecture.
-   The order of the layers in each list will be their order in the module.
-   There are only three types of layer available:

    -   linear: specified using the string 'linear' and afterwards (in the next
        place in the list), the multiplication factor between the input
        dimension and the output dimension of the layer.
    -   quantization: specified using the string 'quantization' (nothing else)

        NOTE that the quantization layer MUST be between two linear layers!

    -   relu: specified using the string 'relu' (nothing else)
-   Disclaimer: Meant to be used with only one quantization layer and relu
    activation functions appearing between two linear layers. Good behaviour is
    promised only under those circumstances.
'''
