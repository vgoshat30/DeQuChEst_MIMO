################################################################################
#                     Training, Testing and Logging Data                       #
################################################################################

DATA_MAT_FILE = ['dataGamma.mat', ]
TEST_LOG_MAT_FILE = 'testLogGamma.mat'
EPOCH_LOSS_LOG_MAT_FILE = 'epochLog.mat'

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
LR_RANGE = [0.005, ]
M_RANGE = [40, ]
C_INCREMENT_RANGE = [[50, 50], ]
C_STEPS_AMOUNT = 1
A_RANGE = [0.1, ]
FORCE_QUANTIZATION_DIM = False

ARCHITECTURE = [

    ['linear', 0.1
     ],

    # ['linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.5,
    #  'quantization',
    #  'linear', 2,
    #  'relu',
    #  'linear', 2,
    #  'relu',
    #  'linear', 2,
    #  'relu',
    #  'linear', 2,
    #  'relu',
    #  'linear', 2,
    #  'relu',
    #  'linear', 2,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.9,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.8,
    #  'relu',
    #  'linear', 0.7,
    #  'relu',
    #  'linear', 0.7
    #  ]
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
