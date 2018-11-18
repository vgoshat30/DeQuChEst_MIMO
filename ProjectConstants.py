################################################################################
###                   Training, Testing and Logging Data                     ###
################################################################################

DATA_MAT_FILE = ['data_SNR_4.mat', ]
TEST_LOG_MAT_FILE = 'testLog_NonLinear_SNR_4.mat'

################################################################################
###                          Models to Activate                              ###
################################################################################

# Only uncommented models will be trained and tested
modelsToActivate = [
    # 'Passing Gradient',
    'Soft to Hard Quantization'
]

################################################################################
###                       Neural Network constants                           ###
################################################################################

BATCH_SIZE = 8
EPOCH_RANGE = range(2, 20)
LR_RANGE = [0.1, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22,
            0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35]
M_RANGE = range(3, 21)
MAGIC_C_RANGE = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]


'''Instructions for using ARCHITECTURE constant:
Specify different combinations of NN architecture using the following rules:
-   The ARCHITECTURE must be specified as a list of lists, each list
    representing one architecture.
-   The order of the layers in each list will be their order in the module.
-   There are only three types of layer available:
    -   linear: specified using the string 'linear' and afterwards (in the next
        place in the list), the multiplication factor between the input
        dimention and the output dimention of the layer.
    -   quantization: specified using the string 'quantization' (nothing else)

        NOTE that the quantization layer MUST be between two linear layers!

    -   relu: specified using the string 'relu' (nothing else)
-   Disclaimer: Meant to be used with only one quantization layer and relu
    activation functions appearing between two linear layers. Good behaviour is
    promised only under those circumstances.
'''
ARCHITECTURE = [

    ['linear', 2,
     'relu',
     'linear', 0.5,
     'quantization',
     'linear', 2,
     'relu',
     'linear', 0.5],

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

    # ['linear', 2,
    #  'linear', 2,
    #  'linear', 2,
    #  'relu',
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'linear', 0.5,
    #  'quantization',
    #  'linear', 2,
    #  'linear', 2,
    #  'linear', 2,
    #  'relu',
    #  'linear', 0.5,
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
