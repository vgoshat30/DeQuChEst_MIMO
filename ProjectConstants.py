################################################################################
###                   Training, Testing and Logging Data                     ###
################################################################################

DATA_MAT_FILE = 'newdata_test.mat'
TEST_LOG_MAT_FILE = 'newLog_test.mat'

################################################################################
###                          Models to Activate                              ###
################################################################################

# Only uncommented models will be trained and tested
modelsToActivate = [
    'Passing Gradient',
    'Soft to Hard Quantization'
]

################################################################################
###                       Neural Network constants                           ###
################################################################################

BATCH_SIZE = 8
EPOCH_RANGE = [10, ]
LR_RANGE = [0.2]
M_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
MAGIC_C_RANGE = [5, ]

LAYERS_DIM_MULTIPLIERS_RANGE = [[10, 8, 5, 10, 5, 2],
                                [12, 10, 4, 15, 10, 5],
                                [8, 5, 2, 8, 5, 2]]
