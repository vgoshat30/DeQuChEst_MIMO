################################################################################
###                   Training, Testing and Logging Data                     ###
################################################################################

DATA_MAT_FILE = 'data.mat'
TEST_LOG_MAT_FILE = 'tempTestLog.mat'

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
EPOCH_RANGE = [5, 8]
LR_RANGE = [0.2, 0.21]
M_RANGE = [5, 6, 7]
MAGIC_C_RANGE = [7, 8]
