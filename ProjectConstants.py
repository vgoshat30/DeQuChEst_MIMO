################################################################################
###                   Training, Testing and Logging Data                     ###
################################################################################

DATA_MAT_FILE = 'data_a30_u12.mat'
TEST_LOG_MAT_FILE = 'testLog_a30_u12.mat'

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
EPOCH_RANGE = [10, 12]
LR_RANGE = [0.2]
M_RANGE = [8, ]
MAGIC_C_RANGE = [5, 6, 8, 10]
