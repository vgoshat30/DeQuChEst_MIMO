""" Constants file """
import math

################################################################################
###                       Neural Network constants                           ###
################################################################################

BATCH_SIZE = 8
INPUT_DIMENSION = 240
OUTPUT_DIMENSION = 80
CODEBOOK_LR = 0.1

# Only uncommented models will be trained and tested
modelsToActivate = [
    # 'Passing Gradient',
    'Soft to Hard Quantization'
]

# ------------------------------

EPOCH_RANGE = [5, 8]
LR_RANGE = [0.2, 0.21]
M_RANGE = [17, 18, 19, 20]
INITIAL_C_FACTOR = 5
C_DECAY = 0.99
C_BOOST_FREQUENCY = 300

MAGIC_C = 8
