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
    'Passing Gradient',
    'Soft to Hard Quantization'
]

# ------------------------------

EPOCH_RANGE = [2, 5, 7]
LR_RANGE = [0.2, 0.1, 0.05, 0.01]
M_RANGE = [2, 3, 4, 5, 6, 7, 8]
