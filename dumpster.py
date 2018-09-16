import scipy.io as sio
import scipy.optimize as optim
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from random import random
from datetime import datetime
from torch.nn.parameter import Parameter
import torch
import sys
import time
from testLogger import *
import UserInterface as UI
from ProjectConstants import *

# theoryMatFile = sio.loadmat('theoreticalBounds.mat')
# theoryLoss = theoryMatFile['m_fCurves']
# theoryRate = theoryMatFile['v_fRate']
#
# log = createMatFile('testLog.mat', 'tanh', theoryRate, theoryLoss)

log = testlogger('testLog.mat')
# log.delete()
# log.content('all')
log.plot()
