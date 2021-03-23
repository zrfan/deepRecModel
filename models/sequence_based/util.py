# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
import sys
sys.path.append("../")
from models.base_estimator_model import BaseEstimatorModel
from models.ConfigParam import ConfigParam
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import GRUCell, RNNCell

class QAAttGRUCell(RNNCell):
    """
    Gated Recurrent Unit Cell

    """
    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        pass
