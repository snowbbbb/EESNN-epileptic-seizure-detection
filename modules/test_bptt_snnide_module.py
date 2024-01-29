import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
import copy
from modules.broyden import broyden, analyze_broyden
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


class SNNIDEModule(nn.Module):

    """ 
    SNN module with implicit differentiation on the equilibrium point in the inner 'Backward' class.
    """

    def __init__(self, snn_func, snn_func_copy):
        super(SNNIDEModule, self).__init__()
        self.snn_func = snn_func
        self.snn_func_copy = snn_func_copy

    def forward(self, z1, u, **kwargs):
        time_step = kwargs.get('time_step', 100)
        threshold = kwargs.get('threshold', 30)
        input_type = kwargs.get('input_type', 'constant')
        solver_type = kwargs.get('solver_type', 'broy')
        leaky = kwargs.get('leaky', None)
        get_all_rate = kwargs.get('get_all_rate', False)

        if get_all_rate:
            with torch.no_grad():
                if input_type != 'constant':
                    if len(u.size()) == 3:
                        u = u.permute(2, 0, 1)
                    else:
                        u = u.permute(4, 0, 1, 2, 3)
                r_list = self.snn_func.snn_forward(u, time_step, input_type=input_type, get_all_rate=True)
            return r_list

        if input_type != 'constant':
            if len(u.size()) == 3:
                u = u.permute(2, 0, 1)
            else:
                u = u.permute(4, 0, 1, 2, 3)

        z1_out = self.snn_func.snn_forward(u, time_step, input_type=input_type)

        return z1_out

