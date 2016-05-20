# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import bempp.api as bem
from numpy.linalg import norm
from bempp.api import GridFunction as gf
from Login import make_callback


def fs(grid, **kwargs):
    return bem.function_space(grid, "P", 1, **kwargs)

def k_define():
    return 1

def proc_count():
    return 32

global K0
K0 = k_define()

def tolerance():
    return 1E-5

bem.global_parameters.assembly.potential_operator_assembly_type = 'dense'

