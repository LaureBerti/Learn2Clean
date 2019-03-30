#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>


__author__ = """Laure Berti-Equille"""
__email__ = 'laure.berti@ird.fr'
__version__ = '0.2.1'

import pandas as pd
import numpy as np
from .loading import *
from .normalization import *
from .feature_selection import *
from .outlier_detection import *
from .duplicate_detection import *
from .consistency_checking import *
from .imputation import *
from .regression import *
from .classification import *
from .clustering import *
from .qlearning import *


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=ImportWarning)

np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
