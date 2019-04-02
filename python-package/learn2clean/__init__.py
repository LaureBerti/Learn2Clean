# coding: utf-8

__author__ = """Laure Berti-Equille"""
__email__ = 'laure.berti@ird.fr'
__version__ = '0.2.1'
__name__ = 'Learn2Clean'

import pandas as pd
import numpy as np
from .loading.reader import Reader
from .normalization.normalizer import Normalizer
from .feature_selection.feature_selector import Feature_selector
from .outlier_detection.outlier_detector import Outlier_detector
from .duplicate_detection.duplicate_detector import Duplicate_detector
from .consistency_checking.consistency_checker import Consistency_checker
from .imputation.imputer import Imputer
from .regression.regressor import Regressor
from .classification.classifier import Classifier
from .clustering.clusterer import Clusterer

__all__ = ['Reader', 'Normalizer', 'Feature_selector', 'Outlier_detector',
           'Duplicate_detector', 'Consistency_checker', 'Imputer',
           'Regressor', 'Classifier', 'Clusterer', ]

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
