"""
[encoder] package provides functions to encode a pruned caffe model.
The pruned model is acquired through the training process.
"""
from crestdeep.utils import extract_weights_from_caffe
from sklearn.cluster import KMeans
import numpy as np
import sys
import os
