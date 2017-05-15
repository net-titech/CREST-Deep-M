"""
[utils] package provides I/O and miscancellous functions
"""
import numpy as np
import pickle as pkl
import os
import sys
try:
    caffe_root = os.environ["CAFFE_ROOT"]
    sys.path.insert(0, caffe_root + '/python')
    import caffe
    cafe.set_mode_cpu() # We are not training models here
except KeyError:
    print("CAFFE_ROOT is not found. Some function won't be available.")

### CAFFE

def load_weights(caffe_model, prototxt):
    """Extract weights as numpy arrays.

    Keyword arguments:
    caffe_model -- string contains path to the caffe file (e.g. net.caffemodel)
    prototxt -- string contains path to the prototxt file (e.g. net.prototxt)

    Return: A tuple (weights, biases)
    weights -- dictionary {layer name -> numpy array}
    biases -- dictionary {layer name -> numpy array}
    layers -- list of strings for parameterized layer name
    """
    net, layers = read_caffe(caffe_model, prototxt)
    weights = {}
    biases = {}
    for layer in layers:
        weights[layer] = net.params[layer][0].data
        biases[layer] = net.params[layer][1].data
    return weights, biases, layers

def read_caffe(caffe_model, prototxt):
    """Load a caffe model as a python object.

    Keyword arguments:
    caffe_model -- string contains path to the caffe file (e.g. net.caffemodel)
    prototxt -- string contains path to the prototxt file (e.g. net.prototxt)

    Return: A tuple (net, layers)
    net -- python object contains the caffe model
    layers -- list of strings for parameterized layer name
    """
    net = caffe.Net(prototxt, caffe_model, caffe.TEST)
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x,
                    net.params.keys())
    return net, layers

###
