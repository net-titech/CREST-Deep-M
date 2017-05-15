"""
[encoder] module provides functions to encode a pruned caffe model.
The pruned model is acquired through the training process.
"""
from crestdeep import utils
from sklearn.cluster import KMeans
import numpy as np
import sys
import os

class CodecMeta:
    """Metadata for encoder."""
    def __init__(self, conv_bits=8, fc_bits=4, ind_bits=4,
                 nz_bits=32, bias_bits=32):
        """Store bit length information for encryption."""
        self.conv = conv_bits
        self.fc = fc_bits
        self.ind = ind_bits
        self.nz = nz_bits
        self.bias = bias_bits

    def get_nval(self, layer_name):
        """Get number of values each pruned layer is supposed to have."""
        if 'conv' in layer_name:
            return 2**self.conv
        elif 'fc' in layer_name or 'ip' in layer_name:
            return 2**self.fc
        else:
            print("WARNING: " + layer_name + " num val is not implemented.")
        return 256 # Return a default value

    def get_dtype(self, layer_name):
        """Get number of values each pruned layer is supposed to have."""
        if 'conv' in layer_name:
            return self.dtype(self.conv)
        elif 'fc' in layer_name or 'ip' in layer_name:
            return self.dtype(self.fc)
        elif 'bias' in layer_name:
            return np.float32
        elif 'nz' in layer_name:
            return self.dtype(self.nz)
        elif 'ind' in layer_name:
            return self.dtype(self.ind)
        else:
            print("WARNING: " + layer_name + " dtype is not implemented.")
        return np.uint16 # Return a default value

    def dtype(self, bits):
        if 4 == bits or 8 == bits:
            return np.uint8
        elif 16 == bits:
            return np.uint16
        elif 32 == bits:
            return np.uint32
        elif 64 == bits:
            return np.uint64
        else:
            print("WARNING: Unknown bit length.")
            return np.uint16 # Return default data type

default_codec = CodecMeta()

def encode_index(nz_index, bits=4):
    """Encode nonzero indices using 4-bit"""
    max_val = 2**bits
    if bits == 4 or bits == 8:
        data_type = np.uint8
    elif bits == 16:
        data_type = np.uint16
    else:
        print("Unimplemented index encoding with " + str(bits) + " bits.")
        sys.exit()
    code = np.zeros_like(nz_index, dtype=np.uint32)
    adv = 0
    # Encode with relative to array index
    for i, val in enumerate(nz_index):
        cur_i = i + adv
        code[i] = val - cur_i
        if (val - cur_i != 0):
            adv += val - cur_i
    # Check if there is overflow
    if (code.max() >= max_val):
        print("Overflow index codebook. Unimplemented handling.")
        sys.exit()
    # Special case of 4-bit encoding
    if (bits == 4):
        return encode_values(code, bits=4)
    return np.asarray(code, dtype=data_type)

def encode_values(vals, bits=4):
    """Encode two 4 bits values into one 8 bits value.
    TODO: encode_values function is a quick fix. Generalize!

    Keyword arguments:
    vals -- 8-bit value array.
    bits -- bit length of output array elements.

    Return:
    new_vals -- arrays of 8-bits values containing two 4-bits each.

    Usage:
    >>> encode_values(np.array([1,2,3,4], dtype=np.uint8))
    array([33, 67], dtype=uint8)
    >>> encode_values(np.array([1, 5, 2, 5, 7, 3, 5, 3], dtype=np.uint8))
    array([81, 82, 55, 53], dtype=uint8)
    >>> encode_values(array([1, 5, 6, 5, 7, 6, 7, 5, 4], dtype=np.uint8))
    array([ 81,  86, 103,  87,   4], dtype=uint8)
    """
    if bits != 4: # Only work for 4 bit
        return vals
    if vals.max() > 15:
        print("Cannot encode given array as 4-bit array.")
        sys.exit()
    if vals.size % 2 != 0:
        vals = np.resize(vals, vals.size+1)
        vals[-1] = 0 # make sure the last value is 0
    even_idx = np.arange(0, vals.size, 2)
    odd_idx = np.arange(1, vals.size, 2)
    new_vals = vals[even_idx] + vals[odd_idx] * 16
    return np.asarray(new_vals, dtype=np.uint8)

def encode_pruned(caffe_model, prototxt, codec=default_codec):
    """Encode the caffe_model in sparse quantized format.

    Keyword arguments:
    caffe_model -- string path to pruned caffe file (e.g. net_pruned.caffemodel)
    prototxt -- string path to model description (e.g. net.caffemodel)
    codec -- CodecMeta object for encoder's bit length information

    Return:
    codebook -- dictionary {layer name -> array of unique values}
    encoded -- dictionary {layer name -> array of indices in codebook, ind arr}
    biases -- dictionary {layer name -> biases}
    layers -- list of layer name strings
    """
    weights, biases, layers = utils.load_weights (caffe_model, prototxt)
    codebook = {}
    encoded = {}
    for layer in layers:
        w = weights[layer].reshape(1,-1)
        nz_ind = np.nonzero(w)[0]
        w = w[nz_ind]
        nz_ind = encode_index(nz_ind, bits=codec.ind)
        values, encode = np.unique(w, return_inverse=True)
        # TODO: Generalize encoding fully connected as 4-bit
        if 'fc' in layer and codec.fc == 4:
            encode = encode_values(encode, bits=4)
        if values.size > codec.get_nval(layer):
            print("ERROR: " + layer + " has too many values.")
            sys.exit()
        codebook[layer] = values
        encoded[layer] = (np.asarray(encode, dtype=codec.get_dtype(layer)),
                          nz_ind)
    return codebook, encoded, biases, layers

def pack_encoded(codebook, encoded, biases, layers, codec=default_codec,
                 filename="net.bin"):
    """Pack the encoded models as Han structure and write to disk as
    a binary file named `filename`.

    Keyword arguments:
    codebook -- dictionary {layer name -> array of unique values}
    encoded -- dictionary {layer name -> (array of indices in codebook, nz_ind)}
    biases -- dictionary {layer name -> biases}
    layers -- list of layer name strings
    filename -- string path to binary file will be written to disk

    Return: None
    """
    # Open to write file as byte
    fout = open(filename, 'wb')
    # Number of nonzero for each layer
    nz_num = [encoded[layer][0].size for layer in layers]
    nz_num = np.asarray(nz_num, dtype=codec.get_dtype('nz'))
    nz_num.tofile(fout, format=codec.get_dtype('nz'))
    # For each layer save codebook, bias, values and indices
    for layer in layers:
        codebook[layer].tofile(fout, format=codec.get_dtype('bias'))
        biases[layer].tofile(fout, format=codec.get_dtype('bias'))
        spm_stream, ind_stream = encoded[layer]
        spm_stream.tofile(fout, format=codec.get_dtype(layer))
        ind_stream.tofile(fout, format=codec.get_dtype('ind'))
    # Save the file
    fout.close()
