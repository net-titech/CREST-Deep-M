"""
[decoder] module provides functions to encode a pruned caffe model.
The pruned model is acquired through the training process.
"""

def decode_index(encoded_ind, org_size=None, bits=4):
    """Decode nonzero indices"""
    if org_size is None:
        print("Original size must be specified.")
        sys.exit(1)
    decode = np.zeros(org_size, dtype=np.uint32)
    if (bits == 4):
        decode[np.arange(0,org_size,2)] = encoded_ind % 2**bits
        decode[np.arange(1,org_size,2)] = encoded_ind / 2**bits
    decode = np.cumsum(decode+1) - 1
    return np.asarray(decode, dtype=np.uint32)
