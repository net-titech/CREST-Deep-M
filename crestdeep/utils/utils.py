"""
[utils] package provides I/O and miscancellous functions
"""

try:
    caffe_root = os.environ["CAFFE_ROOT"]
    sys.path.insert(0, caffe_root + '/python')
    import caffe
    cafe.set_mode_cpu() # We are not training models here
except KeyError:
    print("CAFFE_ROOT is not found. Some function won't be available.")

def load_weights(caffe_model):
    """Extract weights."""

def read_caffe(caffe_model, prototxt):
    """Load a caffe model as a python object.

    Keyword arguments:
    caffe_model -- string contains path to the caffe weight file
    """
    net = caffe.Net(prototxt, caffe_model, caffe.TEST)
    layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x,
                    net.params.keys())
