#!/usr/bin/env sh
# Origin: caffe/examples/mnist/create_mnist

CAFFE_HOME="/home/matthias/Dropbox/WorkingFiles/caffe"
DATA="../data/mnist"
BUILD="$CAFFE_HOME/build/examples/mnist"

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $DATA/mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $DATA/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
