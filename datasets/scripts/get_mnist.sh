#!/usr/bin/env sh
# Origin: caffe/data/mnist/get_mnist.sh
# Location: CREST-Deep-M/datasets/scripts/get_mnist.sh

DIR="$(cd "$(dirname "$0")"; pwd -P)"
DATA_DIR="$DIR/../data"
tmp=${0#*_}
DATA_NAME=${tmp%.*}
cd "$DATA_DIR"

if [ ! -d "$DATA_NAME" ]; then
  mkdir "$DATA_NAME"
fi

cd "$DATA_NAME"

echo "Downloading to $DATA_NAME..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
