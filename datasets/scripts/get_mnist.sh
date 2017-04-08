#!/usr/bin/env sh
# Origin: caffe/data/mnist/get_mnist.sh
# Location: CREST-Deep-M/datasets/scripts/get_mnist.sh

DIR="$(cd "$(dirname "$0")"; pwd -P)"  # Get full path of the script folder
DATA_DIR="$DIR/../data"  # Location of the directory to save the data
tmp=${0#*_}  # Strip the prefix until underscore of this program's name
DATA_NAME=${tmp%.*}  # Strip the suffix until the dot of this prog's name
cd "$DATA_DIR"  # Change current director to the data directory

# Check if there is an mnist folder already
if [ ! -d "$DATA_NAME" ]; then
  mkdir "$DATA_NAME"
fi

cd "$DATA_NAME"  # Change directory to mnist folder

echo "Downloading to $DATA_NAME..."  # Downloading to mnist

# Download the data, extract, and delete the inputs
for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
