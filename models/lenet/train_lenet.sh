#!/usr/bin/env sh
set -e  # Exit immediately if a command exits with a non-zero status

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

$CAFFE_ROOT/build/tools/caffe train --solver=$DIR/lenet_solver.prototxt $@
