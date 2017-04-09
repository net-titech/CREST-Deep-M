#!/usr/bin/env bash
# Add the caffe build location

if [ -z ${CAFFE_ROOT+x} ]; then
  echo "export CAFFE_ROOT=/home/matthias/Dropbox/WorkingFiles/caffe" >> ~/.bashrc;
else
  echo "CAFFE_ROOT is set to '$CAFFE_ROOT'";
fi

source ~/.bashrc
