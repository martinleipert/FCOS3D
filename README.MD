# FCOS 3D 

Implementation of the FCOS detector for three-dimensional data. 

## Usage

For Inference just feed the network with an imageList.

For training the network, targets needs to be an appropriately filled boxlist. 
See the minimal example in test directory.

## Credits 

The code of the FCOS implementation is based on the code by rosinality:
https://github.com/rosinality/fcos-pytorch/blob/master/model.py 

The code ImageList and BoxList is based on the source of detectron.
