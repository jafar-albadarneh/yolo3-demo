from __future__ import division # to map division to _truediv_() which return float result

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from empty_layer import EmptyLayer
from detection_layer import DetectionLayer

#### Parse YOLO CFG ####
def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Return a list of blocks. Each block describes a 
    block in the neural network. Block is represented 
    as  a dictionary in the list that are accessed via
    key-value pairs
    """
    ######## PREPROCESSING OF CFG #######
    file = open(cfgfile, 'r')
    # store lines in a list
    lines = file.read().split('\n')
    # get rid of the empty lines
    lines = [x for x in lines if len(x) > 0]
    # get rid of the comments
    lines = [x for x in lines if x[0] != '#']
    # get rid of the fringe whitespaces
    lines = [x.rstrip() for x in lines]

    # maintain a list of blocks
    block = {} #dict
    blocks = [] #list

    for line in lines:
        if line[0] == "[":  # a mark for new block
            if len(block) != 0: # check if same block
                blocks.append(block) # add it to the blocks list
                block = {} # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block) # catch up the last block
    return blocks


#### Network Construction ####
# there exists 5 types of layers as explained in:
# https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/

# pytorch provides pre-built layers for types:
# - Convolutional and - Upsample
# for the other modules (types):
# - Shortcut 
# - Yolo 
# - Route: (possibly concatenated) brings feature map
# from previous layer
# we should create our own
# by extending torch nn.Module class

def create_modules(blocks):
    """
    takes a list of blocks returned by the `parse_cfg`
    function
    """
    # Get network hyperparams from first block
    # input, pre-processing, learning_rate .. etc
    net_info = blocks[0]
    # list containing nn.modules
    module_list = nn.ModuleList()
    prev_filters = 3 # as the image has 3 channels RGB
    output_filters = []
    print "processing {} blocks".format(len(blocks))
    ### Create NN Modules from blocks
    for index, x in enumerate(blocks[1:]):
        # define a sequential module to execute
        # a number of nn.Module objects, as a block
        # may contain more than one layer.
        # For ex, a block of type convolutional has a
        # batch norm layer as well as leakyReLU activation
        # layer in addition to a conv layer
        module = nn.Sequential()

        # check the type of the block
        # create a new module for the block
        # append to module_list
        if(x["type"] == "convolutional"):
            # Get Info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)/2
            else:
                pad = 0

            # Add the covolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check for activation layer
            # either Linear or Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace= True)
                module.add_module("leaky_{0}".format(index), activn)
        
        ## if it's an Upsample layer    
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode= "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        ## Check if is a Route Layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #start of route
            start = int(x["layers"][0])
            # end, if there exist one.
            try:
                end = int(x["layers"][0])
            except:
                end = 0
            
            #positive annotiation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        ### if a shortcut layer
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        ### if a yolo layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            anchors = x["anchors"].split(",")
            mask = [int(x) for x in mask]

            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        ### Last thing
        module_list.append(module)
        # update filters
        prev_filters = filters
        output_filters.append(filters)

    # return a tuple with net_info and module_list
    return (net_info, module_list)

#### Run the network ###
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))

