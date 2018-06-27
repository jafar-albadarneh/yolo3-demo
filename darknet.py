from __future__ import division # to map division to _truediv_() which return float result

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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



