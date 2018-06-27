import torch.nn as nn

class DetectionLayer(nn.Module):
    """
    holds the anchors used to detect bounding boxes.
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors