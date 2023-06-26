import torch.nn as nn
import torch


class NNValue(nn.Module):
    def __init__(self, pretrained):
        super(NNValue, self).__init__()
        self.pretrained = pretrained

    def forward(self, depth_img, goal, ray, hist_action):
        return self.pretrained(depth_img, goal, ray, hist_action)
