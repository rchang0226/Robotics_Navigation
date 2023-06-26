import torch.nn as nn
import torch


class MyValue(nn.Module):
    def __init__(self, pretrained):
        super(MyValue, self).__init__()
        self.pretrained = pretrained

    def forward(self, depth_img, goal, ray, hist_action):
        return self.pretrained(depth_img, goal, ray, hist_action)
