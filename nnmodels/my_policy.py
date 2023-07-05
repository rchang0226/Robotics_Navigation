import torch.nn as nn
import torch
import numpy as np
from util.mathpy import *
import torchvision
import cv2
import math


class MyPolicy(nn.Module):
    def __init__(self, pretrained):
        super(MyPolicy, self).__init__()
        self.is_disc_action = False

        """ layers for inputs of depth_images """
        self.conv1 = pretrained.conv1
        self.conv2 = pretrained.conv2
        self.conv3 = pretrained.conv3
        self.fc_img = pretrained.fc_img

        """ layers for inputs of goals and rays """
        self.fc_ray = pretrained.fc_ray
        self.fc_action = pretrained.fc_action

        """ layers for inputs concatenated information """
        self.img_goal_ray1 = pretrained.img_goal_ray1
        # two dimensions of actions: upward and downward; turning
        self.img_goal_ray2 = pretrained.img_goal_ray2

        self.resnet = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.resnet.classifier = nn.Linear(576, 512)
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 96)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.action_log_std = pretrained.action_log_std

        # projection matrix
        self.p = np.array(
            [
                [1.19175, 0, 0, 0],
                [0, 1.58901, 0, 0],
                [0, 0, -1.00010, -0.10001],
                [0, 0, -1, 0],
            ]
        )

    def forward(self, color_img, depth_img, goal, ray, hist_action):
        # color_img = color_img.permute(0, 3, 1, 2)
        # color_img = torch.squeeze(color_img, 0)
        # color_img = torchvision.transforms.Resize((120, 160))(color_img)
        # color_img = self.relu(self.linear1(
        #     self.resnet(color_img.double())))
        # color_img = self.relu(self.linear2(color_img))
        # print(color_img.shape)
        # print(list(map(type, color_img)))
        color_img, depth, color, gold = color_img
        if "bottle" in color_img.pandas().xyxy[0]["name"].values:
            #     # cv2.imwrite("depth.png", depth)
            #     cv2.imwrite("color.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            print(color_img.pandas().xyxy[0]["name"][0])
            xmin = color_img.pandas().xyxy[0]["xmin"][0]
            xmax = color_img.pandas().xyxy[0]["xmax"][0]
            ymin = color_img.pandas().xyxy[0]["ymin"][0]
            ymax = color_img.pandas().xyxy[0]["ymax"][0]
            xmid = (xmin + xmax) / 2
            xmid = int(xmid / 2)
            ymid = (ymin + ymax) / 2
            ymid = int(ymid / 2)
            # drawn = cv2.circle(depth, (xmid, ymid), radius=2, color=150, thickness=1)
            # cv2.imwrite("drawn.png", drawn)
            # color_img.save()
            # converted = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            # drawn = cv2.circle(
            #     converted, (xmid, ymid), radius=2, color=(200, 50, 0), thickness=1
            # )
            # cv2.imwrite("color.png", drawn)
            deep = depth[ymid, xmid]

            # vfov = 64 ish
            # hfov = 80
            size = (xmax - xmin) * (ymax - ymin) / 4
            depth = 1 / size * 1200
            vdeg = (64 - ymid) / 2
            horizontal = depth * abs(math.cos(math.radians(vdeg)))
            vertical = depth * math.sin(math.radians(vdeg))
            hdeg = (80 - xmid) / 2
            print((horizontal - gold[0], hdeg - gold[2], vertical - gold[1]))

            # xmid = xmid * 3
            # ymid = ymid * 2.8
            # depth = 4
            # z = depth - self.p[2, 3]
            # x = (xmid * depth - self.p[0, 3] - self.p[0, 2] * z) / self.p[0, 0]
            # y = (ymid * depth - self.p[1, 3] - self.p[1, 2] * z) / self.p[1, 1]
            # print((x, y, z))

        color_img = torch.rand(1, 96)

        depth_img = self.relu(self.conv1(depth_img))
        depth_img = self.relu(self.conv2(depth_img))
        depth_img = self.relu(self.conv3(depth_img))
        depth_img = depth_img.view(depth_img.size(0), -1)
        depth_img = self.relu(self.fc_img(depth_img))

        ray = ray.view(ray.size(0), -1)
        ray = self.relu(self.fc_ray(ray))

        hist_action = hist_action.view(hist_action.size(0), -1)
        hist_action = self.relu(self.fc_action(hist_action))

        img_goal_ray_aciton = torch.cat((depth_img, color_img, ray, hist_action), 1)
        img_goal_ray_aciton = self.relu(self.img_goal_ray1(img_goal_ray_aciton))
        action_mean = self.tanh(self.img_goal_ray2(img_goal_ray_aciton))

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, color_img, depth_img, goal, ray, hist_action):
        action_mean, _, action_std = self.forward(
            color_img, depth_img, goal, ray, hist_action
        )
        # print "action:", action_mean, action_std
        action = torch.clamp(torch.normal(action_mean, action_std), -1, 1)
        # print action, "\n\n\n"
        return action

    def get_log_prob(self, color_img, depth_img, goal, ray, hist_action, actions):
        action_mean, action_log_std, action_std = self.forward(
            color_img, depth_img, goal, ray, hist_action
        )
        return normal_log_density(actions, action_mean, action_log_std, action_std)
