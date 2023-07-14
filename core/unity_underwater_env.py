import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import time
import uuid
import random
import os
from util import *

from typing import List
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from torchvision import transforms
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.midas_net_custom import MidasNet_small
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from nnmodels.funie import GeneratorFunieGAN
from PIL import Image
from torchvision.utils import make_grid
import random

DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
DIM_GOAL = 3
DIM_ACTION = 2
BITS = 2


class DPT_depth:
    def __init__(
        self,
        device,
        model_type="dpt_large",
        model_path=os.path.abspath("./") + "/DPT/weights/dpt_large-midas-2f21e586.pt",
        optimize=True,
    ):
        self.optimize = optimize
        self.THRESHOLD = torch.tensor(np.finfo("float").eps).to(device)

        # load network
        if model_type == "dpt_large":  # DPT-Large
            self.net_w = self.net_h = 384
            resize_mode = "minimal"
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )

        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            self.net_w = self.net_h = 384
            resize_mode = "minimal"
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.normalization = NormalizeImage(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        elif model_type == "dpt_hybrid_kitti":
            self.net_w = 1216
            self.net_h = 352
            resize_mode = "minimal"

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.normalization = NormalizeImage(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )

        elif model_type == "dpt_hybrid_nyu":
            self.net_w = 640
            self.net_h = 480
            resize_mode = "minimal"

            self.model = DPTDepthModel(
                path=model_path,
                scale=0.000305,
                shift=0.1378,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            self.normalization = NormalizeImage(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )

        elif model_type == "midas_v21":  # Convolutional model
            self.net_w = self.net_h = 384

            resize_mode = "upper_bound"
            self.model = MidasNet_large(model_path, non_negative=True)
            self.normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif model_type == "midas_v21_small":
            self.net_w = self.net_h = 256
            resize_mode = "upper_bound"
            self.model = MidasNet_small(model_path, non_negative=True)

            self.normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

        # select device
        self.device = device

        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )

        self.model.eval()

        if optimize == True and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()

        self.model.to(self.device)

    def run(self, rgb_img):
        img_input = self.transform({"image": rgb_img})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(0),
                    size=(DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            depth_min = prediction.min()
            depth_max = prediction.max()
            if depth_max - depth_min > self.THRESHOLD:
                prediction = (prediction - depth_min) / (depth_max - depth_min)
            else:
                prediction = np.zeros(prediction.shape, dtype=prediction.dtype)

            # plt.imshow(np.uint16(prediction * 65536))
            # plt.show()

        # cv2.imwrite("depth.png", ((prediction*65536).astype("uint16")), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return prediction


class PosChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.goal_depthfromwater = msg.read_float32_list()

    def goal_depthfromwater_info(self):
        return self.goal_depthfromwater

    def assign_testpos_visibility(self, data: List[float]) -> None:
        msg = OutgoingMessage()
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)


visibility_constant = 1
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
gan = GeneratorFunieGAN()
gan.load_state_dict(torch.load("funie_generator.pth"))
if torch.cuda.is_available():
    gan.cuda()
gan.eval()

img_width, img_height, channels = 256, 256, 3
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)


class Underwater_navigation:
    def __init__(
        self,
        depth_prediction_model,
        adaptation,
        randomization,
        rank,
        HIST,
        start_goal_pos=None,
        training=True,
    ):
        if adaptation and not randomization:
            raise Exception(
                "Adaptation should be used with domain randomization during training"
            )
        self.adaptation = adaptation
        self.randomization = randomization
        self.HIST = HIST
        self.training = training
        self.twist_range = 30  # degree
        self.vertical_range = 0.1
        self.action_space = spaces.Box(
            np.array([-self.twist_range, -self.vertical_range]).astype(np.float32),
            np.array([self.twist_range, self.vertical_range]).astype(np.float32),
        )
        self.observation_space_img_depth = (
            self.HIST,
            DEPTH_IMAGE_HEIGHT,
            DEPTH_IMAGE_WIDTH,
        )
        self.observation_space_goal = (self.HIST, DIM_GOAL)
        self.observation_space_ray = (self.HIST, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_info = PosChannel()
        config_channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(
            os.path.abspath("./") + "/underwater_env/water",
            side_channels=[config_channel, self.pos_info],
            worker_id=rank,
            base_port=5005,
        )

        if self.randomization == True:
            if self.training == False:
                visibility = 3 * (13 ** random.uniform(0, 1))
                if start_goal_pos == None:
                    raise AssertionError
                self.start_goal_pos = start_goal_pos
                self.pos_info.assign_testpos_visibility(
                    self.start_goal_pos + [visibility]
                )
        else:
            if self.training == False:
                if start_goal_pos == None:
                    raise AssertionError
                self.start_goal_pos = start_goal_pos
                visibility = 3 * (13**visibility_constant)
                self.pos_info.assign_testpos_visibility(
                    self.start_goal_pos + [visibility]
                )

        config_channel.set_configuration_parameters(
            time_scale=10, capture_frame_rate=100
        )
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        if depth_prediction_model == "dpt":
            self.dpt = DPT_depth(
                self.device,
                model_type="dpt_large",
                model_path=os.path.abspath("./")
                + "/DPT/weights/dpt_large-midas-2f21e586.pt",
            )
        elif depth_prediction_model == "midas":
            self.dpt = DPT_depth(
                self.device,
                model_type="midas_v21_small",
                model_path=os.path.abspath("./")
                + "/DPT/weights/midas_v21_small-70d6b9c8.pt",
            )

        self.total_steps = 0
        self.total_correct = 0

    def reset(self):
        self.step_count = 0
        if self.randomization == True:
            if self.adaptation == True:
                visibility_para = random.uniform(-1, 1)
                visibility = 3 * (13 ** ((visibility_para + 1) / 2))
                self.visibility_para_Gaussian = np.clip(
                    np.random.normal(visibility_para, 0.02, 1), -1, 1
                )
                if self.training == False:
                    self.pos_info.assign_testpos_visibility(
                        self.start_goal_pos + [visibility]
                    )
                else:
                    self.pos_info.assign_testpos_visibility([0] * 9 + [visibility])
            else:
                visibility_para = random.uniform(-1, 1)
                visibility = 3 * (13 ** ((visibility_para + 1) / 2))
                self.visibility_para_Gaussian = np.array([0])
                if self.training == False:
                    self.pos_info.assign_testpos_visibility(
                        self.start_goal_pos + [visibility]
                    )
                else:
                    self.pos_info.assign_testpos_visibility([0] * 9 + [visibility])
        else:
            visibility = 3 * (13**visibility_constant)
            # visibility = 1000 # testing in the new shader
            self.visibility_para_Gaussian = np.array([0])
            if self.training == False:
                self.pos_info.assign_testpos_visibility(
                    self.start_goal_pos + [visibility]
                )
            else:
                self.pos_info.assign_testpos_visibility([0] * 9 + [visibility])

        # waiting for the initialization
        self.env.reset()
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())
        if self.training == False:
            my_open = open(
                os.path.join(assets_dir(), "learned_models/test_pos.txt"), "a"
            )
            data = [
                str(obs_goal_depthfromwater[4]),
                " ",
                str(obs_goal_depthfromwater[5]),
                " ",
                str(obs_goal_depthfromwater[3]),
                "\n",
            ]
            for element in data:
                my_open.write(element)
            my_open.close()
        obs_img_ray, _, done, _ = self.env.step([0, 0])

        # observations per frame
        obs_preddepth = self.dpt.run(obs_img_ray[0] ** 0.45)
        obs_ray = np.array(
            [
                np.min(
                    [
                        obs_img_ray[1][1],
                        obs_img_ray[1][3],
                        obs_img_ray[1][5],
                        obs_img_ray[1][33],
                        obs_img_ray[1][35],
                    ]
                )
                * 8
                * 0.5
            ]
        )

        # obs_ray = np.array([0])
        obs_goal_depthfromwater = np.array(self.pos_info.goal_depthfromwater_info())

        self.prevPos = (
            obs_goal_depthfromwater[4],
            obs_goal_depthfromwater[3],
            obs_goal_depthfromwater[5],
        )

        if self.training == False:
            my_open = open(
                os.path.join(assets_dir(), "learned_models/test_pos.txt"), "a"
            )
            data = [
                str(obs_goal_depthfromwater[4]),
                " ",
                str(obs_goal_depthfromwater[5]),
                " ",
                str(obs_goal_depthfromwater[3]),
                "\n",
            ]
            for element in data:
                my_open.write(element)
            my_open.close()

        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        # print(np.shape(obs_preddepth), np.shape(obs_goal_depthfromwater[:3]), np.shape(obs_ray), "\n\n\n")
        self.obs_preddepths = np.array([obs_preddepth.tolist()] * self.HIST)
        self.obs_goals = np.array([obs_goal_depthfromwater[:3].tolist()] * self.HIST)
        self.obs_rays = np.array([obs_ray.tolist()] * self.HIST)
        self.obs_actions = np.array([[0, 0]] * self.HIST)
        self.obs_visibility = np.reshape(self.visibility_para_Gaussian, [1, 1, 1])

        # cv2.imwrite(
        #     "img_rgb_reset.png",
        #     256 * cv2.cvtColor(obs_img_ray[0] ** 0.45, cv2.COLOR_RGB2BGR),
        # )
        # cv2.imwrite("img_depth_pred_reset.png", 256 * self.obs_preddepths[0])

        self.firstDetect = True

        color_img = 256 * obs_img_ray[0] ** 0.45
        color_img = Image.fromarray(color_img.astype(np.uint8))
        color_img = transform(color_img).unsqueeze(0).to(self.device).float()
        color_img = gan(color_img).detach()
        grid = make_grid(color_img, normalize=True)
        color_img = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        # color_img = cv2.resize(color_img, dsize=(320, 256))
        color_img = yolo(color_img)

        detected = False

        for index, name in enumerate(color_img.pandas().xyxy[0]["name"].values):
            if name == "bottle":
                print(color_img.pandas().xyxy[0]["name"][index])
                xmin = color_img.pandas().xyxy[0]["xmin"][index]
                xmax = color_img.pandas().xyxy[0]["xmax"][index]
                ymin = color_img.pandas().xyxy[0]["ymin"][index]
                ymax = color_img.pandas().xyxy[0]["ymax"][index]
                xmid = (xmin + xmax) / 2
                xmid = int(xmid / 2)
                ymid = (ymin + ymax) / 2
                ymid = int(ymid / 2)
                # deep = depth[ymid, xmid]

                # vfov = 64 ish
                # hfov = 80
                size = (xmax - xmin) * (ymax - ymin) / 4
                depth = 1 / size * 1200
                vdeg = (64 - ymid) / 2
                horizontal = depth * abs(math.cos(math.radians(vdeg)))
                vertical = depth * math.sin(math.radians(vdeg))
                hdeg = (80 - xmid) / 2
                self.obs_goals = np.array([[horizontal, vertical, hdeg]] * self.HIST)
                self.randomGoal = False
                self.firstDetect = False
                detected = True
        if not detected:
            print("Using known goal location")
            self.randomGoal = True

        x0 = obs_goal_depthfromwater[4]
        y0 = obs_goal_depthfromwater[3]
        z0 = obs_goal_depthfromwater[5]
        currAng = obs_goal_depthfromwater[6]
        ang = currAng - self.obs_goals[0][2]
        ang = normalize_angle(ang)
        if ang > 270:
            ang = 360 - ang
            x = x0 - self.obs_goals[0][0] * math.sin(math.radians(ang))
            z = z0 + self.obs_goals[0][0] * math.cos(math.radians(ang))
        elif ang > 180:
            ang = ang - 180
            x = x0 - self.obs_goals[0][0] * math.sin(math.radians(ang))
            z = z0 - self.obs_goals[0][0] * math.cos(math.radians(ang))
        elif ang > 90:
            ang = 180 - ang
            x = x0 + self.obs_goals[0][0] * math.sin(math.radians(ang))
            z = z0 - self.obs_goals[0][0] * math.cos(math.radians(ang))
        else:
            x = x0 + self.obs_goals[0][0] * math.sin(math.radians(ang))
            z = z0 + self.obs_goals[0][0] * math.cos(math.radians(ang))

        y = y0 + self.obs_goals[0][1]
        self.prevGoal = [x, y, z]
        print(self.prevGoal)
        if self.randomGoal:
            self.prevGoal[0] += random.uniform(-3, 3)
            self.prevGoal[1] += random.uniform(-0.25, 0.25)
            self.prevGoal[2] += random.uniform(-3, 3)
            print(self.prevGoal)
            x1 = obs_goal_depthfromwater[4]
            y1 = obs_goal_depthfromwater[3]
            z1 = obs_goal_depthfromwater[5]
            x = self.prevGoal[0]
            y = self.prevGoal[1]
            z = self.prevGoal[2]

            ang = obs_goal_depthfromwater[6]
            goalDir = [x - x1, y - y1, z - z1]
            horizontal = math.sqrt(goalDir[0] ** 2 + goalDir[2] ** 2)
            vertical = goalDir[1]
            a = np.array([goalDir[0], goalDir[2]])
            a = a / np.linalg.norm(a)
            b = np.array([0, 1])
            goalAng = math.degrees(math.acos(np.dot(a, b)))
            if a[0] < 0:
                goalAng = 360 - goalAng
            hdeg = ang - goalAng
            if hdeg > 180:
                hdeg -= 360
            elif hdeg < -180:
                hdeg += 360

            self.obs_goals = np.array([[horizontal, vertical, hdeg]] * self.HIST)

        print("Score: {} / {}".format(self.total_correct, self.total_steps))

        return (
            self.obs_preddepths,
            self.obs_goals,
            self.obs_rays,
            self.obs_actions,
        )

    def step(self, action):
        self.time_before = time.time()
        # action[0] controls its vertical speed, action[1] controls its rotation speed
        action_ver = action[0]
        # action_ver = 0
        action_rot = action[1] * self.twist_range

        # observations per frame
        obs_img_ray, _, done, _ = self.env.step([action_ver, action_rot])
        obs_preddepth = self.dpt.run(obs_img_ray[0] ** 0.45)
        obs_ray = np.array(
            [
                np.min(
                    [
                        obs_img_ray[1][1],
                        obs_img_ray[1][3],
                        obs_img_ray[1][5],
                        obs_img_ray[1][33],
                        obs_img_ray[1][35],
                    ]
                )
                * 8
                * 0.5
            ]
        )

        # obs_ray = np.array([0])
        obs_goal_depthfromwater = self.pos_info.goal_depthfromwater_info()

        """
            compute reward
            obs_goal_depthfromwater[0]: horizontal distance
            obs_goal_depthfromwater[1]: vertical distance
            obs_goal_depthfromwater[2]: angle from robot's orientation to the goal (degree)
            obs_goal_depthfromwater[3]: robot's current y position
            obs_goal_depthfromwater[4]: robot's current x position            
            obs_goal_depthfromwater[5]: robot's current z position     
            obs_goal_depthfromwater[6]: robot's current orientation       
        """
        # 1. give a negative reward when robot is too close to nearby obstacles, seafloor or the water surface
        obstacle_distance = (
            np.min(
                [
                    obs_img_ray[1][1],
                    obs_img_ray[1][3],
                    obs_img_ray[1][5],
                    obs_img_ray[1][7],
                    obs_img_ray[1][9],
                    obs_img_ray[1][11],
                    obs_img_ray[1][13],
                    obs_img_ray[1][15],
                    obs_img_ray[1][17],
                ]
            )
            * 8
            * 0.5
        )
        obstacle_distance_vertical = (
            np.min(
                [
                    obs_img_ray[1][81],
                    obs_img_ray[1][79],
                    obs_img_ray[1][77],
                    obs_img_ray[1][75],
                    obs_img_ray[1][73],
                    obs_img_ray[1][71],
                ]
            )
            * 8
            * 0.5
        )
        if (
            obstacle_distance < 0.5
            or np.abs(obs_goal_depthfromwater[3]) < 0.24
            or obstacle_distance_vertical < 0.12
        ):
            reward_obstacle = -10
            done = True
            print(
                "Too close to the obstacle, seafloor or water surface!",
                "\nhorizontal distance to nearest obstacle:",
                obstacle_distance,
                "\ndistance to water surface",
                np.abs(obs_goal_depthfromwater[3]),
                "\nvertical distance to nearest obstacle:",
                obstacle_distance_vertical,
            )
        else:
            reward_obstacle = 0

        # 2. give a positive reward if the robot reaches the goal
        if self.training:
            if obs_goal_depthfromwater[0] < 0.6:
                reward_goal_reached = (
                    10
                    - 8 * np.abs(obs_goal_depthfromwater[1])
                    - np.abs(np.deg2rad(obs_goal_depthfromwater[2]))
                )
                done = True
                print("Reached the goal area!")
            else:
                reward_goal_reached = 0
        else:
            if obs_goal_depthfromwater[0] < 0.8:
                reward_goal_reached = (
                    10
                    - 8 * np.abs(obs_goal_depthfromwater[1])
                    - np.abs(np.deg2rad(obs_goal_depthfromwater[2]))
                )
                done = True
                print("Reached the goal area!")
            else:
                reward_goal_reached = 0

        # 3. give a positive reward if the robot is reaching the goal
        reward_goal_reaching_horizontal = (
            -np.abs(np.deg2rad(obs_goal_depthfromwater[2])) + np.pi / 3
        ) / 10
        if (obs_goal_depthfromwater[1] > 0 and action_ver > 0) or (
            obs_goal_depthfromwater[1] < 0 and action_ver < 0
        ):
            reward_goal_reaching_vertical = np.abs(action_ver)
            # print("reaching the goal vertically", obs_goal_depthfromwater[1], action_ver)
        else:
            reward_goal_reaching_vertical = -np.abs(action_ver)
            # print("being away from the goal vertically", obs_goal_depthfromwater[1], action_ver)

        # 4. give negative rewards if the robot too often turns its direction or is near any obstacle
        reward_turning = 0
        if 0.5 <= obstacle_distance < 1.0:
            reward_goal_reaching_horizontal *= (obstacle_distance - 0.5) / 0.5
            reward_obstacle -= (1 - obstacle_distance) * 2

        reward = (
            reward_obstacle
            + reward_goal_reached
            + reward_goal_reaching_horizontal
            + reward_goal_reaching_vertical
            + reward_turning
        )
        self.step_count += 1
        # print(self.step_count)

        if self.step_count > 500:
            done = True
            print("Exceeds the max num_step...")

        # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        obs_preddepth = np.reshape(
            obs_preddepth, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH)
        )
        self.obs_preddepths = np.append(
            obs_preddepth, self.obs_preddepths[: (self.HIST - 1), :, :], axis=0
        )

        color_img = 256 * obs_img_ray[0] ** 0.45
        color_img = Image.fromarray(color_img.astype(np.uint8))
        color_img = transform(color_img).unsqueeze(0).to(self.device).float()
        color_img = gan(color_img).detach()
        grid = make_grid(color_img, normalize=True)
        color_img = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        # color_img = cv2.resize(color_img, dsize=(320, 256))
        color_img = yolo(color_img)

        obs_goal = np.reshape(np.array(obs_goal_depthfromwater[0:3]), (1, DIM_GOAL))

        detected = False

        for index, name in enumerate(color_img.pandas().xyxy[0]["name"].values):
            if name == "bottle":
                print(color_img.pandas().xyxy[0]["name"][index])
                xmin = color_img.pandas().xyxy[0]["xmin"][index]
                xmax = color_img.pandas().xyxy[0]["xmax"][index]
                ymin = color_img.pandas().xyxy[0]["ymin"][index]
                ymax = color_img.pandas().xyxy[0]["ymax"][index]
                xmid = (xmin + xmax) / 2
                xmid = int(xmid / 2)
                ymid = (ymin + ymax) / 2
                ymid = int(ymid / 2)
                # deep = depth[ymid, xmid]

                # vfov = 64 ish
                # hfov = 80
                size = (xmax - xmin) * (ymax - ymin) / 4
                depth = 1 / size * 1200
                vdeg = (64 - ymid) / 2
                horizontal = depth * abs(math.cos(math.radians(vdeg)))
                vertical = depth * math.sin(math.radians(vdeg))
                hdeg = (80 - xmid) / 2
                if self.firstDetect:
                    self.obs_goals = np.array(
                        [[horizontal, vertical, hdeg]] * self.HIST
                    )
                else:
                    obs_goal = np.reshape(
                        np.array([horizontal, vertical, hdeg]), (1, DIM_GOAL)
                    )
                    self.obs_goals = np.append(
                        obs_goal, self.obs_goals[: (self.HIST - 1), :], axis=0
                    )
                self.firstDetect = False

                x0 = obs_goal_depthfromwater[4]
                y0 = obs_goal_depthfromwater[3]
                z0 = obs_goal_depthfromwater[5]
                currAng = obs_goal_depthfromwater[6]
                ang = currAng - self.obs_goals[0][2]
                ang = normalize_angle(ang)
                if ang > 270:
                    ang = 360 - ang
                    x = x0 - self.obs_goals[0][0] * math.sin(math.radians(ang))
                    z = z0 + self.obs_goals[0][0] * math.cos(math.radians(ang))
                elif ang > 180:
                    ang = ang - 180
                    x = x0 - self.obs_goals[0][0] * math.sin(math.radians(ang))
                    z = z0 - self.obs_goals[0][0] * math.cos(math.radians(ang))
                elif ang > 90:
                    ang = 180 - ang
                    x = x0 + self.obs_goals[0][0] * math.sin(math.radians(ang))
                    z = z0 - self.obs_goals[0][0] * math.cos(math.radians(ang))
                else:
                    x = x0 + self.obs_goals[0][0] * math.sin(math.radians(ang))
                    z = z0 + self.obs_goals[0][0] * math.cos(math.radians(ang))

                y = y0 + self.obs_goals[0][1]
                self.prevGoal = [x, y, z]
                print(self.prevGoal)
                detected = True
                self.total_correct += 1

        if not detected:
            # self.obs_goals = np.append(
            #     np.reshape(np.array([1, 0, 0]), (1, DIM_GOAL)),
            #     self.obs_goals[: (self.HIST - 1), :],
            #     axis=0,
            # )
            x1 = obs_goal_depthfromwater[4]
            y1 = obs_goal_depthfromwater[3]
            z1 = obs_goal_depthfromwater[5]
            x = self.prevGoal[0]
            y = self.prevGoal[1]
            z = self.prevGoal[2]

            ang = obs_goal_depthfromwater[6]
            goalDir = [x - x1, y - y1, z - z1]
            horizontal = math.sqrt(goalDir[0] ** 2 + goalDir[2] ** 2)
            vertical = goalDir[1]
            a = np.array([goalDir[0], goalDir[2]])
            a = a / np.linalg.norm(a)
            b = np.array([0, 1])
            goalAng = math.degrees(math.acos(np.dot(a, b)))
            if a[0] < 0:
                goalAng = 360 - goalAng
            hdeg = ang - goalAng
            if hdeg > 180:
                hdeg -= 360
            elif hdeg < -180:
                hdeg += 360

            obs_goal = np.reshape(np.array([horizontal, vertical, hdeg]), (1, DIM_GOAL))
            self.obs_goals = np.append(
                obs_goal, self.obs_goals[: (self.HIST - 1), :], axis=0
            )
            print("object not detected. Angle is {}".format(hdeg))

        self.total_steps += 1

        # single beam sonar and adaptation representation
        obs_ray = np.reshape(np.array(obs_ray), (1, 1))
        self.obs_rays = np.append(obs_ray, self.obs_rays[: (self.HIST - 1), :], axis=0)

        # # construct the observations of depth images, goal infos, and rays for consecutive 4 frames
        # obs_preddepth = np.reshape(obs_preddepth, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
        # self.obs_preddepths_buffer = np.append(obs_preddepth,
        #                                        self.obs_preddepths_buffer[:(2 ** (self.HIST - 1) - 1), :, :], axis=0)
        # self.obs_preddepths = np.stack((self.obs_preddepths_buffer[0], self.obs_preddepths_buffer[1],
        #                                self.obs_preddepths_buffer[3], self.obs_preddepths_buffer[7]), axis=0)
        #
        # obs_goal = np.reshape(np.array(obs_goal_depthfromwater[0:3]), (1, DIM_GOAL))
        # self.obs_goals_buffer = np.append(obs_goal, self.obs_goals_buffer[:(2 ** (self.HIST - 1) - 1), :], axis=0)
        # self.obs_goals = np.stack((self.obs_goals_buffer[0], self.obs_goals_buffer[1],
        #                                 self.obs_goals_buffer[3], self.obs_goals_buffer[7]), axis=0)
        #
        # obs_ray = np.reshape(np.array(obs_ray), (1, 1))  # single beam sonar
        # self.obs_rays_buffer = np.append(obs_ray, self.obs_rays_buffer[:(2 ** (self.HIST - 1) - 1), :], axis=0)
        # self.obs_rays = np.stack((self.obs_rays_buffer[0], self.obs_rays_buffer[1],
        #                            self.obs_rays_buffer[3], self.obs_rays_buffer[7]), axis=0)
        #
        obs_action = np.reshape(action, (1, DIM_ACTION))
        self.obs_actions = np.append(
            obs_action, self.obs_actions[: (self.HIST - 1), :], axis=0
        )

        self.time_after = time.time()
        # print("execution_time:", self.time_after - self.time_before)
        # print("goals:", self.obs_goals, "\nrays:", self.obs_rays, "\nactions:",
        #       self.obs_actions, "\nvisibility_Gaussian:", self.visibility_para_Gaussian, "\nreward:", reward)

        # cv2.imwrite("img_rgb_step.png", 256 * cv2.cvtColor(obs_img_ray[0] ** 0.45, cv2.COLOR_RGB2BGR))
        # cv2.imwrite("img_depth_pred_step.png", 256 * self.obs_preddepths[0])

        if self.training == False:
            my_open = open(
                os.path.join(assets_dir(), "learned_models/test_pos.txt"), "a"
            )
            data = [
                str(obs_goal_depthfromwater[4]),
                " ",
                str(obs_goal_depthfromwater[5]),
                " ",
                str(obs_goal_depthfromwater[3]),
                "\n",
            ]
            for element in data:
                my_open.write(element)
            my_open.close()

        return (
            self.obs_preddepths,
            self.obs_goals,
            self.obs_rays,
            self.obs_actions,
            reward,
            done,
            0,
        )


# env = []
# for i in range(1):
#     env.append(Underwater_navigation('midas', True, True, i, 4))
#
# while True:
#     a = 0
#     done = False
#     cam, goal, ray, action, visibility = env[0].reset()
#     # cam, goal, ray = env[1].reset()
#     # cam, goal, ray = env[2].reset()
#     # cam, goal, ray = env[3].reset()
#     # cam, goal, ray = env2.reset()
#     print(a, ray)
#     while not done:
#         cam, goal, ray, action, visibility, reward, done, _ = env[0].step([0, 1.0])
#         print(goal, ray, action)
#         # cam, goal, ray, reward, done, _ = env[1].step([0.0, 0.0])
#         # cam, goal, ray, reward, done, _ = env[2].step([0.0, 0.0])
#         # cam, goal, ray, reward, done, _ = env[3].step([0.0, 0.0])
#         # cam, goal, ray, reward, done, _ = env2.step([0.0, 0.0])
#         # print(a, ray)
#         a += 1
#         # print(obs[1], np.shape(obs[1]))
#         # cv2.imwrite("img2.png", 256 * cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
