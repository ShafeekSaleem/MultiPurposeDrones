import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient()
        self.action_space = spaces.Discrete(7)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()

        
        self.drone.enableApiControl(True)
        self.image_request = airsim.ImageRequest(
            1, airsim.ImageType.Scene, False, False
        )

        self.drone.armDisarm(True)

        # Set start position and velocity
        self.drone.moveToPositionAsync(0.0, 0.0, -15, 5).join()
        self.drone_state = self.drone.getMultirotorState()
        self.state["position"] = self.drone_state.kinematics_estimated.position
        time.sleep(3)

    def transform_obs(self, responses):
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width,3))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_temp = image.resize((self.image_shape[0], self.image_shape[1]))
        im_temp = np.mean(np.array(im_temp), axis=2).astype(np.uint8)
        im_final = np.reshape(im_temp, (self.image_shape[0],self.image_shape[1],1))
        return im_final

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

    def _compute_reward(self):
        thresh_dist = 7
        beta = 1

        z = -15.0 # Set z axis

        target_dest = np.array([-45.0, -95.0, z])

        pts = [
            np.array([0.0, 0.0, z]),
            np.array([-20.0, 0.0, z]),
            np.array([-45.0, 0.0, z]),
            np.array([-45.0, -50.0, z]),
            np.array([-45.0, -95.0, z]),
        ]

        quad_prev_pt = np.array(
            list(
                (
                    self.state["prev_position"].x_val,
                    self.state["prev_position"].y_val,
                    self.state["prev_position"].z_val,
                )
            )
        )

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        if self.state["collision"]:
            print("collided")
            reward = -100
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                )

            if dist > thresh_dist:
                print("Far from track")
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.25
                reward_target = 0.01
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.25
                )
                if np.linalg.norm(target_dest-quad_pt) < np.linalg.norm(target_dest-quad_prev_pt):
                    print("reward for distance keeping: "+str(reward_dist))
                    print("reward for speed: "+str(reward_speed))
                    print("reward for target: "+str(reward_target))
                    
                    reward = reward_dist + reward_speed + reward_target
                    print("Total reward for the step: ", str(reward))
                else:
                    print("reward for distance keeping: "+str(reward_dist))
                    print("reward for speed: "+str(reward_speed))
                    print("reward for target: "+str(-1*reward_target))
                    reward = reward_dist + reward_speed - reward_target
                    print("Total reward for the step: ", str(reward))

        done = 0
        if reward <= -10:
            done = 1
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
