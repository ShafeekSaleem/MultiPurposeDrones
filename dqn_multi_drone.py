import setup_path
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import airsim
from marl.airpettingzoo.airsim_pettingzoo import MultiDroneEnv

# Configuration paramaters for the whole setup
seed = 42

# setup the environment
env = MultiDroneEnv(step_length=0.25, image_shape=(84, 84, 1), no_of_agents=4, target_position=airsim.Vector3r(-45.0, -95.0, -15))

def print_pos():
    print("Drone0: ",obs["Drone0"]['abs_position_NED'])
    print("============================")
    print("Drone1: ",obs["Drone1"]['abs_position_NED'])
    print("============================")
    print("Drone2: ",obs["Drone2"]['abs_position_NED'])
    print("============================")
    print("Drone3: ",obs["Drone3"]['abs_position_NED'])
    print("============================")
    
obs = env.reset()
print(type(obs["Drone0"]['velocity']))
# print_pos()

# time.sleep(3)
# actions = [np.random.choice(7) for i in range(4)]
# obs, rewards, dones, _ = env.step(actions)
# print_pos()
# print(obs["similarity_matrix"])

# time.sleep(3)
# actions = [np.random.choice(7) for i in range(4)]
# obs, rewards, dones, _ = env.step(actions)
# print_pos()
# print(obs["similarity_matrix"])