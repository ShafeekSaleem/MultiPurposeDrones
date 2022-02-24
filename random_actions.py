import setup_path
import gym
import airgym
import time
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from airgym.envs.drone_env import AirSimDroneEnv

#env = AirSimDroneEnv(step_length=0.25, image_shape=(84, 84, 1))
# env = gym.make(
#                 "airgym:airsim-drone-sample-v0",
#                 step_length=1,
#                 image_shape=(84, 84, 1),
#             )
# Reset it, returns the starting frame
# img2d,image = env.reset()
# img = Image.fromarray(img2d, 'RGB')
# img.save('norm.png')
# image.save('preprocessed.png')
# frame.show()

# Render
# env.render()

# for i in range(5):
#   # Perform a random action, returns the new frame, reward and whether the game is over
#   act = env.action_space.sample()
#   frame, reward, is_done, _ = env.step(act)
#   print("=============================")
#   print("action" + str(act))
#   print(reward, _)
