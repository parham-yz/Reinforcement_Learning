
import gym
import gym.spaces
from numpy.core.fromnumeric import resize
import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package
import matplotlib.pyplot as plt
import IPython
import time
import numpy as np
import collections
from ai import *
from gymWrappers import *




device = torch.device("cuda")
DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
env = make_env(DEFAULT_ENV_NAME)
jimi = Agent(env.action_space)
jimi.loadKnowlage("best.dat")
jimi.epsilon = 0
d = env.unwrapped.get_action_meanings()


state = env.reset()
while True:
    action = jimi.dicide(state)
    state, reward, done, _ = env.step(action)


    frame = np.array(state[0])
    cv2.imshow('w',cv2.resize(frame,(600,600)))
    cv2.waitKey(10)

    if done:
        state = env.reset()
        jimi.reset()
        


