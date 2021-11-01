
import gym
import gym.spaces
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
from IPython.display import clear_output



def vis(x,d):
    state , action, reward, done , newState = x
    for i in range(len(state)):
        frame = np.array(torch.cat([state[i,0],newState[i,0]],1))
        print(d[int(action[i])],reward[i],done[i])
        cv2.imshow('w',cv2.resize(frame,[1400,700]))
        cv2.waitKey(0)

device = torch.device("cuda")
DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
rewards = collections.deque(maxlen=20)



env = make_env(DEFAULT_ENV_NAME)
jimi = Agent(env.action_space)

d = env.unwrapped.get_action_meanings()


state = env.reset()
while True:
    action = jimi.dicide(state)
    newState, reward, done, _ = env.step(action)
    jimi.giveFeedBack([state.copy(),action,reward,done,newState])
    state = newState.copy()

    if done:
        rewards.append(jimi.totalReward)
        clear_output(wait=True)
        plt.plot(jimi.temp_history)
        plt.show()
        print(np.round(np.mean(rewards),2), jimi.interaction_counter, np.round(jimi.epsilon,2))
        if jimi.totalReward > 10:
            break
        state = env.reset()
        jimi.reset()
        
jimi.saveKnowlage("best.dat")

