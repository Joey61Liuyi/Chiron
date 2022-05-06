import numpy as np
import pandas as pd
import math
import numpy as np
import pandas as pd
import math
import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import torch

class Env_low(gym.Env):

    def __init__(self, user_num, C, D, alpha, tau, delta_max, amplifier, communication_time):
        self.user_num = user_num
        self.C = C
        self.D = D
        self.alpha = alpha
        self.tau = tau
        # self.delta_max = np.random.uniform(low=1, high=2, size=5)
        self.delta_max = delta_max
        self.amplifier = amplifier
        self.done = False
        self.communication_time = communication_time
        self.observation_space = Box(shape = (1,), low=0, high=self.tau)
        self.action_space = Box(shape=(self.user_num,), low=0.01, high=1)
        # print("Max_delta:",self.delta_max)
        self.seed()

    def reset(self):
        self.done = False
        np.random.uniform(low=0.1, high=1, size=1)[0] * self.amplifier
        self.index = 0
        self.state = [np.random.uniform(low=0.1, high=1, size=1)[0]*self.amplifier]

        return self.state


    def step(self, action):

        action = action/sum(action)
        # print("Inner_Ratio", action)

        total_unit_price = self.state
        # print("Total_Unit_Price", total_unit_price)

        unit_price = total_unit_price * action
        # print("Unit_Price", unit_price)

        delta = unit_price / (self.tau * self.D * self.C * self.alpha)
        for i in range(0, self.user_num):
            if delta[i] > self.delta_max[i]:
                delta[i] = self.delta_max[i]


        self.time_cmp = (self.tau * self.D * self.C) / delta
        self.time_cmp += self.communication_time
        # print("Inner_computing_time", self.time_cmp)
        time_globle = np.max(self.time_cmp)
        time_idle = time_globle - self.time_cmp

        inner_reward1 = np.sum(time_idle)/1000
        inner_reward2 = np.average(time_idle*time_idle)
        inner_reward3 = np.var(self.time_cmp)
        inner_rewrad4 = math.pow(inner_reward2, 0.5)
        inner_reward5 = math.log(inner_reward1, 10)
        var = np.var(self.time_cmp)
        # print("Inner _VAR", -var)
        state_ = [np.random.uniform(low=0.1, high=1, size=1)[0]*self.amplifier]
        self.index +=1
        if self.index == 10:
            self.done = True
        self.state = state_

        return state_, -inner_reward1, self.done, {"time_var": inner_reward2}