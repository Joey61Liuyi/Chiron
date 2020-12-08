import numpy as np
import pandas as pd
import math
class Env_low(object):

    def __init__(self, user_num, C, D, alpha, tau, delta_max, amplifier, communication_time):
        self.user_num = user_num
        self.C = C
        self.D = D
        self.alpha = alpha
        self.tau = tau
        # self.delta_max = np.random.uniform(low=1, high=2, size=5)
        self.delta_max = delta_max
        self.amplifier = amplifier
        self.communication_time = communication_time
        # print("Max_delta:",self.delta_max)

    def reset(self):
        self.state = np.random.uniform(low=0.1, high=1, size=1)[0]*self.amplifier
        return self.state


    def step(self, action):

        action = action/sum(action)
        print("Inner_Ratio", action)

        total_unit_price = self.state
        print("Total_Unit_Price", total_unit_price)

        unit_price = total_unit_price * action
        print("Unit_Price", unit_price)

        delta = unit_price / (self.tau * self.D * self.C * self.alpha)
        for i in range(0, self.user_num):
            if delta[i] > self.delta_max[i]:
                delta[i] = self.delta_max[i]


        self.time_cmp = (self.tau * self.D * self.C) / delta
        self.time_cmp += self.communication_time
        print("Inner_computing_time", self.time_cmp)
        time_globle = np.max(self.time_cmp)
        var = np.var(self.time_cmp)
        print("Inner_VAR", -var)

        state_ = np.random.uniform(low=0.1, high=1, size=1)[0]*self.amplifier
        self.state = state_


        return (-var)/1000, state_
