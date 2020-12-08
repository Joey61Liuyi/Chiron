import numpy as np
from collections import defaultdict
import pandas as pd
import math
class Env_MCPPO(object):

    def __init__(self, user_num, his_len, info_num, C, D, alpha, tau, lamda, budget, dataset, delta_max, profit_increase, communication_time, amplifier, reducer):
        self.user_num = user_num
        self.his_len = his_len
        self.info_num = info_num
        self.lamda = lamda
        self.C = C
        self.D = D
        self.alpha = alpha
        self.tau = tau
        self.budget_total = budget
        self.index = None
        self.budget = None
        self.dataset = dataset

        self.state = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')
        self.delta_max = delta_max
        # self.delta_max = np.random.uniform(low=1, high=2, size=5)
        # print("Max_delta:",self.delta_max)
        self.profit_increase = profit_increase
        self.communication_time = communication_time
        self.amplifier = amplifier
        self.reducer = reducer

    def reset(self):
        self.index = 0
        self.budget = self.budget_total
        self.state = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')
        self.state_new = [self.index , self.budget]
        self.state_new = np.array(self.state_new)
        self.state_all = np.append(self.state.reshape(-1, ), self.state_new.reshape(-1, ))
        self.state_all = self.state_all.reshape(-1, )
        return self.state_all

    def step(self, action):  # continuous action

        # if self.dataset == 'cifar':
        #     w = np.array([38, 38, 38, 38, 38])
        #     b = np.array([1, 1, 1, 1, 1])
        # else:
        #     w = 23
        #     b = 1
        print("MCPPO_Action", action)

        action = action*self.amplifier

        # action = action*23 + 1  #origin: 200
        print("MCPPO_True Action", action)

        delta = action / (self.tau * self.D * self.C * self.alpha)
        print("MCPPO_delta", delta)

        for i in range(0, self.user_num):
            if delta[i] > self.delta_max[i]:
                delta[i] = self.delta_max[i]
        print("MCPPO_True delta", delta)

        self.time_cmp = (self.tau * self.D * self.C) / delta
        self.time_cmp += self.communication_time
        print("MCPPO_computing time", self.time_cmp)


        time_globle = np.max(self.time_cmp)
        print("MCPPO_round time", time_globle)

        price = np.dot(action, delta)
        print("MCPPO_Price", price)

        self.budget -= price

        profit = self.profit_increase[self.index]
        print("MCPPO_Profit", profit)

        reward = self.lamda * profit - time_globle      # self.lamda * profit - (time_globle + 10 * price)
        reward_compare = self.lamda * profit - time_globle
        self.index += 1
        # print("round time:",time_globle)
        # print("Price:",price)
        reward = reward/self.reducer  #origin: 1000
        print('MCPPO_time VAR:  ', np.var(self.time_cmp))
        print('MCPPO_index: ', self.index)
        print('MCPPO_reward: ', reward)
        print("================================================================================")


        self.state = np.delete(self.state, 0, axis=0)

        ob_list = []
        for one in range(0,self.user_num):
            ob_list.append(action[one])
            ob_list.append(delta[one])
            ob_list.append(self.time_cmp[one])

        state_ = np.array(ob_list)
        state_ = np.reshape(state_, (1,self.user_num,3))

        self.state = np.concatenate((self.state, state_), axis = 0)
        self.state_new = [self.index, self.budget]
        self.state_new = np.array(self.state_new)
        self.state_all = np.append(self.state.reshape(-1, ), self.state_new.reshape(-1, ))
        self.state_all = self.state_all.reshape(-1, )

        return self.state_all, reward, time_globle, price, self.budget, profit, reward_compare, -np.var(self.time_cmp)