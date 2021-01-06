import numpy as np
import pandas as pd
import math
class Env_Baseline(object):

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
        self.dataset = dataset
        self.state = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')
        # self.delta_max = np.random.uniform(low=1, high=2, size=5)
        self.delta_max = delta_max
        # print("Max_delta:",self.delta_max)
        self.profit_increase = profit_increase
        self.communication_time = communication_time
        self.amplifier = amplifier
        self.reducer = reducer
    def reset(self):
        self.budget = self.budget_total
        self.index = 0
        self.state = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')
        return self.state

    def step(self, action):

        # if self.dataset == 'cifar':
        #     w = np.array([38, 38, 38, 38, 38])
        #     b = np.array([1, 1, 1, 1, 1])
        # else:
        #     w = 23
        #     b = 1


        print("Baseline_Action", action)
        action = action * self.amplifier
        print("Baseline_True Action", action)

        delta = action / (self.tau * self.D * self.C * self.alpha)
        print("Baseline_Delta:", delta)

        for i in range(0, self.user_num):
            if delta[i] > self.delta_max[i]:
                delta[i] = self.delta_max[i]


        self.time_cmp = (self.tau * self.D * self.C) / delta
        self.time_cmp += self.communication_time
        print('Baseline_Computing time:', self.time_cmp)

        VAR=np.var(self.time_cmp)
        print("Baseline_VAR", VAR)

        time_globle = np.max(self.time_cmp)
        price = np.dot(action, delta)
        print("Baseline_Price:",price)

        self.budget -= price

        reward = -(time_globle + 10 * price)
        print("Baseline_Reward:", reward)

        profit = self.profit_increase[self.index]

        reward_compare = self.lamda * profit - time_globle
        self.index += 1
        # print("round time:",time_globle)
        # print("Price:",price)
        print("Baseline_Index:", self.index)
        reward = reward/self.reducer  #origin: 1000


        self.state = np.delete(self.state, 0, axis=0)

        ob_list = []
        for one in range(0,self.user_num):
            ob_list.append(action[one])
            ob_list.append(delta[one])
            ob_list.append(self.time_cmp[one])

        state_ = np.array(ob_list)
        state_ = np.reshape(state_, (1,self.user_num,3))
        self.state = np.concatenate((self.state, state_), axis = 0)
        idle_time = np.sum(time_globle - self.time_cmp)




        return self.state, reward, time_globle, price, reward_compare, self.budget, profit, -idle_time
