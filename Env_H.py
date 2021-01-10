import numpy as np
import pandas as pd
import math
class Env_H(object):

    def __init__(self, user_num, his_len, info_num, C, D, alpha, tau, lamda, budget, dataset, amplifier, reducer, delta_max, profit_increase, communication_time):

        self.user_num = user_num
        self.his_len = his_len
        self.info_num = info_num
        self.delta_max = delta_max

        self.lamda = lamda
        self.C = C
        self.D = D
        self.alpha = alpha
        self.tau = tau
        self.budget_total = budget
        self.dataset = dataset

        # self.delta_max = np.random.uniform(low=1, high=2, size=5)

        self.state = None
        self.state_l = None
        self.state_h = None
        self.index = None
        self.profit_increase = None

        self.amplifier = amplifier
        self.reducer = reducer
        self.profit_increase = profit_increase
        self.communication_time = communication_time

    def reset(self):
        self.done = False
        self.budget = self.budget_total
        self.index = 0
        self.state = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')
        state_add = np.array([self.index, self.budget])
        self.state_h = np.append(self.state.reshape(-1, ), state_add.reshape(-1, ))
        self.state_h = self.state_h.reshape(-1, )

        return self.state_h

    def set_money(self, money):

        money = self.amplifier * money
        # print(money, "the total price")
        self.state_l = money

        # if money > self.budget:
        #     self.state_l = self.budget
        #     self.done = True
        # else:
        #     self.state_l = money

    def step(self, ratio):

        ratio = ratio/sum(ratio)
        total_unit_price = self.state_l
        print("HRL_total_unit_price", total_unit_price)

        unit_price = total_unit_price * ratio
        print("HRL_unit_price", unit_price)

        delta = unit_price / (self.tau * self.D * self.C * self.alpha)
        print("HRL_Delta", delta)

        # max_delta limitation
        for i in range(0, self.user_num):
            if delta[i] > self.delta_max[i]:
                delta[i] = self.delta_max[i]
        print("HRL_True_Delta", delta)

        # time var
        self.time_cmp = (self.tau * self.D * self.C) / delta
        self.time_cmp += self.communication_time
        print("HRL_Computing time", self.time_cmp)

        time_globle = np.max(self.time_cmp)
        print("HRL_global time", time_globle)

        var = np.var(self.time_cmp)
        time_idle = time_globle - self.time_cmp
        reward_inner = np.sum(time_idle)
        time_efficiency = np.sum(self.time_cmp)/(self.user_num * time_globle)


        print("HRL_Time VAR", var)

        # money reduce
        price = np.dot(unit_price, delta)
        print("HRL_Price", price)

        self.budget -= price
        # print(self.budget)
        self.done = self.budget <= 0

        # calculate reward
        profit = self.profit_increase[self.index]
        reward = self.lamda * profit - time_globle  # self.lamda * profit - (time_globle + 10 * price)
        self.index += 1
        reward /= self.reducer
        print("HRL_Reward", reward)

        print("================================================================================")

        # state transition
        self.state = np.delete(self.state, 0, axis=0)
        ob_list = []
        for one in range(0, self.user_num):
            ob_list.append(unit_price[one])  # TODO may need discussion
            ob_list.append(delta[one])
            ob_list.append(self.time_cmp[one])

        ob_list = np.array(ob_list)
        ob_list = np.reshape(ob_list, (1, self.user_num, 3))

        self.state = np.concatenate((self.state, ob_list), axis=0)
        state_add = np.array([self.index, self.budget])
        self.state_h = np.append(self.state.reshape(-1, ), state_add.reshape(-1, ))
        self.state_h = self.state_h.reshape(-1, )


        return -reward_inner/100, reward, self.state_h, self.done, profit, time_globle, time_efficiency
