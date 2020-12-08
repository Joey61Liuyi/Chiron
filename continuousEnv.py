import numpy as np
import pandas as pd
import math


class EnvArgs(object):
    def __init__(self, user_num, his_len, info_num, C, D, alpha, tau, lamda, seed, budget, data):
        self.user_num = user_num
        self.his_len = his_len
        self.info_num = info_num
        self.C = C
        self.D = D
        self.alpha = alpha
        self.tau = tau
        self.lamda= lamda
        self.seed = seed
        self.budget = budget
        self.data = data



class ContinuousEnv(object):

    def __init__(self, env_args):
        self.user_num = env_args.user_num
        self.his_len = env_args.his_len
        self.info_num = env_args.info_num
        self.lamda = env_args.lamda
        self.C = env_args.C
        self.D = env_args.D
        self.alpha = env_args.alpha
        self.tau = env_args.tau
        self.seed = env_args.seed
        np.random.seed(self.seed)
        self.budget_total = env_args.budget
        self.data = env_args.data
        self.state = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')
        # self.delta_max = np.random.uniform(low=1, high=2, size=5)
        self.delta_max = np.array([1.4359949, 1.02592623, 1.54966248, 1.43532239, 1.4203678])
        # print("Max_delta:",self.delta_max)
        self.getloss()

    def getloss(self):

        if self.data == 'mnist':
            Loss = pd.read_csv('loss_mnist_500.csv')
        elif self.data == 'fmnist':
            Loss = pd.read_csv('tep_fmnist_500.csv')
        elif self.data == 'cifar':
            Loss = pd.read_csv('tep_cifar_500.csv')

        Loss = Loss.to_dict()
        Loss = Loss['1']
        loss_list = []
        for i in Loss:
            loss_list.append(Loss[i])

        num = len(loss_list)
        buffer = 0
        profit_increase = []
        for i in range(0,num):

            loss_list[i] = -math.log(loss_list[i])

        for one in loss_list:

            profit_increase.append(one - buffer)
            buffer = one



        self.profit_increase = profit_increase

    def reset(self):
        self.budget = self.budget_total
        self.index = 0
        self.state_high = np.zeros((self.his_len, self.user_num, self.info_num), 'float32')


        self.state_low = np.zeros((self.his_len, self.user_num, 1), 'float32')

        return self.state_outter



    def step_low(self, total_unit_price, ratio):
        w = 23
        b = 1
        total_unit_price = total_unit_price * w + b
        unit_price   = total_unit_price * ratio

        delta = unit_price / (self.tau * self.D * self.C * self.alpha)
        for i in range(0, self.user_num):
            if delta[i] > self.delta_max[i]:
                delta[i] = self.delta_max[i]


        self.time_cmp = (self.tau * self.D * self.C) / delta
        time_globle = np.max(self.time_cmp)
        var = np.var(self.time_cmp)
        return var




    def step(self, action):  # continuous action

        # w = np.array([23, 23, 23, 23, 23])
        # b = np.array([1, 1, 1, 1, 1])
        # print("Action", action)
        # action = action * w + b
        action = action/(sum(action))


        print("True Action", action)

        delta = action / (self.tau * self.D * self.C * self.alpha)


        # for i in range(0, self.user_num):
        #     if delta[i] > self.delta_max[i]:
        #         delta[i] = self.delta_max[i]


        self.time_cmp = (self.tau * self.D * self.C) / delta
        print('Computing time:', self.time_cmp)

        time_globle = np.max(self.time_cmp)
        price = np.dot(action, delta)
        print("Price:", price)

        self.budget -= price

        reward = -(time_globle + 10 * price)
        print("Reward:", reward)
        profit = self.profit_increase[self.index]
        reward_compare = 5000 * profit - time_globle -price
        self.index += 1
        # print("round time:",time_globle)
        # print("Price:",price)
        print("Round:", self.index)
        reward = reward/100  #origin: 1000


        self.state = np.delete(self.state, 0, axis=0)

        ob_list = []
        for one in range(0,self.user_num):
            ob_list.append(action[one])
            ob_list.append(delta[one])
            ob_list.append(self.time_cmp[one])

        state_ = np.array(ob_list)
        state_ = np.reshape(state_, (1,5,3))
        self.state = np.concatenate((self.state, state_), axis = 0)

        return self.state, reward, time_globle, price, reward_compare, self.budget, profit
