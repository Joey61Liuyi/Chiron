import numpy as np
import pandas as pd
import math
import copy
class Configs(object):

    def __init__(self, data, budget):

        self.data = data
        self.lamda = 4000
        self.user_num = 5
        self.tau = 5
        self.his_len = 5
        self.info_num = 3


        if self.data == 'cifar':
            theta_num = 62006
            if self.user_num == 5:
                data_size = np.array([40, 38, 32, 46, 44]) * 250 * 0.8

            else:
                data_size = pd.read_csv('Multi_client_data/'+str(self.user_num)+'_cifar.csv')
                data_size = np.array(data_size[0].tolist())*0.8
        else:
            theta_num = 21840
            if self.user_num == 5:
                data_size = np.array([10000, 12000, 14000, 8000, 16000]) * 0.8
            else:
                data_size = pd.read_csv('Multi_client_data/' + str(self.user_num) + 'mnist.csv')
                data_size = np.array(data_size['data_size'].tolist())

        self.D = (data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9
        self.alpha = 0.1
        self.tau = 5
        self.C = 20


        self.BATCH = 5   #todo
        self.A_UPDATE_STEPS = 5  # origin:5
        self.C_UPDATE_STEPS = 5
        self.HAVE_TRAIN = False
        self.A_LR = 0.00003  # origin:0.00003
        self.C_LR = 0.00003
        self.GAMMA = 0.95  # origin: 0.95
        self.dec = 0.3  #

        self.EP_MAX = 1000  #todo
        self.EP_MAX_pre_train = 1000

        self.EP_LEN = 100

        self.budget = budget  #todo


        if self.user_num== 5:
            self.delta_max = np.array([1.4359949, 1.05592623, 1.54966248, 1.43532239, 1.4203678])
        else:
            self.delta_max = np.random.uniform(low=1, high=2, size=self.user_num)


        if self.user_num==5:
            self.amplifier_hrl = 70
            self.amplifier_baseline = np.array([23, 23, 23, 13, 23])
            # self.amplifier_baseline = 23
        else:

            self.amplifier_hrl = np.sum(self.delta_max * self.tau * self.C * self.D * self.alpha)
            self.amplifier_baseline = np.max(self.delta_max * self.tau * self.C * self.D * self.alpha) * 10 / 9

        self.reducer_hrl = 1000
        self.reducer_baseline = 100

        reducer_pretrain_dict = {
            5: 1000,
            50: 10
        }

        self.reducer_pretrain = reducer_pretrain_dict[self.user_num]

        self.comunication_time = np.random.uniform(low=10, high=20, size=self.user_num)

        # self.comunication_time = 0


        if self.data == 'mnist':
            Loss = pd.read_csv('loss_mnist_500.csv')
            Loss = Loss.to_dict()
            Loss = Loss['1']
            loss_list = []
            for i in Loss:
                loss_list.append(Loss[i])

            num = len(loss_list)
            buffer = -math.log(1)
            profit_increase = []

            self.loss_list = copy.copy(loss_list)
            for i in range(0, num):
                loss_list[i] = -math.log(loss_list[i])

            for one in loss_list:
                profit_increase.append(one - buffer)
                buffer = one

            self.acc_increase_list = profit_increase

        else:
            data_info = pd.read_csv('Multi_client_data/'+str(self.user_num)+'user_'+self.data+'_'+str(self.tau)+'_0.005.csv')
            accuracy_list = data_info['loss'].tolist()
            num = len(accuracy_list)
            for i in range(0, num):
                accuracy_list[i] = -math.log(accuracy_list[i])

            buffer = 0
            self.acc_increase_list = []
            for one in accuracy_list:
                self.acc_increase_list.append(one-buffer)
                buffer = one



if __name__ == '__main__':
    c = Configs('fmnist', 800)

    a = c.acc_increase_list
    c = Configs('mnist', 800)
    b = c.acc_increase_list

    a = np.array(a)[0:17]
    b = np.array(b)[0:17]

    print(c.loss_list)

    print(3000/np.sum(b))
    # print(b)