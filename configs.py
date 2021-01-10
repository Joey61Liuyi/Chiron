import numpy as np
import pandas as pd
import math
import copy
class Configs(object):

    def __init__(self, data, budget):

        self.data = data

        if self.data == 'mnist' or 'fmnist':
            self.lamda = 4000
        elif self.data == 'cifar':
            self.lamda = 1800
        elif self.data == 'cifar100':
            self.lamda = 1800
        self.user_num = 5
        self.user_num = 5
        self.his_len = 5
        self.info_num = 3
        self.tau = 1


        if self.data == 'cifar':
            theta_num = 62006
            if self.user_num == 5:
                data_size = np.array([40, 38, 32, 46, 44]) * 250 * 0.8

            else:
                data_size = pd.read_csv('Multi_client_data/'+str(self.user_num)+'_cifar.csv')
                data_size = np.array(data_size[0].tolist())*0.8
            self.D = (data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9

        elif self.data == 'cifar100':
            theta_num = 69656
            if self.user_num == 5:
                data_size = np.array([40, 38, 32, 46, 44]) * 250 * 0.8

            else:
                each = 50000 / self.user_num
                data_size = np.ones(self.user_num) * each
            self.D = (data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9

        elif self.data == 'PTB':
            data_size = np.array([15500, 19500, 21600,  12400, 23958])
            theta_num = 111760
            self.D = (data_size / 10) * (32 * (theta_num + 10000)) / 1e9

        else:
            theta_num = 21840
            if self.user_num == 5:
                data_size = np.array([10000, 12000, 14000, 8000, 16000]) * 0.8
            else:
                data_size = pd.read_csv('Multi_client_data/' + str(self.user_num) + self.data + '.csv')
                data_size = np.array(data_size['data_size'].tolist())
            self.D = (data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9




        self.alpha = 0.1

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


        if self.user_num == 5:
            self.delta_max = np.array([1.4359949, 1.05592623, 1.54966248, 1.43532239, 1.4203678])
        else:
            self.delta_max = np.random.uniform(low=1, high=2, size=self.user_num)


        if self.user_num == 5:
            self.amplifier_hrl = np.sum(self.delta_max * self.tau * self.C * self.D * self.alpha)
            self.amplifier_baseline = np.max(self.delta_max * self.tau * self.C * self.D * self.alpha) * 10 / 9
            # self.amplifier_hrl = 70
            # self.amplifier_baseline = np.array([23, 23, 23, 13, 23])
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


        federated_info = pd.read_csv('5user_' + self.data + '_'+str(self.tau)+'_0.001.csv')
        performance_metrics = federated_info.columns

        metrics_choice = 1 # TODO [test acc, test loss, train acc, train loss]

        self.test_acc = federated_info[performance_metrics[0]]
        self.test_loss = federated_info[performance_metrics[1]]
        self.train_acc = federated_info[performance_metrics[2]]
        self.train_loss = federated_info[performance_metrics[3]]

        self.performance_list = federated_info[performance_metrics[metrics_choice]]
        self.performance_increase = []


        if metrics_choice%2 == 0:
            buffer = 0
            for i in self.performance_list:
                self.performance_increase.append(i-buffer)
                buffer = i
            pass
        else:
            if self.data == 'mnist':
                buffer = -math.log(50)
            elif self.data == 'cifar':
                buffer = -math.log(4)
            else:
                buffer = -math.log(6)


            for i in self.performance_list:
                self.performance_increase.append(-math.log(i) - buffer)
                buffer = -math.log(i)



        if self.data == 'mnist':
            Loss = pd.read_csv('loss_mnist_500.csv')
            Loss = Loss.to_dict()
            Loss = Loss['1']
            Loss = self.test_loss
            buffer = -math.log(400)
            profit_increase = []
            Loss_list = []
            self.loss_list = copy.copy(Loss)
            for i in range(0, len(Loss)):
                Loss_list.append(-math.log(Loss[i]))

            for one in Loss_list:
                profit_increase.append(one - buffer)
                buffer = one

            self.acc_increase_list = profit_increase

        elif self.data == 'fmnist':
            Loss = pd.read_csv('tep_fmnist_500.csv')
            Loss = Loss.to_dict()
            Loss = Loss['1']
            Loss = self.test_loss
            buffer = -math.log(400)
            profit_increase = []
            Loss_list = []
            self.loss_list = copy.copy(Loss)
            for i in range(0, len(Loss)):
                Loss_list.append(-math.log(Loss[i]))

            for one in Loss_list:
                profit_increase.append(one - buffer)
                buffer = one

            self.acc_increase_list = profit_increase

        else:

            Loss = pd.read_csv('tep_cifar_500.csv')
            Loss = Loss['1']
            Loss = self.test_loss
            buffer = -math.log(400)
            profit_increase = []
            Loss_list = []
            self.loss_list = copy.copy(Loss)
            for i in range(0, len(Loss)):
                Loss_list.append(-math.log(Loss[i]))

            for one in Loss_list:
                profit_increase.append(one - buffer)
                buffer = one

            self.acc_increase_list = profit_increase




if __name__ == '__main__':

    c = Configs('fmnist', 800)
    print(c.test_acc)

    print(c.acc_increase_list)
    # c = Configs('cifar', 800)
    # print(c.D)
    # c = Configs('PTB', 800)
    # print(c.D)