# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 16:51
# @Author  : LIU YI

class Configs(object):
    def __init__(self):
        self.data = 'mnist'
        self.user_num = 5
        self.gpu = 1
        self.rounds = 150
        self.local_ep = 1
        self.iid = 0
        self.unequal = 1
        self.frac = 1
        self.lr = 0.005
        self.model = 'cnn'
        if self.data == 'cifar':

            self.batch_size = 50
        else:
            self.batch_size = 10

if __name__ == '__main__':
    config = Configs()