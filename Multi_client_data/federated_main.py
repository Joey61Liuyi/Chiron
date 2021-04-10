#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, googlenet, ResNet18, AlexNet, LeNet
from utils import get_dataset, average_weights, exp_details
import pandas as pd
import random
from cjltest.models import RNNModel
from cjltest.utils_model import MySGD

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':

    np.random.seed(0)
    torch.random.manual_seed(0)
    random.seed(0)
    start_time = time.time()
    acc_list = []
    loss_list = []
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = torch.device('cuda' if args.gpu else 'cpu')
    # load dataset and user groups

    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            global_model = LeNet()
        elif args.dataset == 'PBT':
            global_model = RNNModel()

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.

    print(get_parameter_number(global_model))
    print('---------------------------------------------------------------------------------------')

    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    info = pd.DataFrame(columns=['test_acc', 'test_loss', 'train_acc', 'train_loss'])
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_acc = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        idxs_users = list(user_groups.keys())

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, train_loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            acc, loss = local_model.inference(model=global_model)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(train_loss))
            local_acc.append(acc)

        # update global weights
        global_weights = average_weights(local_weights)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', type(global_weights))
        # print(len(global_weights))
        # print(global_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg = sum(local_acc) / len(local_acc)

        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        info = info.append([{'test_acc': test_acc, 'test_loss': test_loss, 'train_acc': acc_avg, 'train_loss': loss_avg}])
        info.to_csv(str(args.num_users)+'user_'+args.dataset + '_' + str(args.local_ep) +'_' + str(args.lr)+ '_.csv', index=None)

        # print global training loss after every 'i' rounds

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {loss_avg}')
            print('Train Accuracy: {:.2f}% \n'.format(100*acc_avg))



    # Saving the objects train_loss and train_accuracy:
    file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)




    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

