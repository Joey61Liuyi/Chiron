# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 19:38
# @Author  : LIU YI

import argparse
import os
import sys
import time

import math
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.utils import clip_grad_norm_
# import torch.distributed.deprecated as dist
from cjltest.models import RNNModel
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD
from cjltest.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')       # 不需要设置
parser.add_argument('--ps-port', type=str, default='29500')         # 不需要设置
parser.add_argument('--this-rank', type=int, default=1)             # 不需要设置

# 模型与数据集
parser.add_argument('--data', type=str, default='C:\\Users\Yiliu\Desktop\Federated-Learning-PyTorch\data\PBT')
# parser.add_argument('--data', type=str, default='~/dataset')
# parser.add_argument('--data-dir', type=str, default='~/dataset')    # data dir of the dataset
# parser.add_argument('--data-name', type=str, default='cifar10')     # can be default
parser.add_argument('--model', type=str, default='LSTM')       # ResNet34OnFood101
parser.add_argument('--save-path', type=str, default='./')          # 结果的保存路径

# 参数信息
parser.add_argument('--workers-num', type=int, default=5)
parser.add_argument('--epochs', type=int, default=750)              # total epochs
parser.add_argument('--train-bsz', type=int, default=10)           # the total batch size of all workers
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
# 可能需要重点调节的参数
parser.add_argument('--lr', type=float, default=20)                  # 0 for using default value;
parser.add_argument('--K', type=int, default=5)                     # max number of local iterations
parser.add_argument('--gamma', type=float, default=0.2)             # compensated rate
parser.add_argument('--gamma-decay-epoch', type=int, default=200)   # decay in gamma-decay-epoch. Not decay by default.

parser.add_argument('--type', type=str, default='KAVG')             # 选择不同的算法： OSP, LOSP, KAVG
parser.add_argument('--stale-threshold', type=int, default=2)

args = parser.parse_args()

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            # noinspection PyUnresolvedReferences
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, ntokens, eval_batch_size, data_source, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


# noinspection PyTypeChecker
def run(workers, models, save_path, train_data_list, test_data, ntokens,train_batch_size):
    workers_num = len(workers)
    print('Model recved successfully!')
    optimizers_list = []
    if args.lr == 0.0:
        if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
            learning_rate = 0.1
        else:
            learning_rate = 0.01
    else:
        learning_rate = args.lr

    for i in workers:
        optimizer = MySGD(models[i].parameters(), lr=learning_rate)
        optimizers_list.append(optimizer)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 500
    else:
        decay_period = 200

    data_save = pd.DataFrame(columns=['Training Round', 'Training Loss', 'Training Perplexity', 'Test Loss', 'Test Perplexity'])
    print('Begin!')

    # store (train loss, energy, iterations)
    trainloss_file = args.save_path + '/trainloss' + args.model + '.txt'
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)               # 删掉已有同名文件
    f_trainloss = open(trainloss_file, 'a')

    iterations_num_epoch = 0
    sequence_iter = range(0, train_data_list[workers_num - 1].size(0) - 1, args.bptt)

    hidden_list = []
    for i in workers:
        hidden = models[i].init_hidden(train_batch_size)
        hidden_list.append(hidden)

    gamma = args.gamma
    first_label = True
    epoch_train_loss = 0.0
    test_loss = 10.0

    g_list = []
    for i in workers:
        g_temp = [torch.zeros_like(p.data) for p in models[0].parameters()]
        g_list.append(g_temp)
    it_count = 0
    s_time = time.time()

    epoch_loss = []


    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        for i in workers:
            models[i].train()
        iterations_epoch = 0
        user_loss = []
        for j in workers:
            sequence_iter = range(0, train_data_list[j - 1].size(0) - 1, args.bptt)
            batch_loss = []
            for batch, i in enumerate(sequence_iter):
                it_count += 1
                iterations_epoch += 1
                iteration_loss = 0.0
                print('Start Batch {} / {}'.format(batch, len(sequence_iter)))

                data, targets = get_batch(train_data_list[j - 1], i)
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden_list[j - 1] = repackage_hidden(hidden_list[j-1])
                optimizers_list[j-1].zero_grad()

                output, hidden_list[j-1] = models[j](data, hidden_list[j-1])

                loss = criterion(output.view(-1, ntokens), targets)
                loss.backward()

                # `clip_grad_norm_` helps prevent the exploding gradient problem in RNNs / LSTMs.
                clip_grad_norm_(models[j].parameters(), 0.25)
                delta_ws = optimizers_list[j-1].get_delta_w()
                # update local model and cache gradient into list
                for p_layer_idx, p_layer_temp in enumerate(models[j].parameters()):
                    p_layer_temp.data -= delta_ws[p_layer_idx]
                    g_list[j-1][p_layer_idx].data += delta_ws[p_layer_idx]

                iteration_loss += loss.data.item()      # worker i 当前round的平均loss
                batch_loss.append(loss.data.item())

            user_loss.append(sum(batch_loss)/len(batch_loss))

            epoch_train_loss += iteration_loss/workers_num

        epoch_loss.append(sum(user_loss)/len(user_loss))


        if epoch % args.K == 0:
            # Synchronization
            for p_idx, param in enumerate(models[0].parameters()):
                # in each worekr: update local model with the pulled global model and local update
                for w in workers:
                    if args.type == 'LOSP':
                        list(models[w].parameters())[p_idx].data = param.data - gamma * g_list[w - 1][p_idx]
                    elif args.type == 'OSP':
                        list(models[w].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)
                    else:
                        pass

                # # in cloud : update global parameter with the average of all updates
                # global_update_layer = torch.zeros_like(param.data)
                # for w in workers:
                #     global_update_layer += g_list[w-1][p_idx]
                # tensor = global_update_layer / workers_num
                # param.data -= tensor

                # in cloud : update global parameter with the average of all updates
                for w in workers:
                    param.data -= g_list[w - 1][p_idx] / workers_num

                # in each worekr: update local model with the pulled global model and local update
                for w in workers:
                    if args.type == 'KAVG':
                        list(models[w].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)
                    else:
                        pass
            g_list = []
            for w in workers:
                g_temp = [torch.zeros_like(p.data) for p in models[0].parameters()]
                g_list.append(g_temp)

        e_time = time.time()
        # train loss every epoch
        print('Epoch {}, Loss:{}'.format(epoch, loss.data.item()))


        # 训练结束后进行test
        if epoch % 5 == 0:
            # Run on test data.
            train_loss = sum(epoch_loss)/len(epoch_loss)
            epoch_loss = []
            test_loss = evaluate(models[0], ntokens, 10, test_data, criterion=criterion)
            data_save.append([{'Training Round': epoch, 'Training Loss': train_loss, 'Training Perplexity': math.exp(train_loss), 'Test Loss': test_loss, 'Test Perplexity': math.exp(test_loss)}])
            data_save.to_csv('PTB_data.csv')
            print("test_loss:", test_loss)
        f_trainloss.write(str(args.this_rank) +
                          "\t" + str(epoch_train_loss / float(iterations_epoch)) +
                          "\t" + str(args.K) +  # args.K
                          "\t" + str(e_time - epoch_start_time) +       # leave place for one epoch time
                          "\t" + str(iterations_epoch) +       # leave place for overall time
                          "\t" + str(math.exp(test_loss)) +       # leave place for perplexity
                          "\t" + str(test_loss) +       # leave place for test accuracy
                          "\t" + str(e_time - s_time) +  # leave place for total time
                          "\t" + str(0) +  # leave place for one iteration time of comp
                          "\t" + str(it_count) +  # leave place for one iteration time of comm
                          "\t" + str(it_count / args.K) + #global iterations
                          "\t" + str(epoch) +
                          '\n')
        f_trainloss.flush()
        epoch_train_loss = 0.0

        # 在指定epoch, gamma减半
        # 可以自己定义策略
        if (epoch + 1) > args.gamma_decay_epoch:
            if first_label:
                gamma = 0.01
                first_label = False

        if (epoch + 1) % decay_period == 0:
            for i in workers:
                for param_group in optimizers_list[i - 1].param_groups:
                    param_group['lr'] *= 0.1
                    print('LR Decreased! Now: {}'.format(param_group['lr']))

    f_trainloss.close()



def init_processes(workers,
                   models, save_path,
                   train_dataset_list, test_dataset,ntokens,train_batch_size,
                   fn, backend='tcp'):
    fn(workers, models, save_path, train_dataset_list, test_dataset, ntokens,train_batch_size)


if __name__ == '__main__':
    torch.manual_seed(1)

    workers_num = args.workers_num
    workers = [v+1 for v in range(workers_num)]


    corpus = Corpus(args.data)
    train_batch_size = 10
    train_batch_size = int(train_batch_size)
    eval_batch_size = 10

    train_data = batchify(corpus.train, train_batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    # 数据并行的分割规则
    start = None
    end = 0
    print(len(train_data))
    sp = dict()
    each = len(train_data) // len(workers)
    data_distribution = [15500, 19500, 21600, 12400, 23958]

    # for w in workers:
    #     if start is None:
    #         start = 0
    #     else:
    #         start += each
    #
    #     end += each
    #     sp[w] = (start, end)

    for w in workers:
        if start is None:
            start = 0
        else:
            start += data_distribution.pop(0)

        end += data_distribution[0]
        sp[w] = (start, end)

    train_data_list = []
    for i in workers:
        # 取得部分train_data，体现数据并行
        print('Start: {}, End: {}'.format(sp[i][0], sp[i][1]))
        train_data_sub = train_data[sp[i][0]: sp[i][1]].contiguous()
        train_data_list.append(train_data_sub)


    ntokens = len(corpus.dictionary)
    print("--------------------------",ntokens)

    models = []
    for i in range(workers_num + 1):
        model = RNNModel(args.model, ntokens,
                         ninp=10, nhid=10,
                         nlayers=2, dropout=0.2, tie_weights=True)
        models.append(model)

    print(get_parameter_number(model))

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')

    p = TorchProcess(target=init_processes, args=(workers,
                                                  models, save_path,
                                                  train_data_list, test_data, ntokens, train_batch_size,
                                                  run))
    p.start()
    p.join()
