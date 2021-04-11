# CHIRON

Implementation of ICDCS2021 paper : Incentive-Driven Long-term Optimization for Edge Learning by Hierarchical Reinforcement Mechanism.

Experiments are produced on MNIST, Fashion MNIST and CIFAR10.

Since the purpose of these experiments are to illustrate the effectiveness of the HRL pricing strategy in FL's incentive mechanism, only simple models such Lenet are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to dat
a directory and write a wrapper on pytorch dataset class.

## Running the experiments

The FL training process and the pricing strategy training are two offline training. To train the pricing strategy, running federated_main.py in Multi_client_data first to get the FL data. And use the FL data to train the pricing strategy. 
The greedy, DRL and Chiron methods are all integrated in train.py file. All parameters can be modified in configs.py

#### Federated Parameters
* ```data:```       Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```user_num:```   Default: 5.
* ```gpu:```        Default: 1 (runs on GPU). Can also be set to CPU mode.
* ```rounds:```     Default: 150. Number of maximum training rounds.
* ```local_ep:```   Default: 1. Number of the local epoch for each training round.
* ```lr:```         Learning rate set to 0.005 by default.
* ```iid:```        Default: 0. 0 represents iid, and 1 represents non-iid.
* ```unequal:```    Default: 1. 0 represents equal data split, and 1 represents non-equal data split.
* ```batch_size:``` Default: 10. 

#### HRL Parameters
* ```data:```       Default: 'mnist'.
* ```num_users:```  Number of users. Default is 5.
* ```lamda:```      The hyper parameter to control the trade off between accuracy and training time. Default as 1. 
* ```tau:```        Number of local training epochs in each user. Default is 1.
* ```his_len:```    Length of history information stored in the state of DRL. Default is 5.
* ```BATCH:```      Batch size of for each DRL's training. Default is 5.
* ```HAVE_TRAIN:``` Whether to retrain the DRL agent of load the trained one.
* ```EP_MAX:```     Maximum Episodes' length for integrated training.
* ```EP_MAX_pre_train:``` Maximum Episodes' length for inner agent's pre-training.
* ```EP_LEN:```     Before running out the budget, the maximum training rounds of FL.
* ```budget:```     The total budget for federated learning's server



