import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

from Env_H import Env_H
from DNC_PPO import PPO
import matplotlib.pyplot as plt
from configs import Configs
from Env_Baseline import Env_Baseline
from Env_MCPPO import Env_MCPPO
from Env_low import Env_low   #todo inner layer only
import pandas as pd

class HRL_Pricing(object):

    def __init__(self, configs):
        self.configs = configs

    def inner_pretrain(self):

        # todo train the inner layer first
        a_DIM, s_DIM = self.configs.user_num, 1
        ppo_inner = PPO(1, a_DIM, 10, self.configs.A_UPDATE_STEPS, self.configs.C_UPDATE_STEPS, self.configs.HAVE_TRAIN,
                        0)
        env_inner = Env_low(self.configs.user_num, self.configs.C, self.configs.D, self.configs.alpha, self.configs.tau,
                            self.configs.delta_max, self.configs.amplifier_hrl, self.configs.comunication_time)
        dec = self.configs.dec
        a_LR = 0.00003
        c_LR = 0.00003

        # list for plotting
        reward_step = []
        rewards = []
        alosses = []
        closses = []

        for ep in range(self.configs.EP_MAX_pre_train):

            cur_state = env_inner.reset()

            if ep % 50 == 0:
                dec = dec * 0.95
                a_LR = a_LR * 0.85
                c_LR = c_LR * 0.85
            buffer_s = []
            buffer_a = []
            buffer_r = []

            sum_reward = 0
            sum_closs = 0
            sum_aloss = 0

            for t in range(self.configs.EP_LEN):
                action = ppo_inner.choose_action(cur_state.reshape(-1, s_DIM), dec)
                reward, next_state = env_inner.step(action)
                print("Inner_Time VAR", reward)
                print("======================================================")
                print('Inner training episode: ', ep)

                sum_reward += reward

                reward_step.append(reward)

                buffer_a.append(action.copy())
                buffer_s.append(cur_state)
                buffer_r.append(reward)

                cur_state = next_state

                if (t + 1) % 10 == 0:  # Batch = 10
                    discounted_r = np.zeros(len(buffer_r), 'float32')
                    v_s = ppo_inner.get_v(next_state.reshape(-1, s_DIM))
                    running_add = v_s

                    for rd in reversed(range(len(buffer_r))):
                        running_add = running_add * self.configs.GAMMA + buffer_r[rd]
                        discounted_r[rd] = running_add

                    discounted_r = discounted_r[np.newaxis, :]
                    discounted_r = np.transpose(discounted_r)
                    if self.configs.HAVE_TRAIN == False:
                        closs, aloss = ppo_inner.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, dec,
                                                        a_LR,
                                                        c_LR, ep)
                        sum_closs += closs
                        sum_aloss += aloss

            rewards.append(sum_reward / 100)
            alosses.append(sum_aloss / 100)
            closses.append(sum_closs / 100)

        rewards = np.array(rewards)*1000

        plt.plot(rewards)
        plt.ylabel("Time var")
        plt.xlabel("Episodes")
        plt.show()

        plt.plot(rewards[-100:])
        plt.ylabel("Time var last 100")
        plt.xlabel("Episodes")
        plt.show()

        plt.plot(alosses)
        plt.ylabel("Aloss")
        plt.xlabel("Episodes")
        plt.show()

        plt.plot(np.array(alosses)[-200:])
        plt.ylabel("Aloss last 200")
        plt.xlabel("Episodes")
        plt.show()

        plt.plot(closses)
        plt.ylabel("Closs")
        plt.xlabel("Episodes")
        plt.show()

        plt.plot(np.array(closses)[-200:])
        plt.ylabel("Closs last 200")
        plt.xlabel("Episodes")
        plt.show()
        return ppo_inner

    def HRL_train(self, ppo_inner):


        #todo  HRL training
        A_DIM, S_DIM = self.configs.user_num, self.configs.user_num * self.configs.his_len * self.configs.info_num + 2
        ppo_h = PPO(S_DIM, 1, self.configs.BATCH, self.configs.A_UPDATE_STEPS, self.configs.C_UPDATE_STEPS,
                    self.configs.HAVE_TRAIN, 0)
        # ppo_l = PPO(1, A_DIM, self.configs.BATCH, self.configs.A_UPDATE_STEPS, self.configs.C_UPDATE_STEPS,
        #             self.configs.HAVE_TRAIN, 0)
        ppo_l = ppo_inner


        env = Env_H(self.configs.user_num, self.configs.his_len, self.configs.info_num, self.configs.C, self.configs.D, self.configs.alpha, self.configs.tau, self.configs.lamda, self.configs.budget, self.configs.data, self.configs.amplifier_hrl, self.configs.reducer_hrl, self.configs.delta_max, self.configs.acc_increase_list, self.configs.comunication_time)
        dec = self.configs.dec
        A_LR = self.configs.A_LR
        C_LR = self.configs.C_LR
        A_LR = 0.000005
        C_LR = 0.000005


        # list for plotting
        Time_var_list = []
        Accumulated_reward_list = []
        Performance_list = []
        Round_list = []
        closses = []
        Time_list = []

        for ep in range(self.configs.EP_MAX):
            state_h = env.reset()
            if ep % 50 == 0:
                dec = dec * 0.95
                A_LR = A_LR * 0.85
                C_LR = C_LR * 0.85

            buffer_s = []
            buffer_a = []
            buffer_r = []

            buffer_s_low = []
            buffer_a_low = []
            buffer_r_low = []

            time_var_list_tep = []
            accumulated_reward = 0
            performance = 0
            time = 0
            sum_closs = []



            for t in range(self.configs.EP_LEN):

                print("HRL_episode", ep,"HRL_Round Index", t)
                price_total = ppo_h.choose_action(state_h.reshape(-1, S_DIM), dec)

                ratio = ppo_l.choose_action(price_total*self.configs.amplifier_hrl, dec)            #todo

                train_h_sign = (t + 1) % self.configs.BATCH == 0
                train_l_sign = (t + 1) % 2 == 0

                if train_l_sign:
                    self.PPO_trainer(ppo_l, buffer_s_low, buffer_a_low, buffer_r_low, dec, A_LR, C_LR, ep, 1, price_total)

                env.set_money(price_total)
                reward_l, reward_h, state_h_, done, profit, maxtime = env.step(ratio)
                # print(reward_l, reward_h, done)

                if done:
                    break
                time_var_list_tep.append(reward_l*100)
                accumulated_reward += reward_h
                time += maxtime

                buffer_s_low.append(price_total.reshape(-1, 1).copy()*self.configs.amplifier_hrl)
                buffer_a_low.append(ratio.copy())
                buffer_r_low.append(reward_l)

                buffer_s.append(state_h.reshape(-1, S_DIM).copy())
                buffer_a.append(price_total.copy())
                buffer_r.append(reward_h)

                state_h = state_h_

                if train_h_sign:
                    closs, aloss = self.PPO_trainer(ppo_h, buffer_s, buffer_a, buffer_r, dec, A_LR, C_LR, ep, S_DIM, state_h)
                    sum_closs.append(closs)
            closses.append(np.average(np.array(sum_closs)))
            Time_var_list.append(np.average(time_var_list_tep))
            Accumulated_reward_list.append(accumulated_reward)
            Round_list.append(env.index)
            Performance_list.append(self.configs.loss_list[env.index])
            Time_list.append(time)

        plt.plot(closses)
        plt.ylabel("Critic_loss_HRL_h")
        plt.xlabel("Episodes")
        plt.show()

        closses = pd.DataFrame(closses, columns=['Critic_loss'])
        closses.to_csv(str(self.configs.budget)+'_Critic_loss.csv')


        return Round_list, Time_list, Time_var_list, Accumulated_reward_list

    def greedy(self):



        env = Env_Baseline(self.configs.user_num, self.configs.his_len, self.configs.info_num, self.configs.C,
                           self.configs.D, self.configs.alpha, self.configs.tau, self.configs.lamda,
                           self.configs.budget, self.configs.data, self.configs.delta_max,
                           self.configs.acc_increase_list, self.configs.comunication_time,
                           self.configs.amplifier_baseline, self.configs.reducer_baseline)
        A_DIM = self.configs.user_num


        Time_var_list = []
        Accumulated_reward_list = []
        Round_list = []
        closses = []
        Time_list = []

        Actionset_list = []


        for ep in range(self.configs.EP_MAX):
            print(ep)
            cur_state = env.reset()
            sum_accuracy = 0
            sum_time = 0
            sum_reward = 0
            time_var_tep = []

            if len(Actionset_list) < 20:
                actionset = np.random.random(self.configs.EP_LEN*A_DIM)
                actionset = actionset.reshape(self.configs.EP_LEN, A_DIM)
                for t in range(self.configs.EP_LEN):
                    action = actionset[t]
                    next_state, reward, maxtime, totalprice, reward_compare, budget, profit, time_var = env.step(action)

                    if budget>0:
                        time_var_tep.append(time_var)
                        sum_reward += reward
                        sum_time += maxtime
                        round = env.index
                    else:
                        break

            else:
                tep = np.random.random(1)[0]
                if tep <= 0.2:
                    actionset = np.random.random(self.configs.EP_LEN * A_DIM)
                    actionset = actionset.reshape(self.configs.EP_LEN, A_DIM)
                    for t in range(self.configs.EP_LEN):
                        action = actionset[t]
                        next_state, reward, maxtime, totalprice, reward_compare, budget, profit, time_var = env.step(
                            action)

                        if budget > 0:
                            time_var_tep.append(time_var)
                            sum_reward += reward
                            sum_time += maxtime
                            round = env.index
                        else:
                            break
                else:
                    actionset = Actionset_list[0][0]
                    for t in range(self.configs.EP_LEN):
                        action = actionset[t]
                        next_state, reward, maxtime, totalprice, reward_compare, budget, profit, time_var = env.step(
                            action)

                        if budget > 0:
                            time_var_tep.append(time_var)
                            sum_reward += reward
                            sum_time += maxtime
                            round = env.index
                        else:
                            break


            for one in Actionset_list:
                if (one[0] == actionset).all():
                    Actionset_list.remove(one)

            Actionset_list.append((actionset, sum_reward))
            Actionset_list = sorted(Actionset_list, key=lambda x: x[1], reverse=True)
            if len(Actionset_list) > 20:
                Actionset_list.pop()


            Round_list.append(round)
            Time_list.append(sum_time)
            Accumulated_reward_list.append(sum_reward)
            Time_var_list.append(np.average(time_var_tep))


        return Round_list, Time_list, Time_var_list, Accumulated_reward_list

    def Baseline_train(self):

        env = Env_Baseline(self.configs.user_num, self.configs.his_len, self.configs.info_num, self.configs.C, self.configs.D, self.configs.alpha, self.configs.tau, self.configs.lamda, self.configs.budget, self.configs.data, self.configs.delta_max, self.configs.acc_increase_list, self.configs.comunication_time, self.configs.amplifier_baseline, self.configs.reducer_baseline)
        A_DIM, S_DIM = self.configs.user_num, self.configs.user_num * self.configs.his_len * self.configs.info_num
        ppo = PPO(S_DIM, A_DIM, self.configs.BATCH, self.configs.A_UPDATE_STEPS, self.configs.C_UPDATE_STEPS, self.configs.HAVE_TRAIN, 0)
        dec = self.configs.dec

        A_LR = self.configs.A_LR
        C_LR = self.configs.C_LR
        Performance_list = []
        Round_list = []
        Time_list = []
        Time_var_list = []
        Accumulated_reward_list = []

        for ep in range(self.configs.EP_MAX):
            cur_state = env.reset()
            if ep % 50 == 0:
                dec = dec * 0.95
                A_LR = A_LR * 0.85
                C_LR = C_LR * 0.85

            buffer_s = []
            buffer_a = []
            buffer_r = []

            Time = 0
            Time_var_list_tep = []
            accumulated_reward = 0

            for t in range(self.configs.EP_LEN):
                action = ppo.choose_action(cur_state.reshape(-1, S_DIM), dec)

                next_state, reward, maxtime, totalprice, reward_compare, budget, profit, time_var = env.step(action)

                Time += maxtime
                Time_var_list_tep.append(time_var)
                accumulated_reward += reward

                if budget > 0:

                    buffer_a.append(action.copy())
                    buffer_s.append(cur_state.reshape(-1, S_DIM).copy())
                    buffer_r.append(reward)

                    cur_state = next_state

                    train_sign = (t + 1) % self.configs.BATCH == 0

                    if train_sign:
                        self.PPO_trainer(ppo, buffer_s, buffer_a, buffer_r, dec, A_LR, C_LR, ep, S_DIM, next_state)

                else:
                    break

            Performance_list.append(self.configs.loss_list[env.index])
            Round_list.append(env.index)
            Time_list.append(Time)
            Time_var_list.append(np.average(Time_var_list_tep))
            Accumulated_reward_list.append(accumulated_reward)

        return Round_list, Time_list, Time_var_list, Accumulated_reward_list

    def MCPPO_train(self):
        env= Env_MCPPO(self.configs.user_num, self.configs.his_len, self.configs.info_num, self.configs.C, self.configs.D, self.configs.alpha, self.configs.tau, self.configs.lamda, self.configs.budget, self.configs.data, self.configs.delta_max, self.configs.acc_increase_list, self.configs.comunication_time, self.configs.amplifier_baseline, self.configs.reducer_baseline)
        A_DIM, S_DIM = self.configs.user_num, self.configs.user_num * self.configs.his_len * self.configs.info_num + 2
        ppo = PPO(S_DIM, A_DIM, self.configs.BATCH, self.configs.A_UPDATE_STEPS, self.configs.C_UPDATE_STEPS,
                  self.configs.HAVE_TRAIN, 0)
        dec = self.configs.dec
        A_LR = self.configs.A_LR
        C_LR = self.configs.C_LR

        A_LR = 0.000005
        C_LR = 0.000005


        Performance_list = []
        Rround_list = []
        Time_list = []
        Time_var_list = []
        Accumulated_reward_list = []


        for ep in range(self.configs.EP_MAX):
            cur_state = env.reset()
            if ep % 50 == 0:
                dec = dec * 0.95
                A_LR = A_LR * 0.85
                C_LR = C_LR * 0.85

            buffer_s = []
            buffer_a = []
            buffer_r = []

            time = 0
            Time_var_list_tep = []
            accumulated_reward = 0

            for t in range(self.configs.EP_LEN):
                action = ppo.choose_action(cur_state, dec)

                next_state, reward, maxtime, totalprice, budget, profit, reward_compare, time_var = env.step(action)

                time += maxtime
                Time_var_list_tep.append(time_var)
                accumulated_reward += reward

                if budget > 0:
                    buffer_a.append(action.copy())
                    buffer_s.append(cur_state.copy())
                    buffer_r.append(reward)

                    cur_state = next_state

                    train_sign = (t + 1) % self.configs.BATCH == 0

                    # if train_sign:
                    #     self.PPO_trainer(ppo, buffer_s, buffer_a, buffer_r, dec, A_LR, C_LR, ep, S_DIM, next_state)
                else:
                    self.PPO_trainer(ppo, buffer_s, buffer_a, buffer_r, dec, A_LR, C_LR, ep, S_DIM, next_state)
                    break

            Performance_list.append(self.configs.loss_list[env.index])
            Rround_list.append(env.index)
            Time_list.append(time)
            Time_var_list.append(np.average(Time_var_list_tep))
            Accumulated_reward_list.append(accumulated_reward)

        pass

        return Performance_list, Rround_list, Time_list, Time_var_list, Accumulated_reward_list

    def PPO_trainer(self, ppo, buffer_s, buffer_a, buffer_r, dec, A_LR, C_LR, ep, S_DIM, state_):

        discounted_r = np.zeros(len(buffer_r), 'float32')
        v_s = ppo.get_v(np.array(state_).reshape(-1, S_DIM))
        running_add = v_s

        for rd in reversed(range(len(buffer_r))):
            running_add = running_add * self.configs.GAMMA + buffer_r[rd]
            discounted_r[rd] = running_add

        discounted_r = discounted_r[np.newaxis, :]
        discounted_r = np.transpose(discounted_r)

        if self.configs.HAVE_TRAIN == False:
            closs, aloss = ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, dec, A_LR, C_LR, ep)
            return closs, aloss




if __name__ == '__main__':

    dataset = 'cifar'
    budget_list = [600, 800, 1000, 1200]
    # budget_list = [400, 500, 600, 700, 800]
    # budget_list = [400]
    methods_list = ['greedy', 'Baseline', 'HRL']
    train_acc_data = []
    train_loss_data = []
    test_acc_data = []
    test_loss_data = []
    rounds_data = []
    time_average_data = []
    time_var_data = []
    ppo_l = None

    for one in budget_list:

        np.random.seed(2)
        tf.compat.v1.set_random_seed(2)
        tf.random.set_random_seed(2)


        configs = Configs(dataset, one)
        hrl = HRL_Pricing(configs)

        if ppo_l == None:
            ppo_l = hrl.inner_pretrain()
        Rround_HRL, Time_HRL, Time_var_HRL, Accumulated_reward_HRL = hrl.HRL_train(ppo_l)
        # Rround_HRL, Time_HRL, Time_var_HRL, Accumulated_reward_HRL = [], [], [], []
        np.random.seed(5)
        tf.compat.v1.set_random_seed(5)
        tf.random.set_random_seed(5)
        #
        Rround_Baseline, Time_Baseline, Time_var_Baseline, Accumulated_reward_Baseline = hrl.Baseline_train()
        # Rround_Baseline, Time_Baseline, Time_var_Baseline, Accumulated_reward_Baseline = [], [], [], []

        np.random.seed(2)
        tf.compat.v1.set_random_seed(2)
        tf.random.set_random_seed(2)

        # Performance_greedy, Rround_greedy, Time_greedy, Time_var_greedy, Accumulated_reward_greedy = hrl.greedy_train()
        Rround_greedy, Time_greedy, Time_var_greedy, Accumulated_reward_greedy = hrl.greedy()


        # hrl.inner_pretrain()

        plt.plot(Accumulated_reward_HRL)
        plt.ylabel("Accumulated_reward_HRL")
        plt.xlabel("Episodes")
        plt.show()

        Accumulated_reward_HRL = pd.DataFrame(Accumulated_reward_HRL, columns=['Accumulated_reward_HRL'])
        Accumulated_reward_HRL.to_csv(str(one) + 'budget_Reward.csv')

        plt.plot(np.array(Accumulated_reward_HRL)[-200:])
        plt.ylabel("Accumulated_reward_HRL_CUT")
        plt.xlabel("Episodes")
        plt.show()
    #
        Time_average_Baseline = np.array(Time_Baseline)/np.array(Rround_Baseline)
        Time_average_greedy = np.array(Time_greedy) / np.array(Rround_greedy)
        Time_average_HRL = np.array(Time_HRL) / np.array(Rround_HRL)

        time_var_data.append([-np.average(np.array(Time_var_greedy)[-25:]), -np.average(np.array(Time_var_Baseline)[-25:]),
                              -np.average(np.array(Time_var_HRL)[-25:])])
        rounds_data.append([np.average(np.array(Rround_greedy)[-25:]), np.average(np.array(Rround_Baseline)[-25:]),
                              np.average(np.array(Rround_HRL)[-25:])])

        train_acc_data.append([np.average(configs.train_acc[np.array(Rround_greedy)[-25:]]), np.average(configs.train_acc[np.array(Rround_Baseline)[-25:]]), np.average(configs.train_acc[np.array(Rround_HRL)[-25:]])])
        train_loss_data.append([np.average(configs.train_loss[np.array(Rround_greedy)[-25:]]),
                               np.average(configs.train_loss[np.array(Rround_Baseline)[-25:]]),
                               np.average(configs.train_loss[np.array(Rround_HRL)[-25:]])])
        test_acc_data.append([np.average(configs.test_acc[np.array(Rround_greedy)[-25:]]),
                               np.average(configs.test_acc[np.array(Rround_Baseline)[-25:]]),
                               np.average(configs.test_acc[np.array(Rround_HRL)[-25:]])])
        test_loss_data.append([np.average(configs.test_loss[np.array(Rround_greedy)[-25:]]),
                               np.average(configs.test_loss[np.array(Rround_Baseline)[-25:]]),
                               np.average(configs.test_loss[np.array(Rround_HRL)[-25:]])])

        time_average_data.append([np.average(np.array(Time_average_greedy)[-25:]), np.average(np.array(Time_average_Baseline)[-25:]),
                              np.average(np.array(Time_average_HRL)[-25:])])


        # check convergence
        plt.plot(Accumulated_reward_Baseline)
        plt.ylabel("Accumulated_reward_Baseline")
        plt.xlabel("Episodes")
        plt.show()

        plt.plot(Accumulated_reward_greedy)
        plt.ylabel("Accumulated_reward_greedy")
        plt.xlabel("Episodes")
        plt.show()


        # Compare

        plt.plot(np.array(Time_var_Baseline), color='black', label='Baseline')
        # plt.plot(np.array(Time_var_greedy), color='blue', label='greedy')
        plt.plot(np.array(Time_var_HRL), color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Time variance Compare")
        plt.show()

        plt.plot(np.array(Time_var_Baseline)[-20:], color='black', label='Baseline')
        # plt.plot(np.array(Time_var_greedy)[-20:], color='blue', label='greedy')
        plt.plot(np.array(Time_var_HRL)[-20:], color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Time variance Compare (last 20)")
        plt.show()


        plt.plot(np.array(Time_Baseline), color='black', label='Baseline')
        # plt.plot(np.array(Time_greedy), color='blue', label='greedy')
        plt.plot(np.array(Time_HRL), color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Time total Compare")
        plt.show()

        plt.plot(np.array(Time_Baseline)[-20:], color='black', label='Baseline')
        # plt.plot(np.array(Time_greedy)[-20:], color='blue', label='greedy')
        plt.plot(np.array(Time_HRL)[-20:], color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Time total Compare (last 20)")
        plt.show()

        plt.plot(np.array(Rround_Baseline), color='black', label='Baseline')
        plt.plot(np.array(Rround_greedy), color='blue', label='greedy')
        plt.plot(np.array(Rround_HRL), color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Rounds Compare")
        plt.show()

        plt.plot(np.array(Rround_Baseline)[-20:], color='black', label='Baseline')
        plt.plot(np.array(Rround_greedy)[-20:], color='blue', label='greedy')
        plt.plot(np.array(Rround_HRL)[-20:], color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Rounds Compare (last 20)")
        plt.show()


        plt.plot(np.array(Time_average_Baseline), color='black', label='Baseline')
        # plt.plot(np.array(Time_average_greedy), color='blue', label='greedy')
        plt.plot(np.array(Time_average_HRL), color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Time average Compare")
        plt.show()

        plt.plot(np.array(Time_average_Baseline)[-20:], color='black', label='Baseline')
        # plt.plot(np.array(Time_average_greedy)[-20:], color='blue', label='greedy')
        plt.plot(np.array(Time_average_HRL)[-20:], color='red', label='HRL')
        plt.legend()
        plt.xlabel("Episodes")
        plt.title("Time average Compare (last 20)")
        plt.show()

    test_acc_data = pd.DataFrame(test_acc_data, columns=methods_list, index= budget_list)
    test_acc_data.plot(kind='bar')
    plt.title('Test ACC')
    plt.show()
    test_acc_data.to_csv(dataset+'_test_acc_data.csv')

    test_loss_data = pd.DataFrame(test_loss_data, columns=methods_list, index= budget_list)
    test_loss_data.plot(kind='bar')
    plt.title('Test Loss')
    plt.show()
    test_loss_data.to_csv(dataset+'_test_loss_data.csv')

    train_acc_data = pd.DataFrame(train_acc_data, columns=methods_list, index= budget_list)
    train_acc_data.plot(kind='bar')
    plt.title('Train ACC')
    plt.show()
    train_acc_data.to_csv(dataset+'_train_acc_data.csv')

    train_loss_data = pd.DataFrame(train_loss_data, columns=methods_list, index= budget_list)
    train_loss_data.plot(kind='bar')
    plt.title('Train Loss')
    plt.show()
    train_loss_data.to_csv(dataset+'_train_loss_data.csv')

    rounds_data = pd.DataFrame(rounds_data, columns=methods_list, index= budget_list)
    rounds_data.plot(kind='bar')
    plt.title('Rounds')
    plt.show()
    rounds_data.to_csv(dataset+'_round_data.csv')

    time_average_data = pd.DataFrame(time_average_data, columns=methods_list, index= budget_list)
    time_average_data.plot(kind='bar')
    plt.title('Average Time')
    plt.show()
    time_average_data.to_csv(dataset+'_time_average_data.csv')


    time_var_data = pd.DataFrame(time_var_data, columns=methods_list, index= budget_list)
    time_var_data.plot(kind='bar')
    plt.title('Time Variance')
    plt.show()
    time_var_data.to_csv(dataset+'_time_var_data.csv')


