import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from continuousEnv import ContinuousEnv, EnvArgs
import numpy as np
import random
from DNC_PPO import PPO
import csv
import matplotlib.pyplot as plt
import math


def main(budget, data):
    seed = 2

    np.random.seed(seed)

    user_num = 5      #5
    his_len = 1       #5
    info_num = 1      #3
    lamda = 50   #Todo origin: 50
    C = 20
    print("C:", C)



    Datasize_mnist = np.array([10000, 12000, 14000, 8000, 16000])*0.8
    Datasize_cifar = np.array([40, 38, 32, 46, 44])*250*0.8

    theta_num_mnist = 21840
    theta_num_cifar = 62006

    if data == 'cifar':
        D_true = (Datasize_cifar / 10) * (32 * (theta_num_cifar + 10 * (3 * 32 * 32))) / 1e9
        # pass #TODO
    else:
        D_true = (Datasize_mnist / 10) * (32 * (theta_num_mnist + 10 * 28 * 28)) / 1e9

    D = D_true
    print("D:",D)

    alpha = 0.1
    print("alpha:",alpha)

    tau = 5

    env_args = EnvArgs(user_num, his_len, info_num, C, D, alpha, tau, lamda, seed, budget, data)
    env = ContinuousEnv(env_args)


    # set the DRL agent
    A_DIM, S_DIM = user_num, user_num * his_len * info_num      # A_DIM, S_DIM = user_num, user_num * his_len * info_num

    BATCH = 5 ##Todo  #origin:20

    A_UPDATE_STEPS = 5  #origin:5
    C_UPDATE_STEPS = 5
    HAVE_TRAIN = False
    A_LR = 0.00003   # origin:0.00003
    C_LR = 0.00003
    v_s = np.zeros(user_num)
    GAMMA = 0.95   #origin: 0.95
    EP_MAX = 1500
    EP_LEN = 100
    dec = 0.3   #origin: 0.3
    action = np.zeros(user_num)
    ppo = PPO(S_DIM, A_DIM, BATCH, A_UPDATE_STEPS, C_UPDATE_STEPS, HAVE_TRAIN, 0)

    # define csvfiles for writing results
    Algs = "dnc"
    csvFile1 = open("test-lambda=0.5-Rewards_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)
    csvFile2 = open("test-lambda=0.5-Actions_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
    writer2 = csv.writer(csvFile2)
    csvFile3 = open("test-lambda=0.5-Aloss_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
    writer3 = csv.writer(csvFile3)
    csvFile4 = open("test-lambda=0.5-Closs_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
    writer4 = csv.writer(csvFile4)

    accu_rewards = []
    rewards = []
    actions = []
    closses = []
    alosses = []
    maxtimes = []
    totalprices = []
    performance_list = []

    reward_step = []

    rounds = []
    time_average = []

    cur_state = env.reset()
    for ep in range(EP_MAX):
        cur_state = env.reset()
        if ep % 50 == 0:
            dec = dec * 0.95
            A_LR = A_LR * 0.85
            C_LR = C_LR * 0.85
        buffer_s = []
        buffer_a = []
        buffer_r = []
        plot_reward = 0
        sum_reward = 0
        sum_action = 0
        sum_closs = 0
        sum_aloss = 0
        sum_maxtime=0
        sum_totalprice=0
        performance=0

        for t in range(EP_LEN):


            action = ppo.choose_action(cur_state.reshape(-1,S_DIM), dec)

            # print(action)

    #         action = np.random.random(np.shape(action))
            next_state, reward, maxtime, totalprice, reward_compare, budget, profit = env.step(action)

            reward_step.append(reward)

            # if budget < 0:
            #     break
    #         print(action,T,E)
            performance += profit
            sum_reward += reward
            plot_reward += reward_compare
            sum_action += action
            sum_maxtime += maxtime
            sum_totalprice += totalprice
            # sum_T += T
            # sum_E += E

            if budget > 0:

                buffer_a.append(action.copy())
                buffer_s.append(cur_state.reshape(-1,S_DIM).copy())
                buffer_r.append(reward)

                cur_state = next_state


                # update ppo
                if (t + 1) % BATCH == 0:
                    discounted_r = np.zeros(len(buffer_r), 'float32')
                    v_s = ppo.get_v(next_state.reshape(-1, S_DIM))
                    running_add = v_s

                    for rd in reversed(range(len(buffer_r))):
                        running_add = running_add * GAMMA + buffer_r[rd]
                        discounted_r[rd] = running_add

                    discounted_r = discounted_r[np.newaxis, :]
                    discounted_r = np.transpose(discounted_r)
                    if HAVE_TRAIN == False:
                        closs, aloss = ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, dec, A_LR, C_LR, ep)
                        sum_closs += closs
                        sum_aloss += aloss

            else:
                break
        print('ep                                                                                         :', ep)
        t += 1
        print('instant ep:', ep)
        print("instant reward:", reward)
        print("instant action:", action)
        print("instant maxtime:",maxtime)
        print("instant totalprice:", totalprice)

        accu_rewards.append(plot_reward)
        performance_list.append(performance)
        rewards.append(sum_reward)
        actions.append(sum_action / EP_LEN)
        maxtimes.append(sum_maxtime)

        rounds.append(t)
        totalprices.append(sum_totalprice / t)
        time_average.append(sum_maxtime/t)

        closses.append(sum_closs / EP_LEN)
        alosses.append(sum_aloss / EP_LEN)
        # Ts.append(sum_T / EP_LEN)
        # Es.append(sum_E / EP_LEN)
        # print("average reward:", sum_reward / EP_LEN)
        # print("average action:", sum_action / EP_LEN)
        # print("average maxtime:", sum_maxtime / EP_LEN)
        # print("average totalprice:", sum_totalprice / EP_LEN)
        # print("average closs:", sum_closs / EP_LEN)
        # print("average aloss:", sum_aloss / EP_LEN)

    plt.plot(reward_step)
    plt.show()


    plt.plot(rewards)
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    # plt.savefig("Reward_seed2_ep=5_optimal_cons.png", dpi=200)
    plt.show()

    # plt.plot(alosses)
    # plt.ylabel("Aloss")
    # plt.xlabel("Episodes")
    # # plt.savefig("Aloss_seed2_ep=5_optimal_cons.png", dpi=200)
    # plt.show()
    #
    # plt.plot(closses)
    # plt.ylabel("Closs")
    # plt.xlabel("Episodes")
    # # plt.savefig("Closs_seed2_ep=5_optimal_cons.png", dpi=200)
    # plt.show()


    #


    # plt.plot(accu_rewards)
    # plt.ylabel("-5000*ln(loss)-T")
    # plt.xlabel("Episodes")
    # # plt.savefig("Reward_seed2_ep=5_optimal_cons.png", dpi=200)
    # plt.show()



    # plt.plot(maxtimes)
    # plt.ylabel("Episode Total Time")
    # plt.xlabel("Episodes")
    # plt.show()
    #
    plt.plot(performance_list)
    plt.ylabel("-ln(loss)")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(maxtimes)
    plt.ylabel("Episode Total Time")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(rounds)
    plt.ylabel("rounds")
    plt.xlabel("Episodes")
    plt.show()
    # plt.plot(compare_list)
    # plt.ylabel("-5000*ln(loss)-T")
    # plt.xlabel("Episodes")
    # plt.show()

    plt.plot(time_average)
    plt.ylabel("Rounds Average Time")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(totalprices)
    plt.ylabel("Average Price")
    plt.xlabel("Episodes")
    # plt.savefig("Totalprice_seed2_ep=5_optimal_cons.png", dpi=200)
    plt.show()

    loss_result = np.mean(performance_list[-100:])
    time_result = np.mean(maxtimes[-100:])
    rounds_result = np.mean(rounds[-100:])
    a_time_result = np.mean(time_average[-100:])
    a_price_result = np.mean(totalprices[-100:])

    print("loss result: ", loss_result)
    print("time result: ", time_result)
    print("rounds result: ", rounds_result)
    print("Average Time result: ", a_time_result)
    print("Average Price result: ", a_price_result)

    return [loss_result, time_result, rounds_result, a_time_result, a_price_result]



    # REWARD=[]
    # for i in rewards:
    #     REWARD.append(-i*1000)
    #
    # writer1.writerow(REWARD)
    #
    # writer1.writerow(rewards)
    # # writer1.writerow(Ts)
    # # writer1.writerow(Es)
    # for i in range(len(actions)):
    #     writer2.writerow(actions[i])
    #
    # writer3.writerow(closses)
    # writer4.writerow(alosses)
    # csvFile1.close()
    # csvFile2.close()
    # csvFile3.close()
    # csvFile4.close()


    # if __name__ == '__main__':
    #     main()
    # writer1.writerow(rewards)
    # writer1.writerow(Ts)
    # writer1.writerow(Es)
    # for i in range(len(actions)):
    #     writer2.writerow(actions[i])
    # writer3.writerow(closses)
    # writer4.writerow(alosses)
    # csvFile1.close()
    # csvFile2.close()
    # csvFile3.close()
    # csvFile4.close()
    # tmp = []
    # fig = plt.figure()
    # for i in range(len(alosses)):
    #     tmp.append(-sum(alosses[0:0+i+1])/len(alosses[0:0+i+1]))
    # plt.plot(tmp)
    # plt.show()
    # # fig.savefig("aloss.png")
    # print(tmp)




if __name__ == '__main__':

    import pandas as pd
    names = ["-LN(loss)", "Time_totall", "Rounds", "Time_Average", "Average_Price"]
    budget_list = [1000]

    data_list = ['mnist']   #  'fmnist', 'cifar'
    for dd in data_list :
        tep = []
        for one in budget_list:
            tep.append(main(one, dd))
        data = pd.DataFrame(tep, index=budget_list, columns=names)
        data.to_csv(dd + '_baseline_result_1500.csv')