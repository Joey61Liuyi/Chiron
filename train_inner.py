import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Env_low import Env_low
import numpy as np
import random
from DNC_PPO import PPO
import csv
import matplotlib.pyplot as plt
import math
import wandb


def main():


    wandb.init(project="ICDCS_EXTENSION", name="Previous version")
    user_num = 5
    C = 20

    Datasize_mnist = np.array([10000, 12000, 14000, 8000, 16000])*0.8
    theta_num_mnist = 21840
    D_true = (Datasize_mnist / 10) * (32 * (theta_num_mnist + 10 * 28 * 28)) / 1e9
    D = D_true

    alpha = 0.1
    tau = 5
    delta_max = np.array([1.4359949, 1.05592623, 1.54966248, 1.43532239, 1.4203678])
    amplifier = 70
    comunication_time = np.random.uniform(low=10, high=20, size=user_num)

    env = Env_low(user_num, C, D, alpha, tau, delta_max, amplifier, comunication_time)

    S_DIM = 1
    A_DIM = user_num
    BATCH = 20 #TODO
    A_UPDATE_STEPS = 5
    C_UPDATE_STEPS = 5
    HAVE_TRAIN = False
    A_LR = 0.00005   # origin:0.00003
    C_LR = 0.00005


    GAMMA = 0.95   #origin: 0.95
    EP_MAX = 1000
    EP_LEN = 100
    dec = 0.3


    ppo = PPO(S_DIM, A_DIM, BATCH, A_UPDATE_STEPS, C_UPDATE_STEPS, HAVE_TRAIN, 0)


    # Plot Initial
    reward_step = []
    rewards = []
    alosses = []
    closses = []

    for ep in range(EP_MAX):

        cur_state = env.reset()

        if ep % 50 == 0:
            dec = dec * 0.95
            A_LR = A_LR * 0.85
            C_LR = C_LR * 0.85
        buffer_s = []
        buffer_a = []
        buffer_r = []

        sum_reward = 0
        sum_closs = 0
        sum_aloss = 0
        time_var = 0

        for t in range(EP_LEN):
            action = ppo.choose_action(np.array(cur_state).reshape(-1,S_DIM), dec)
            next_state, reward, done, info= env.step(action)
            print("VAR", reward)
            print("==================================================================")

            sum_reward += reward
            time_var += info["time_var"]
            reward_step.append(reward)
            buffer_a.append(action.copy())
            buffer_s.append(cur_state)
            buffer_r.append(reward)

            cur_state = next_state

            #update PlPO per BATCH
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
            if done:
                break

        rewards.append(sum_reward/EP_LEN)
        info = {
            "epoch": ep,
            "test_reward": sum_reward/EP_LEN,
            "best_reward": max(rewards),
            "time_var": time_var/EP_LEN
        }
        wandb.log(info)
        alosses.append(sum_aloss/EP_LEN)
        closses.append(sum_closs/EP_LEN)

    plt.plot(np.array(rewards)*1000)
    plt.ylabel("Time var")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(np.array(rewards)[-200:]*1000)
    plt.ylabel("Time var last 200")
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


if __name__ == '__main__':
    main()