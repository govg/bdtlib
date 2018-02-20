import sys
import math
import numpy as np
from random import randint
import create_syntheticdata
import matplotlib.pyplot as plt

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random, LinUCB


def calc_frobnorm(A, B):
    C = np.matrix(A) - np.matrix(B)
    f = np.linalg.norm(C)
    return f


def run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret, best_cum_regret, T, X, Y, narm, d, M,
                    filename_avg_regret_data, filename_cum_regret_data):

    regret = np.zeros(T, dtype=np.float)
    avg_regret = np.zeros(T, dtype=np.float)
    root_T = np.zeros(T, dtype=np.float)
    lin_T = np.zeros(T, dtype=np.float)

    for i in range(0, T):
        root_T[i] = 10000 * math.sqrt(i + 1)
        lin_T[i] = 100 * (i + 1)

        # Get context
        context = X[i, :]

        # Find the optimum arm in hindsight
        true_rewards = Y[i, :]
        opt_arm_in_hindsight = np.argmax(true_rewards)

        # Pull arm as per Online Bootstrap
        arm, exp_reward = bandit.choose(context)
        # print exp_reward
        try:
            bandit.update(context, Y[i][arm], exp_reward, reward_type, factor)
        except:
            break

        if i == 0:
            regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
        else:
            regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i - 1]

        avg_regret[i] = 1.0*regret[i] / (i + 1)
        if i % 1000 == 0:
            print i, regret[i], avg_regret[i], arm, opt_arm_in_hindsight, exp_reward, factor, alpha, bandit.name(), flag

    if avg_regret[i] < best_avg_regret[i]:
        best_avg_regret[:] = avg_regret[:]
        best_cum_regret[:] = regret[:]

        filename_best_avg_regret = filename_avg_regret_data + bandit.name() + str(flag)
        np.save(filename_best_avg_regret, best_avg_regret)

        filename_best_cum_regret = filename_cum_regret_data + bandit.name() + str(flag)
        np.save(filename_best_cum_regret, best_cum_regret)

    fp.write(bandit.name() + " regret = " + str(regret[i]) + " avg regret = " + str(avg_regret[i]) + " factor = " + str(
        factor) + " M = " + str(M) + " alpha = " + str(alpha) + "\n")
    fp.flush()

    return best_avg_regret[i]
