import numpy as np
import sys
import time

import create_syntheticdata
from testrecommendation1 import run_recommender

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random, LinUCB

# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 784
N = 5000
K = 10

def run_exp(X, Y, K, d):
    bn = 0
    narm = K
    T = N
    num_bandits = 4
    reward_type = "binary"
    filename_result = 'result_classification/result1.txt'
    # filename_result = 'results/result_independent_real.txt'
    # best_avg_regret = np.zeros((num_bandits,T), dtype=np.float)
    best_avg_regret = np.full((num_bandits, T), 99999999999)
    filename_plot_data = 'classification_plot/1_best_avg_regret'
    fp = open(filename_result, 'a')
    fp.write("Dependent Data\n")
    flag = 1
    while bn < num_bandits:
        bn += 1

        if bn == 1:
            # continue
            factor = 0
            alpha = 0
            M = 0

            start_time = time.time()
            for i in range(10):
                bandit = Random(narm=narm)
                run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn - 1][:], T, X, Y, narm,
                                d, M, filename_plot_data)
            avg_duration = float((time.time() - start_time) / 10.0)
            fp.write('Avg Duration = ' + str(avg_duration) + '\n')
            fp.flush()


        elif bn == 2:
            # continue
            alpha = 0
            M = 0

            start_time = time.time()
            c = 0
            for factor in range(5, 1030, 25):
                bandit = OnlineBootstrap(B=20, narm=narm, d=U + d + 1)
                run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn - 1][:], T, X, Y, narm,
                                d, M, filename_plot_data)
                c += 1

            avg_duration = float(1.0 * (time.time() - start_time) / c)
            fp.write('Avg Duration = ' + str(avg_duration) + '\n')
            fp.flush()


        elif bn == 3:
            # continue
            best_factor = 50
            alpha = 0
            M = int(narm / 2)
            best_regret = 99999999999
            best_M = M
            c = 0
            start_time = time.time()
            for factor in range(5, 1030, 25):
                bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=U + d + 1, M=M)
                r = run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn - 1][:], T, X, Y,
                                    narm, d, M, filename_plot_data)
                c += 1

                if r < best_regret:
                    best_regret = r
                    best_factor = factor

            for m in range(1, narm, 10):
                bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=U + d + 1, M=m)
                r = run_recommender(fp, flag, bandit, reward_type, best_factor, alpha, best_avg_regret[bn - 1][:], T, X,
                                    Y, narm, d, m, filename_plot_data)
                c += 1

                if r < best_regret:
                    best_regret = r
                    best_M = m

            print best_factor
            print best_M
            avg_duration = float(1.0 * (time.time() - start_time) / c)
            fp.write('Avg Duration = ' + str(avg_duration) + '\n')
            fp.flush()

        elif bn == 4:
            M = 0
            factor = 0
            start_time = time.time()
            c = 0
            alpha = 0.5
            while alpha <= 10:
                # pass
                bandit = LinUCB(alpha=alpha, d=U + d + 1, sigma=1, narm=narm)
                r = run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn - 1][:], T, X, Y,
                                    narm, d, M, filename_plot_data)
                alpha += 1
                c += 1

            avg_duration = float(1.0 * (time.time() - start_time) / c)
            fp.write('Avg Duration = ' + str(avg_duration) + '\n')
            fp.flush()

    fp.close()


def main():
    run_exp(X, Y, K, d)


if __name__ == "__main__":
    main()
