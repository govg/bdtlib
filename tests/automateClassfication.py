import numpy as np
import sys
import time
from readMNIST import read_MNIST
from sklearn.preprocessing import normalize
import create_syntheticdata
from testClassificationBandit import run_recommender

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random, LinUCB

# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 784
N = 60000
K = 10

def run_exp(X, Y, K, d):
    bn = 0
    narm = K
    T = N
    cov_mult = 0.0005
    num_bandits = 4
    reward_type = 'binary'
    filename_result = 'result_classification/result1.txt'
    # filename_result = 'results/result_independent_real.txt'
    # best_avg_regret = np.zeros((num_bandits,T), dtype=np.float)
    best_avg_regret = np.full((num_bandits, T), 99999999999)
    filename_plot_data = 'classification_plot/1_best_avg_regret_'
    fp = open(filename_result, 'a')

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
            for factor in range(1, 102, 5):
                bandit = OnlineBootstrap(B=20, narm=narm, d=d, reward_type=reward_type, cov_mult=cov_mult)
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
            for factor in range(1, 102, 5):
                bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=d, M=M, reward_type=reward_type, cov_mult=cov_mult)
                r = run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn - 1][:], T, X, Y,
                                    narm, d, M, filename_plot_data)
                c += 1

                if r < best_regret:
                    best_regret = r
                    best_factor = factor

            for m in range(1, narm+1):
                bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=d, M=m, reward_type=reward_type, cov_mult=cov_mult)
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
            continue
            M = 0
            factor = 0
            start_time = time.time()
            c = 0
            alpha = 0.5
            while alpha <= 10:
                # pass
                bandit = LinUCB(alpha=alpha, d=d, sigma=1, narm=narm)
                r = run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn - 1][:], T, X, Y,
                                    narm, d, M, filename_plot_data)
                alpha += 1
                c += 1

            avg_duration = float(1.0 * (time.time() - start_time) / c)
            fp.write('Avg Duration = ' + str(avg_duration) + '\n')
            fp.flush()

    fp.close()


def main():
    trainImageFile = 'train-images-idx3-ubyte.gz'
    trainLabelFile = 'train-labels-idx1-ubyte.gz'
    testImageFile = 't10k-images-idx3-ubyte.gz'
    testLabelFile = 't10k-labels-idx1-ubyte.gz'

    X, _, Y = read_MNIST(trainImageFile, trainLabelFile, train=True)
    # XTest, _, yTest, _ = read_MNIST(testImageFile, testLabelFile, train=False)

    X = normalize(X)
    run_exp(X, Y, K, d)


if __name__ == "__main__":
    main()
