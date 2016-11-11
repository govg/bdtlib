import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

from bdtlib.contextual import ThompsonSampling, LinUCB, Random

# We can either use the standard boston house price
# dataset, or create our own
'''
boston = load_boston()

X = boston.data
Y = boston.target
d = X.shape[1]
N = X.shape[0]

'''
# Toy data, to play with
d = 10
N = 10000
X = np.random.randint(100, size=(N, d))
w = np.random.randint(10, size=(d, 1))
Y = np.matrix(X)*np.matrix(w)
noise = np.random.randint(10, size=(N, 1))
Y = Y + noise

# 25 trials
T = 25
narm = 10

# Initialise the bandits
bandits = [ThompsonSampling(0.05, d), LinUCB(0.5, d), Random()]
bnum = 0
colors = ['red', 'blue', 'green']

regret = np.zeros(T, dtype=np.float)

for bandit in bandits:
    curcolor = colors[bnum]
    for i in range(0, T):
        # Choose a subset of datapoints as arms
        arms = np.random.randint(N, size=narm)
        cts = X[arms, :]
        rwrds = Y[arms]

        # Ask agent to choose an arm and update it
        # with obtained reward
        arm = bandit.choose(cts)
        bandit.update(cts[arm, :], rwrds[arm])

        # To compute actual regret, we find max possible reward
        opt_arm = np.argmax(rwrds)

        if i == 0:
            regret[i] = rwrds[opt_arm] - rwrds[arm]
        else:
            regret[i] = ((rwrds[opt_arm] - rwrds[arm]) + regret[i-1])

    # plt.figure(bnum+1)
    plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
    plt.legend()
    bnum = bnum + 1
plt.show()
