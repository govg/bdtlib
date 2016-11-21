
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.datasets import load_boston

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
'''
d = 10
N = 10000
X = np.random.randint(100, size=(N, d))
w = np.random.randint(10, size=(d, 1))
Y = np.matrix(X)*np.matrix(w)
noise = np.random.randint(10, size=(N, 1))
Y = Y + noise
'''

'''
# Load Avazu data into memory
data = np.load("../Avazu/small_data.npy")
# The first column is actually timestamp, useless for us
X = data[:, 1:]
Y = np.load("../Avazu/small_labels.npy")

zeromask = Y == 0
onemask = Y == 1

print "Original data dimensions ", X.shape


for i in range(0,Y.shape[0]):
    if zeromask[i] == True:
        if np.random.randint(8) < 6:
            zeromask[i] = False


totalmask = zeromask + onemask
X = X[totalmask,:]
Y = Y[totalmask]
'''

# Load masked Avazu into memory
X = np.load("../Avazu/X33.npy")
Y = np.load("../Avazu/Y33.npy")

N = X.shape[0]
d = X.shape[1]

print "Loaded data with dims : ", N, " x ", d
print "Mean of Y array is : ", Y.mean()

# 25 trials
T = 1000
narm = 20
var = 5

# Initialise the bandits
bandits = [ThompsonSampling(0.5, d), LinUCB(0.5, d)]#, Random()]
bnum = 0
colors = ['red', 'blue', 'green']

regret = np.zeros(T, dtype=np.float)
cumsum = np.zeros(T, dtype=np.float)


if T*narm > Y.shape[0]:
    print "Sampling > total"
    T = Y.shape[0]/narm

for bandit in bandits:
    curcolor = colors[bnum]
    for i in range(0, T):
        # Choose a subset of datapoints as arms
        # arms = np.random.randint(N, size=narm)

        # Since data is sequential, we shall use it in order
        arms = np.arange(narm) + i*narm
        cts = X[arms, :]
        binrewards = Y[arms]

        # Conver all 0s to -20, 1s to +20, add some noise.
        binrewards[binrewards == 0] = -20 
        binrewards[binrewards == 1] = 20
        # noise = np.random.multivariate_normal(np.zeros((narm,)),
        #        np.identity(narm)*var)

        # print noise.shape
        # rwrds = binrewards + noise
        rwrds = binrewards

        # Ask agent to choose an arm and update it
        # with obtained reward
        arm = bandit.choose(cts)
        bandit.update(cts[arm, :], rwrds[arm])

        # To compute actual regret, we find max possible reward
        opt_arm = np.argmax(rwrds)

        if i == 0:
            regret[i] = (rwrds[opt_arm] - rwrds[arm])
            cumsum[i] = regret[i]
        else:
            regret[i] = (((rwrds[opt_arm] - rwrds[arm]) + regret[i-1]))/float(i)
            cumsum[i] = cumsum[i-1] + regret[i]
        print "Iteration : ", i+1, "\tRegret : ", regret[i]

    plt.figure(1)
    plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
    plt.legend()
    plt.figure(2)
    plt.plot(np.arange(T), cumsum, color=curcolor, label=bandit.name())
    plt.legend()
    bnum = bnum + 1
plt.show()
