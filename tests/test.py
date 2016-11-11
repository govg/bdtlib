import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn import linear_model
from numpy.random import multivariate_normal


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
noise = np.random.randint(100, size=(N, 1))
Y = Y + noise
'''

T = 50
narm = 25

# Variables for TS
B = np.matrix(np.identity(d))
mu = np.matrix(np.zeros(d))
# mu = np.matrix(np.random.randint(10,size=(d,1)))
f = np.matrix(np.zeros(d))
nu = 0.05
Binv = np.linalg.inv(B)

# Variables for LinUCB
ucb_lambda = 5
ucb_alpha = 1
ucb_A = np.matrix(np.identity(d)) * ucb_lambda
ucb_b = np.matrix(np.zeros((d, 1)))
ucb_t = np.matrix(np.zeros((d, 1)))
ucb_Ainv = np.matrix(np.identity(d))
firstterm = np.zeros((narm, 1))

# Switch to epsilon greedy
epsilon = 0

# Variables for Baseline RR
clf = linear_model.Ridge()
Xsub = X[0:6000, :]
Ysub = Y[0:6000]
clf.fit(Xsub, Ysub)

# Storing errors
predlr = np.zeros(T, dtype=np.float)
predts = np.zeros(T, dtype=np.float)
predlu = np.zeros(T, dtype=np.float)

regretlr = np.zeros(T, dtype=np.float)
regretts = np.zeros(T, dtype=np.float)
regretlu = np.zeros(T, dtype=np.float)

for i in range(0, T):
    # Common to all algorithms
    arms = np.random.randint(N, size=narm)
    cts = X[arms, :]
    rwrds = Y[arms]

    # First, we compute using Thompson Sampling
    # Draw from posterior
    muarray = np.squeeze(np.asarray(mu))
    mut = multivariate_normal(muarray, nu*nu*Binv)

    # Compute reward, choose maximum
    ts_rewards = np.matrix(cts)*(np.matrix(mut).transpose())
    ts_arm = np.argmax(ts_rewards)

    # Update distribution
    b = np.matrix(cts[ts_arm, :])
    B = B + b.transpose()*b
    f = f + rwrds[ts_arm]*cts[ts_arm, :]
    Binv = np.linalg.inv(B)
    mu = Binv * f.transpose()

    # Second, we implement the LinUCB algorithm
    for k in range(0, narm):
        Xk = np.matrix(cts[k, :])
        firstterm[k, 0] = Xk * ucb_Ainv * Xk.transpose()

    firstterm = np.sqrt(firstterm) * ucb_alpha
    secondterm = np.matrix(cts) * np.matrix(ucb_t)
    ucb_rewards = firstterm + secondterm
    ucb_arm = np.argmax(ucb_rewards)
    if np.random.randint(narm) < epsilon*narm:
        ucb_arm = np.random.randint(narm)
    else:
        ucb_arm = ucb_arm

    # Update the LinUCB model
    x = np.matrix(cts[ucb_arm, :])
    ucb_A = ucb_A + x.transpose()*x
    ucb_b = ucb_b + (rwrds[ucb_arm]*x).transpose()
    ucb_Ainv = np.linalg.inv(ucb_A)
    ucb_t = ucb_Ainv*ucb_b

    # We use our baseline ridge regression formula
    lr_rewards = clf.predict(cts)
    lr_arm = np.argmax(lr_rewards)

    # To compute actual regret, we find max possible reward
    opt_arm = np.argmax(rwrds)

    predts[i] = rwrds[ts_arm] - ts_rewards[ts_arm]
    predlr[i] = rwrds[lr_arm] - lr_rewards[lr_arm]
    predlu[i] = rwrds[ucb_arm] - ucb_rewards[ucb_arm]

    if i == 0:
        regretts[i] = rwrds[opt_arm] - rwrds[ts_arm]
        regretlr[i] = rwrds[opt_arm] - rwrds[lr_arm]
        regretlu[i] = rwrds[opt_arm] - rwrds[ucb_arm]
    else:
        regretts[i] = ((rwrds[opt_arm] - rwrds[ts_arm]) + regretts[i-1])/(i+1)
        regretlr[i] = ((rwrds[opt_arm] - rwrds[lr_arm]) + regretlr[i-1])/(i+1)
        regretlu[i] = ((rwrds[opt_arm] - rwrds[ucb_arm]) + regretlu[i-1])/(i+1)

    # print "Run number", i+1
    # print "TS predicted reward of : ", ts_rewards[ts_arm]
    # print "LR predicted reward of : ", lr_rewards[ucb_arm]
    # print "LinUCB predicted reward of : ", ucb_rewards[ucb_arm]
    # print "Actual reward was : ", rwrds[opt_arm]


# plt.plot(np.arange(T), regret, color='blue')
plt.figure(1)
plt.plot(np.arange(T), predts, color='red', label="Thompson Sampling")
plt.plot(np.arange(T), predlr, color='blue', label="Baseline RR")
plt.plot(np.arange(T), predlu, color='green', label="LinUCB")
plt.plot(np.arange(T), np.zeros(T), color='black')
plt.legend()
plt.figure(2)
plt.plot(np.arange(T), regretts, color='red', label="Thompson Sampling")
plt.plot(np.arange(T), regretlr, color='blue', label="Baseline RR")
plt.plot(np.arange(T), regretlu, color='green', label="LinUCB")
plt.legend()
plt.show()
