import sys
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt


sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, Random


# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 50
U = 100
N = 500000
K = 50
X = np.zeros((N,U+d))
theta_true = np.random.randn(U+d, K)
Y = []
for i in range(N):
    user = randint(0, U - 1)
    X[i][user+d] = 1
    # x = np.random.randn(1, d)
    # X[i, : d] = x   
    true_rating = np.array(np.matrix(X[i,:]) * np.matrix(theta_true)) + np.random.randn(1, K)
    Y.append(np.squeeze(true_rating))

Y = np.array(Y)
# print X
# print theta_true.shape
# print Y.shape
# print Y
# w = np.random.randint(10, size=(d, 1))
# Y = np.matrix(X)*np.matrix(w)
# noise = np.random.randint(10, size=(N, 1))
# Y = Y + noise

def calc_frobnorm(A,B):	
	C = np.matrix(A) - np.matrix(B)
	f = np.linalg.norm(C)
	return f


T = N
# narm = 20
narm = K

# bandits = [Random(narm=narm), OnlineBootstrap(B=10, narm=narm, d=U+d)]
bandits = [OnlineBootstrap(B=1, narm=narm, d=U+d)]


bnum = 0
colors = ['red', 'blue']

regret = np.zeros(T, dtype=np.float)
frob_norm = np.zeros(T+1, dtype=np.float)

for bandit in bandits:
    f = calc_frobnorm(np.transpose(theta_true), bandit.get_params())
    frob_norm[0] = f
    # sys.exit()
    for i in range(0, T):
    	# Get context
		context = X[i,:]
		# print context

		# Find the optimum arm in hindsight
		true_rewards = Y[i, :]
		opt_arm_in_hindsight = np.argmax(true_rewards)

		# print "opt arm : " + str(opt_arm_in_hindsight)

		if bandit.name() == "Online Bootstrap":
			if i >= 300:		
				# Pull arm as per Online Bootstrap
				arm, exp_reward = bandit.choose(context)
				bandit.update(context, Y[i][arm], exp_reward)
				print i, opt_arm_in_hindsight, arm, Y[i][arm], exp_reward
				# print opt_arm_in_hindsight, arm, exp_reward, Y[i][arm]

				# if i == 0:
					# regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				# else:
				regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			else:
				arm, exp_reward = bandit.get_random_arm(context)
				# exp_reward = bandit.get_exp_reward(context, arm)
				bandit.update(context, Y[i][arm], exp_reward)

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			frob_norm[i+1] = calc_frobnorm(np.transpose(theta_true), bandit.get_params())

		else:
			arm = bandit.choose()
			reward = -1
			bandit.update(context, reward)

			if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
			else:
				regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]


    curcolor = colors[bnum]
    plt.figure(1)
    plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
    plt.legend()
    bnum = bnum + 1

    curcolor = colors[bnum]
    plt.figure(2)
    plt.plot(np.arange(T+1), frob_norm, color=curcolor, label='frob_norm')
    plt.legend()

plt.show()












#         # Choose a subset of datapoints as arms
        # arms = np.random.randint(N, size=narm)
#         cts = X[arms, :]
#         rwrds = Y[arms]

#         # Ask agent to choose an arm and update it
#         # with obtained reward
#         arm = bandit.choose(cts)
#         bandit.update(cts[arm, :], rwrds[arm])

#         # To compute actual regret, we find max possible reward
#         opt_arm = np.argmax(rwrds)

#         if i == 0:
#             regret[i] = rwrds[opt_arm] - rwrds[arm]
#         else:
#             regret[i] = ((rwrds[opt_arm] - rwrds[arm]) + regret[i-1])

#     # plt.figure(bnum+1)
#     plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
#     plt.legend()
#     bnum = bnum + 1
# plt.show()
