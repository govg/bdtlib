import sys
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt


sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, Random


# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 20
U = 100
N = 500000
K = 2
X = np.zeros((N,U+d+1))
# theta_true = np.random.randn(U+d+1, K)
mean = np.zeros((U+d+1))
cov = 100*np.identity((U+d+1))
# print mean.shape
# print cov.shape
# sys.exit(0)
theta_true = []

# print theta.shape
for i in range(K):
	theta = np.random.multivariate_normal(mean=mean,cov=cov)
	theta_true.append(theta)

theta_true = np.array(theta_true).transpose()
# print theta_true.shape
# sys.exit(0)



Y = []

for i in range(N):	
	user = randint(0, U - 1)
	X[i][user+d] = 1
	x = np.random.randn(1, d)	
	X[i, : d] = x
	X[i][U+d] = 1   
	true_rating = np.array(np.matrix(X[i,:]) * np.matrix(theta_true)) + np.random.randn(1, K)
	Y.append(np.squeeze(true_rating))


Y = np.array(Y)
# print X
# sys.exit(0)
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

bandits = [Random(narm=narm), OnlineBootstrap(B=10, narm=narm, d=U+d+1)]
# bandits = [OnlineBootstrap(B=1, narm=narm, d=U+d+1)]


bnum = 0
colors = ['red', 'blue', 'green', 'black']

regret = np.zeros(T, dtype=np.float)
frob_norm = np.zeros(T+1, dtype=np.float)
root_T = np.zeros(T, dtype=np.float)
lin_T = np.zeros(T, dtype=np.float)

for bandit in bandits:
    # f = calc_frobnorm(np.transpose(theta_true), bandit.get_params())
    # frob_norm[0] = f
    # sys.exit()
    for i in range(0, T):
		root_T[i] = math.sqrt(i)
		lin_T[i] = i

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
				# bandit.update(context, Y[i][arm], Y[i][opt_arm_in_hindsight])

				# print i, opt_arm_in_hindsight, arm, Y[i][arm], exp_reward
				# print opt_arm_in_hindsight, arm, exp_reward, Y[i][arm]

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			else:
				arm, exp_reward = bandit.get_random_arm(context)
				# exp_reward = bandit.get_exp_reward(context, arm)
				bandit.update(context, Y[i][arm], exp_reward)
				# bandit.update(context, Y[i][arm], Y[i][opt_arm_in_hindsight])

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			# frob_norm[i+1] = calc_frobnorm(np.transpose(theta_true), bandit.get_params())
			# root_T[i] = 2000*math.sqrt(i)
			# lin_T[i] = i

		else:
			arm = bandit.choose()
			reward = -1
			bandit.update(context, reward)

			if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
			else:
				regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			# print i, opt_arm_in_hindsight, arm, Y[i][arm], Y[i][opt_arm_in_hindsight]




    # print regret
    # print lin_T
    # print regret - lin_T
    curcolor = colors[bnum]
    plt.figure(0)
    plt.plot(np.arange(T), regret - lin_T, color=curcolor, label=bandit.name() + '1')
    plt.legend()
    bnum = (bnum + 1) % 4

    curcolor = colors[bnum]
    plt.figure(0)
    plt.plot(np.arange(T), regret - root_T, color=curcolor, label=bandit.name() + '2')
    plt.legend()
    bnum = (bnum + 1) % 4

    # sys.exit(0)
    # curcolor = colors[bnum]
    # # plt.figure(bnum)
    # plt.plot(np.arange(T), root_T, color=curcolor, label='root T')
    # plt.legend()
    # bnum = (bnum + 1) % 4

    # curcolor = colors[bnum]
    # # plt.figure(bnum)
    # plt.plot(np.arange(T), lin_T, color=curcolor, label='T')
    # plt.legend()
    # bnum = (bnum + 1) % 4

    # curcolor = colors[bnum]
    # plt.figure(bnum)
    # plt.plot(np.arange(T+1), frob_norm, color=curcolor, label='frob_norm')
    # plt.legend()
    # bnum = (bnum + 1) % 4

    
# curcolor = colors[bnum]
# # plt.figure(bnum)
# plt.plot(np.arange(T), root_T, color=curcolor, label='root T')
# plt.legend()
# bnum = (bnum + 1) % 4

# curcolor = colors[bnum]
# # plt.figure(bnum)
# plt.plot(np.arange(T), lin_T, color=curcolor, label='T')
# plt.legend()
# bnum = (bnum + 1) % 4

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
