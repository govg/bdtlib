import sys
import math
import numpy as np
from random import randint
import create_syntheticdata
import matplotlib.pyplot as plt

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random


# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 0
U = 150
N = 50000
K = 100
M = 20
filename_context = 'cleaned_data/contexts_synthetic_real_dep'
filename_rating = 'cleaned_data/ratings_synthetic_real_dep'
filename_theta_true = 'cleaned_data/theta_true_synthetic_dep'
# X = np.zeros((N,U+d+1))

# X, Y, theta_true = create_syntheticdata.create_data_independent(d=d, U=U, N=N, K=K)
# np.save(filename_context, X)
# np.save(filename_rating, Y)
# np.save(filename_theta_true, theta_true)

# X, Y, theta_true = create_syntheticdata.create_data_dependent(d=d, U=U, N=N, K=K, M=M)
# np.save(filename_context, X)
# np.save(filename_rating, Y)
# np.save(filename_theta_true, theta_true)


X = np.load(filename_context + '.npy')
Y = np.load(filename_rating + '.npy')
theta_true = np.load(filename_theta_true + '.npy')


def calc_frobnorm(A,B):	
	C = np.matrix(A) - np.matrix(B)
	f = np.linalg.norm(C)
	return f


T = N
narm = K

# bandits = [Random(narm=narm), OnlineBootstrap(B=10, narm=narm, d=U+d+1)]
# bandits = [OnlineBootstrap(B=1, narm=narm, d=U+d+1)]
# bandits = [OnlineCollaborativeBootstrap(B=1, narm=narm, D=U+d+1, M=U+d+1)]
bandits = [Random(narm=narm), OnlineBootstrap(B=10, narm=narm, d=U+d+1), OnlineCollaborativeBootstrap(B=1, narm=narm, D=U+d+1, M=M)]
factor = {'Random' : 1, 'Online Bootstrap' : 300, 'Online Collaborative Bootstrap' : 0.9}

bnum = 0
colors = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'green', 'green', 'green']
reward_type = "real"

regret = np.zeros(T, dtype=np.float)
frob_norm = np.zeros(T+1, dtype=np.float)
root_T = np.zeros(T, dtype=np.float)
lin_T = np.zeros(T, dtype=np.float)

fp = open('results/result_synthetic_dep.txt', 'a')
for bandit in bandits:
    print bandit.name()
    # f = calc_frobnorm(np.transpose(theta_true), bandit.get_params())
    # frob_norm[0] = f
    # sys.exit()
    for i in range(0, T):
		root_T[i] = 10000*math.sqrt(i)
		lin_T[i] = 100*i

    	# Get context
		context = X[i,:]		

		# Find the optimum arm in hindsight
		true_rewards = Y[i, :]
		opt_arm_in_hindsight = np.argmax(true_rewards)

		# print "opt arm : " + str(opt_arm_in_hindsight)

		if bandit.name() == "Online Bootstrap" or bandit.name() == "Online Collaborative Bootstrap":
			if i >= 300:		
				# Pull arm as per Online Bootstrap
				arm, exp_reward = bandit.choose(context)
				bandit.update(context, Y[i][arm], exp_reward, reward_type, factor[bandit.name()])
				# bandit.update(context, Y[i][arm], Y[i][opt_arm_in_hindsight])

				print i, opt_arm_in_hindsight, arm, Y[i][arm], exp_reward
				# print opt_arm_in_hindsight, arm, exp_reward, Y[i][arm]

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			else:
				arm, exp_reward = bandit.get_random_arm(context)
				# exp_reward = bandit.get_exp_reward(context, arm)
				bandit.update(context, Y[i][arm], exp_reward, reward_type, factor[bandit.name()])
				# bandit.update(context, Y[i][arm], Y[i][opt_arm_in_hindsight])

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]
			

		else:
			arm, exp_reward = bandit.choose(context)
			reward = -1
			bandit.update(context, reward, exp_reward, reward_type, factor[bandit.name()])

			if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
			else:
				regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

	
    fp.write(bandit.name() + " regret = " + str(regret[i]) + "\n")
    curcolor = colors[bnum]
    plt.figure(0)
    plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
    plt.legend()
    plt.title('Regret')
    bnum = (bnum + 1) 

    curcolor = colors[bnum]
    plt.figure(1)
    plt.plot(np.arange(T), regret - lin_T, color=curcolor, label=bandit.name())
    plt.legend()
    plt.title('Regret - rT')
    bnum = (bnum + 1) 

    curcolor = colors[bnum]
    plt.figure(2)
    plt.plot(np.arange(T), regret - root_T, color=curcolor, label=bandit.name())
    plt.legend()
    plt.title('Regret - r(T^0.5)')
    bnum = (bnum + 1) 

plt.show()
fp.close()