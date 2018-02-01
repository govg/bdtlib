import sys
import math
import numpy as np
from random import randint
import create_syntheticdata
import matplotlib.pyplot as plt

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random, LinUCB


def calc_frobnorm(A,B):	
	C = np.matrix(A) - np.matrix(B)
	f = np.linalg.norm(C)
	return f


def run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret, T, X, Y, narm, d, M, filename_plot_data):
	
	bn = 0
	bnum = 0
	# colors = ['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'green', 'green', 'green', 'green']
	# reward_type = "real"

	regret = np.zeros(T, dtype=np.float)
	avg_regret = np.zeros(T, dtype=np.float)
	# best_avg_regret = np.zeros((len(bandits),T), dtype=np.float)
	frob_norm = np.zeros(T+1, dtype=np.float)
	root_T = np.zeros(T, dtype=np.float)
	lin_T = np.zeros(T, dtype=np.float)

	
	
    # f = 50
  #   while f <= 1000:		
		
		# f += increment

		# factor = {'Random' : 1, 'Online Bootstrap' : f, 'Online Collaborative Bootstrap' : f, 'LinUCB' : f}
		# print factor		
		# f = calc_frobnorm(np.transpose(theta_true), bandit.get_params())
		# frob_norm[0] = f
		# sys.exit()
	for i in range(0, T):
		root_T[i] = 10000*math.sqrt(i+1)
		lin_T[i] = 100*(i+1) 

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
				bandit.update(context, Y[i][arm], exp_reward, reward_type, factor)
				# bandit.update(context, Y[i][arm], Y[i][opt_arm_in_hindsight])
				# if i % 1000 == 0:
				# 	print i, regret[i], avg_regret[i],  Y[i][arm], exp_reward, bandit.name()
				# print opt_arm_in_hindsight, arm, exp_reward, Y[i][arm]

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]					
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

			else:
				arm, exp_reward = bandit.get_random_arm(context)
				# exp_reward = bandit.get_exp_reward(context, arm)
				bandit.update(context, Y[i][arm], exp_reward, reward_type, factor)
				# bandit.update(context, Y[i][arm], Y[i][opt_arm_in_hindsight])

				if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
				else:
					regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]
			

		else:
			arm, exp_reward = bandit.choose(context)
			bandit.update(context, Y[i][arm], exp_reward, reward_type, factor)
			# if i % 1000 == 0:
			# 	print i, regret[i], avg_regret[i],  Y[i][arm], exp_reward, bandit.name()

			if i == 0:
					regret[i] = Y[i][opt_arm_in_hindsight] - Y[i][arm]
			else:
				regret[i] = (Y[i][opt_arm_in_hindsight] - Y[i][arm]) + regret[i-1]

		avg_regret[i] = regret[i] / (i+1)
		if i % 1000 == 0:
			print i, regret[i], avg_regret[i],  Y[i][arm], exp_reward, factor, alpha, bandit.name(), flag

	if avg_regret[i] < best_avg_regret[i]:
		best_avg_regret[:] = avg_regret[:]
		filename_best_avg_regret = filename_plot_data + bandit.name() + str(flag)
		np.save(filename_best_avg_regret, best_avg_regret)

	fp.write(bandit.name() + " regret = " + str(regret[i]) + " avg regret = " + str(avg_regret[i]) + " factor = " + str(factor) + " M = " + str(M) + " alpha = " + str(alpha) + "\n")
	fp.flush()
	
	return best_avg_regret[i]
	# fp.flush()			
    
	# fp.close()
	# return best_avg_regret
	

# curcolor = colors[bnum]
# plt.figure(0)
# plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
# plt.legend()
# plt.title('Regret')
# bnum = (bnum + 1) 

# curcolor = colors[bnum]
# plt.figure(1)
# plt.plot(np.arange(T), regret - lin_T, color=curcolor, label=bandit.name())
# plt.legend()
# plt.title('Regret - rT')
# bnum = (bnum + 1) 

# curcolor = colors[bnum]
# plt.figure(2)
# plt.plot(np.arange(T), regret - root_T, color=curcolor, label=bandit.name())
# plt.legend()
# plt.title('Regret - r(T^0.5)')
# bnum = (bnum + 1) 
# plt.show()