import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, LinUCB, Random

def create_context_vector(user_id, timestamp, d):
	vec_len = d
	vec = np.zeros(vec_len)

	index = 0

	user_id_bin = bin(user_id)[2:]
	len_bin = len(user_id_bin)
	index = 17 - len_bin
	for i in range(len(user_id_bin)):
		vec[index] = int(user_id_bin[i])
		index += 1

	timestamp_bin = bin(timestamp)[2:]
	len_bin = len(timestamp_bin)
	index += (31 - len_bin)
	for i in range(len(timestamp_bin)):
		vec[index] = int(timestamp_bin[i])
		index += 1

	return vec


def test_movielens10M():
	d = 48
	K = 10681
	bandits = [Random(narm=K), OnlineBootstrap(B=1, narm=K, d=d), OnlineCollaborativeBootstrap(B=1, narm=K, D=d, M=int(K/50))]

	bnum = 0
	colors = ['red', 'red', 'red', 'blue', 'blue', 'blue']

	for bandit in bandits:
		print bandit.name()
		ratio = []
		T = 0
		n = 0
		cum_reward = 0
		filename_read = '../data/ml-10M100K/ratings_time.dat'
		fp = open(filename_read, 'r')
		line = fp.readline()
		f = open(bandit.name()+'_movielens10M.txt', 'w')
		while line:
			temp = line.strip('\n').split('::')
			user_id = int(temp[0])
			movie_id = int(temp[1])
			true_rating = float(temp[2])
			timestamp = int(temp[3])
			line = fp.readline()
			n += 1

			context = create_context_vector(user_id, timestamp, d)
			
			arm, exp_reward = bandit.choose(context)
			if arm != movie_id - 1:
				continue
			
			bandit.update(context, true_rating, exp_reward)
			cum_reward += true_rating
			T += 1

			print user_id, movie_id, true_rating, exp_reward, float(1.0*cum_reward / T), T, n, timestamp
			ratio.append(float(1.0*cum_reward / T))
			f.write(str(cum_reward) + ' ' + str(T) + ' ' + str(float(1.0*cum_reward / T)) + '\n')
			

		ratio = np.array(ratio)
		ratio.dump(bandit.name())
		# curcolor = colors[bnum]
		# plt.figure(0)
		# plt.plot(np.arange(1,T+1), ratio, color=curcolor, label=bandit.name())
		# plt.legend()
		# bnum += 1

		f.close()
		fp.close()

# plt.show()
