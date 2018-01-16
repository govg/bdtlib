import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, LinUCB, Random

def create_context_vector(user_id, user_attribute, d):
	vec_len = d
	vec = np.zeros(vec_len)

	gender = user_attribute[0]
	age = user_attribute[1]
	occupation = user_attribute[2]
	timestamp = user_attribute[3]

	# print user_id, gender, age, occupation, timestamp
	index = 0

	user_id_bin = bin(user_id)[2:]
	len_bin = len(user_id_bin)
	index = 13 - len_bin
	for i in range(len(user_id_bin)):
		vec[index] = int(user_id_bin[i])
		index += 1

	if gender=='M':
		vec[index] = 1
		index += 1

	age_bin = bin(age)[2:]
	len_bin = len(age_bin)
	index += (6 - len_bin)
	for i in range(len(age_bin)):
		vec[index] = int(age_bin[i])
		index += 1

	occupation_bin = bin(occupation)[2:]
	len_bin = len(occupation_bin)
	index += (5 - len_bin)
	for i in range(len(occupation_bin)):
		vec[index] = int(occupation_bin[i])
		index += 1

	timestamp_bin = bin(timestamp)[2:]
	len_bin = len(timestamp_bin)
	index += (30 - len_bin)
	for i in range(len(timestamp_bin)):
		vec[index] = int(timestamp_bin[i])
		index += 1

	return vec

def create_user_attribute_mapping():
	filename_read = '../data/ml-1m/users.dat'
	fp = open(filename_read, 'r')
	mapping = {}

	line = fp.readline()
	while line:
		temp = line.strip('\n').split('::')
		user_id = int(temp[0])
		mapping[user_id] = []
		mapping[user_id].append(temp[1])
		mapping[user_id].append(int(temp[2]))
		mapping[user_id].append(int(temp[3]))

		line = fp.readline()

	return mapping

user_attribute_mapping = create_user_attribute_mapping()

d = 55
K = 3952
# bandits = [LinUCB(alpha=25,d=d,sigma=10,narm=K)]
bandits = [Random(narm=K), OnlineBootstrap(B=1, narm=K, d=d), OnlineCollaborativeBootstrap(B=1, narm=K, D=d, M=int(K/10)) ]
# bandits = [Random(narm=K), OnlineCollaborativeBootstrap(B=1, narm=K, D=d, M=int(K/15))]
# bandits = [Random(narm=K), OnlineBootstrap(B=1, narm=K, d=d)]
# bandit = OnlineBootstrap(B=1, narm=K, d=d)
bnum = 0
colors = ['red', 'red', 'red', 'blue', 'blue', 'blue']

for bandit in bandits:
	print bandit.name()
	ratio = []
	cum_reward = 0
	T = 0
	n = 0
	filename_read = '../data/ml-1m/ratings_time.dat'
	fp = open(filename_read, 'r')
	line = fp.readline()
	f = open(bandit.name()+'_movielens1M.txt', 'w')
	while line:
		temp = line.strip('\n').split('::')
		user_id = int(temp[0])
		movie_id = int(temp[1])
		true_rating = int(temp[2])
		timestamp = int(temp[3])
		line = fp.readline()
		n += 1

		gender = user_attribute_mapping[user_id][0]
		age = user_attribute_mapping[user_id][1]
		occupation = user_attribute_mapping[user_id][2]

		user_attribute = [gender, age, occupation, timestamp]
		context = create_context_vector(user_id, user_attribute, d)
		
		arm, exp_reward = bandit.choose(context)
		if arm != movie_id - 1:
			continue
		
		bandit.update(context, true_rating, exp_reward)
		cum_reward += true_rating
		T += 1

		print user_id, movie_id, true_rating, exp_reward, float(1.0*cum_reward / T), n, timestamp
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

plt.show()
