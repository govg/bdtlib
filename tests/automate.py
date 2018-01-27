import sys
import math
import numpy as np
from random import randint
import create_syntheticdata
from testrecommendation1 import run_recommender

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random, LinUCB

# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 0
U = 10
N = 20000
K = 30
M = 6

filename_context = 'cleaned_data/contexts_synthetic_real'
filename_rating = 'cleaned_data/ratings_synthetic_real'
filename_theta_true = 'cleaned_data/theta_true_synthetic'

X, Y, theta_true = create_syntheticdata.create_data_independent(d=d, U=U, N=N, K=K)

# filename_context = 'cleaned_data/contexts_synthetic_real_dep'
# filename_rating = 'cleaned_data/ratings_synthetic_real_dep'
# filename_theta_true = 'cleaned_data/theta_true_synthetic_dep'

# X, Y, theta_true = create_syntheticdata.create_data_dependent(d=d, U=U, N=N, K=K, M=M)


np.save(filename_context, X)
np.save(filename_rating, Y)
np.save(filename_theta_true, theta_true)


# X = np.load(filename_context + '.npy')
# Y = np.load(filename_rating + '.npy')
# theta_true = np.load(filename_theta_true + '.npy')




def run_exp(X, Y, K, d, M):
	bn = 0
	narm = K
	T = N
	num_bandits = 4
	reward_type = "real"
	filename_result = 'results/dummy.txt'
	# filename_result = 'results/result_independent_real.txt'
	best_avg_regret = np.zeros((num_bandits,T), dtype=np.float)

	fp = open(filename_result, 'a')

	while bn < num_bandits:
		bn += 1

		# if bn == 1:
		# 	factor = 0
		# 	alpha = 0
		# 	M = 0
					
		# 	for i in range(10):
		# 		bandit = Random(narm=narm)
		# 		run_recommender(fp, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M)


		# elif bn == 2:
		# 	alpha = 0
		# 	M = 0
			

		# 	for factor in range(50, 1025, 25):
		# 		bandit = OnlineBootstrap(B=20, narm=narm, d=U+d+1)
		# 		run_recommender(fp, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M)


		# elif bn == 3:
		# 	best_factor = 50
		# 	alpha = 0
		# 	M = int(narm / 2)
		# 	best_regret = 99999999999
		# 	best_M = M
			
		# 	for factor in range(75, 1025, 25):
		# 		bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=U+d+1, M=M)
		# 		r = run_recommender(fp, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M)
		# 		if r < best_regret:
		# 			best_regret = r
		# 			best_factor = factor

		# 	for m in range(1, narm):
		# 		bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=U+d+1, M=m)
		# 		r = run_recommender(fp, bandit, reward_type, best_factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M)

		# 		if r < best_regret:
		# 			best_regret = r
		# 			best_M = m

		# 	print best_factor
		# 	print best_M
			
		if bn == 4:
			M = 0
			factor = 0
			
			alpha = 0.5
			# for alpha in range(0.5, 50, 0.5):
			while alpha <= 5:
				# pass
				bandit = LinUCB(alpha=alpha, d=U+d+1, sigma=1, narm=narm)
				r = run_recommender(fp, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M)
				alpha += 0.5


		fp.flush()

	fp.close()

def main():
	run_exp(X, Y, K, d, M)

if __name__ == "__main__":
    main()