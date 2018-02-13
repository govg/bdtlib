import numpy as np
import sys
import time

import create_syntheticdata
from testrecommendation1 import run_recommender

sys.path.append('../bdtlib')
from bdtlib.reco import OnlineBootstrap, OnlineCollaborativeBootstrap, Random, LinUCB

# Toy data, to play with
# Context vector is 1-hot depicting the current user
d = 0
U = 100
N = 10000
K = 200
M_true = 100

# filename_context = 'cleaned_data2/contexts_synthetic_real3'
# filename_rating = 'cleaned_data2/ratings_synthetic_real3'
# filename_theta_true = 'cleaned_data2/theta_true_synthetic3'
#
# X, Y, theta_true = create_syntheticdata.create_data_independent(d=d, U=U, N=N, K=K)

filename_context = 'cleaned_data2/contexts_synthetic_real_dep5'
filename_rating = 'cleaned_data2/ratings_synthetic_real_dep5'
filename_theta_true = 'cleaned_data2/theta_true_synthetic_dep5'

X, Y, theta_true = create_syntheticdata.create_data_dependent(d=d, U=U, N=N, K=K, M=M_true)
#
#
np.save(filename_context, X)
np.save(filename_rating, Y)
np.save(filename_theta_true, theta_true)


# X = np.load(filename_context + '.npy')
# Y = np.load(filename_rating + '.npy')
# theta_true = np.load(filename_theta_true + '.npy')


def run_exp(X, Y, K, d):
	bn = 0
	narm = K
	T = N
	num_bandits = 4
	reward_type = "real"
	filename_result = 'result_cleaned/result_sparse8.txt'
	# filename_result = 'results/result_independent_real.txt'
	# best_avg_regret = np.zeros((num_bandits,T), dtype=np.float)
	best_avg_regret = np.full((num_bandits, T), 99999999999)
	filename_plot_data = 'plots_cleaned/8_best_avg_regret_sparse_'
	fp = open(filename_result, 'a')
	fp.write("Dependent Data\n")
	flag = 2
	while bn < num_bandits:
		bn += 1

		if bn == 1:
			# continue
			factor = 0
			alpha = 0
			M = 0
			
			start_time = time.time()
			for i in range(10):
				bandit = Random(narm=narm)
				run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M, filename_plot_data)
			avg_duration = float((time.time() - start_time) / 10.0)
			fp.write('Avg Duration = ' + str(avg_duration) + '\n')
			fp.flush()


		elif bn == 2:
			# continue
			alpha = 0
			M = 0
			
			start_time = time.time()
			c = 0
			for factor in range(5, 1030, 25):
				bandit = OnlineBootstrap(B=20, narm=narm, d=U+d+1)
				run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M, filename_plot_data)
				c += 1

			avg_duration = float(1.0*(time.time() - start_time) / c)
			fp.write('Avg Duration = ' + str(avg_duration) + '\n')
			fp.flush()


		elif bn == 3:
			# continue
			best_factor = 50
			alpha = 0
			M = int(narm / 2)
			best_regret = 99999999999
			best_M = M
			c = 0
			start_time = time.time()
			for factor in range(5, 1030, 25):
				bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=U+d+1, M=M)
				r = run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M, filename_plot_data)
				c += 1

				if r < best_regret:
					best_regret = r
					best_factor = factor

			for m in range(1, narm, 10):
				bandit = OnlineCollaborativeBootstrap(B=1, narm=narm, D=U+d+1, M=m)
				r = run_recommender(fp, flag, bandit, reward_type, best_factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, m, filename_plot_data)
				c += 1

				if r < best_regret:
					best_regret = r
					best_M = m

			print best_factor
			print best_M
			avg_duration = float(1.0*(time.time() - start_time) / c)
			fp.write('Avg Duration = ' + str(avg_duration) + '\n')
			fp.flush()
			
		elif bn == 4:
			M = 0
			factor = 0
			start_time = time.time()
			c = 0
			alpha = 0.5			
			while alpha <= 10:
				# pass
				bandit = LinUCB(alpha=alpha, d=U+d+1, sigma=1, narm=narm)
				r = run_recommender(fp, flag, bandit, reward_type, factor, alpha, best_avg_regret[bn-1][:], T, X, Y, narm, d, M, filename_plot_data)
				alpha += 1
				c += 1

			avg_duration = float(1.0*(time.time() - start_time) / c)
			fp.write('Avg Duration = ' + str(avg_duration) + '\n')
			fp.flush()
		

	fp.close()

def main():
	run_exp(X, Y, K, d)

if __name__ == "__main__":
	main()
