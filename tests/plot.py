import os
import sys
import numpy as np
import matplotlib.pyplot as plt

filename1 = 'plots/best_avg_regret_Random'
filename2 = 'plots/best_avg_regret_Online Bootstrap'
filename3 = 'plots/best_avg_regret_Online Collaborative Bootstrap'
filename4 = 'plots/best_avg_regret_LinUCB'

colors = ['red', 'blue', 'green', 'black']

T = 2000

random = np.load(filename1 + '.npy')
OnlBootstrap = np.load(filename2 + '.npy')
OnlCollabBootstrap = np.load(filename3 + '.npy')
linucb = np.load(filename4 + '.npy')

# bnum = 0

plt.figure(1)
plt.plot(np.arange(T), random, color='red', label="Random")
plt.plot(np.arange(T), OnlBootstrap, color='blue', label="Online Bootstrap")
plt.plot(np.arange(T), OnlCollabBootstrap, color='green', label="Online Collaborative Bootstrap")
plt.plot(np.arange(T), linucb, color='black', label="Linear UCB")


plt.legend()


plt.title('avg regret')
plt.show()