import numpy as np
import matplotlib.pyplot as plt

from bdtlib.mab import EpsGreedy, LinUCB, Random

# We'll work with 10 arms, 50 trials
N = 10
T = 100

# Rewards are randomly generated between 10 and 20
rewards = np.random.randint(5, 20, size=N)
barm = np.argmax(rewards)

# Initialise the bandits
bandits = [EpsGreedy(0.2, N), LinUCB(N), Random()]
regret = np.zeros(T)

bnum = 0
colors = ['red', 'blue', 'green']

for bandit in bandits:
    curcolor = colors[bnum]
    for i in range(0, T):
        # Add some noise to current rewards, could also be Gaussian noise
        noise = np.random.randint(3, size=N)
        currewards = rewards + noise
        armset = np.arange(N)
        currewards = currewards[armset]

        # Let the bandit choose the needed reward
        arm = bandit.choose(armset)
        optarm = np.argmax(currewards)
        bandit.update(arm, currewards[arm])

        # Update the regret for the bandit
        if i == 0:
            regret[i] = currewards[optarm] - currewards[armset[arm]]
        else:
            regret[i] = (currewards[optarm] - currewards[armset[arm]]) 
            regret[i] = regret[i] + regret[i-1]

    plt.plot(np.arange(T), regret, color=curcolor, label=bandit.name())
    plt.legend()
    bnum = bnum + 1
plt.show()
