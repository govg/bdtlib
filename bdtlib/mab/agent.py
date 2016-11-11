
'''
Describes agents for MAB without context

Each agent provides the following two methods :
    choose ()   :   choose an arm from the given contexts. Returns
                    the index of the arm to be played
    update()    :   Updates the model parameters given the context
                    and the reward
'''

import numpy as np
'''
Will contain the following policies :
    LinUCB
    Random
    Epsilon Greedy

'''


class EpsGreedy():
    def __init__(self, epsilon, d):
        self.epsilon = epsilon
        self.narms = d
        self.means = np.zeros(self.narms)
        self.attempts = np.zeros(self.narms)
        self.total = 0
        return

    def choose(self, arms):
        # If we haven't attempted enough arms, we'll blindly
        # attempt one
        if self.attempts.sum() < self.epsilon * self.narms:
            optarm = np.random.randint(arms.shape[0])
        else:
            # Attempt the first new arm, or exploit best arm so far
            if np.random.randint(10) < self.epsilon * 10:
                optarm = np.random.randint(arms.shape[0])
                for i in range(0, arms.shape[0]):
                    if self.attempts[arms[i]] == 0:
                        optarm = i
                        break
            else:
                optarm = np.argmax(self.means[arms])

        return arms[optarm]

    def update(self, arm, reward):
        # Update the number of attempts and the mean rewards
        self.attempts[arm] = self.attempts[arm] + 1
        self.means[arm] = self.means[arm] * (self.attempts[arm]-1)
        self.means[arm] = self.means[arm] + reward
        self.means[arm] = self.means[arm] / self.attempts[arm]
        self.total = self.total + 1
        return

    def name(self):
        return "Epsilon Greedy"


class LinUCB():
    def __init__(self, d):
        self.narms = d
        self.means = np.zeros((self.narms, 1))
        self.attempts = np.zeros((self.narms, 1))
        self.total = 0
        return

    def choose(self, arms):
        # Compute the confidence intervals for all the arms
        confid = np.zeros(arms.shape[0])
        for i in range(0, arms.shape[0]):
            # We need to play each arm at least once to attempt this
            if self.attempts[arms[i]] == 0:
                optarm = i
                return arms[optarm]
            else:
                confid[i] = (2*np.log(self.total))/self.attempts[arms[i]]
                confid[i] = np.sqrt(confid[i]) + self.means[arms[i]]
        optarm = np.argmax(confid)
        return arms[optarm]

    def update(self, arm, reward):
        # Update the number of attempts and the mean rewards
        self.attempts[arm] = self.attempts[arm] + 1
        self.means[arm] = self.means[arm] * (self.attempts[arm]-1)
        self.means[arm] = self.means[arm] + reward
        self.means[arm] = self.means[arm] / self.attempts[arm]
        self.total = self.total + 1
        return

    def name(self):
        return "LinUCB"


class Random():
    # Best policy ever
    def __init__(self):
        return

    def choose(self, arms):
        optarm = np.random.randint(arms.shape[0])
        return arms[optarm]

    def update(self, arm, reward):
        return

    def name(self):
        return "Random"
