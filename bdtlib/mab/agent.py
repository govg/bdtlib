
'''
Describes agents for MAB without context

Each agent provides the following two methods :
    choose ()   :   choose an arm from the given contexts. Returns
                    the index of the arm to be played
    update()    :   Updates the model parameters given the context
                    and the reward
'''

import numpy as np
from numpy.random import multivariate_normal
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
        self.arms = np.zeros(self.narms)
        self.attempts = np.zeros(self.narms)
        return

    def choose(self, arms):

        if self.attempts.sum() < self.epsilon * self.narms:
            optarm = arms[np.random.randint(arms.shape[0])]
        else:
            if np.random.randint(10) < self.epsilon * 10:
                optarm = np.random.randint(arms.shape[0])
            else:
                optarm = np.argmax(self.arms[arms])

        return arms[optarm]

    def update(self, arm, reward):
        self.attempts[arm] = self.attempts[arm] + 1
        self.arms[arm] = (self.arms[arm] + reward)/self.attempts[arm]
        return

    def name(self):
        return "Epsilon Greedy"


class LinUCB():
    def __init__(self, nu, d):
        self.narms = d
        self.arms = np.zeros((self.narms, 1))
        self.attempts = np.zeros((self.narms, 1))
        return

    def choose(self, arms):

        return

    def update(self, arm, reward):
        return

    def name(self):
        return "LinUCB"


class Random():
    def __init__(self):

        return

    def choose(self, arms):
        optarm = np.random.randint(arms.shape[0])
        return arms[optarm]

    def update(self, arm, reward):
        return

    def name(self):
        return "Random"
