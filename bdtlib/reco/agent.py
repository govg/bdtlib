'''
Describes agents for MAB with and without context

Each agent provides the following two methods :
    choose ()   :   choose an arm from the given contexts. Returns
                    the index of the arm to be played
    update()    :   Updates the model parameters given the context
                    and the reward
'''
import time
import math
import random
import numpy as np
from numpy.random import multivariate_normal
'''
Will contain the following policies :   
    Thompson Sampling   :   Updates using Bayes rule, generates weight
                            vector using distribution
    LinUCB              :   Similar to normal LinUCB rule
    Random              :   Randomly select an arm
    Eps-Greedy          :   Epsilon greedy selection of arm
    Bootstrap           :   Bootstrap method for arm selection

'''


class ThompsonSampling():
    def __init__(self, nu, d):
        self.d = d
        self.B = np.matrix(np.identity(d))
        self.mu = np.matrix(np.zeros(d))
        self.f = np.matrix(np.zeros(d))
        self.nu = nu
        self.Binv = np.matrix(np.identity(d))

        return

    def choose(self, contexts):
        muarray = np.squeeze(np.asarray(self.mu))
        mut = multivariate_normal(muarray, self.nu*self.nu*self.Binv)

        rewards = np.matrix(contexts) * (np.matrix(mut).transpose())
        optarm = np.argmax(rewards)

        return optarm

    def update(self, context, reward):
        b = np.matrix(context)
        self.B = self.B + b.transpose()*b
        self.f = self.f + reward*context
        self.Binv = np.linalg.inv(self.B)
        self.mu = self.Binv * self.f.transpose()
        return
    
    def name(self):
        return "Thompson Sampling"


class LinUCB():
    def __init__(self, alpha, d):
        self.d = d
        self.A = np.matrix(np.identity(d))
        self.b = np.matrix(np.zeros((d, 1)))
        self.t = np.matrix(np.zeros((d, 1)))
        self.alpha = alpha
        self.Ainv = np.matrix(np.identity(d))

    def choose(self, contexts):
        narm = contexts.shape[0]
        first = np.zeros((narm, 1))

        for k in range(0, narm):
            Xk = np.matrix(contexts[k, :])
            first[k, 0] = Xk * self.Ainv * Xk.transpose()

        first = np.sqrt(first) * self.alpha
        second = np.matrix(contexts) * np.matrix(self.t)
        rewards = first + second
        optarm = np.argmax(rewards)

        return optarm

    def update(self, context, reward):
        x = np.matrix(context)
        self.A = self.A + x.transpose()*x
        self.b = self.b + (reward*x).transpose()
        self.Ainv = np.linalg.inv(self.A)
        self.t = self.Ainv*self.b
        return

    def name(self):
        return "LinUCB"



class EpsGreedy():
    def __init__(self, epsilon, d, alpha):
        self.epsilon = epsilon
        self.d = d
        self.A = np.matrix(np.identity(d))
        self.b = np.matrix(np.zeros(d))
        self.t = np.matrix(np.zeros(d))
        self.alpha = alpha
        self.Ainv = np.matrix(np.identity(d))

    def choose(self, contexts):
        narm = contexts.shape[0]
        first = np.zeros((narm, 1))

        for k in range(0, narm):
            Xk = np.matrix(contexts[k, :])
            first[k, 0] = Xk * self.Ainv * Xk.transpose()

        first = np.sqrt(first) * self.alpha
        second = np.matrix(contexts) * np.matrix(self.t)
        rewards = first + second
        optarm = np.argmax(rewards)
        threshold = self.epsilon * narm
        if np.random.randint(narm) > threshold:
            return optarm
        else:
            return np.random.randint(narm)

    def update(self, context, reward):
        x = np.matrix(context)
        self.A = self.A + x.tranpose()*x
        self.b = self.b + (reward*x).transpose()
        self.Ainv = np.linalg.inv(self.A)
        self.t = self.Ainv*self.b
        return

    def name(self):
        return "Epsilon Greedy"


class Random():
    def __init__(self, narm=10):
        self.narm = narm
        # self.flag = flag

    def choose(self):
        # N = contexts.shape[0]
        # if self.flag:
        optarm = random.randint(0, self.narm - 1)
        # else:
            # optarm = self.arm
        return optarm

    def update(self, context, reward):
        return

    def name(self):
        return "Random"


class OnlineBootstrap():
    def __init__(self, B=1, narm=10, d=10):     
        self.B = B
        self.d = d
        self.narm = narm

        mean = np.zeros((self.d))
        cov = 15*np.identity((self.d))

        self.theta_all = []
        for i in range(self.narm):
            thetas = []
            for j in range(self.B):
                theta = np.random.multivariate_normal(mean=mean,cov=cov)
                thetas.append(theta)

            self.theta_all.append(thetas)

        self.theta_all = np.array(self.theta_all)

        # self.theta_all = np.random.randn(self.narm, self.B, self.d)  # Initialize all arm features 
        # print self.theta_all

    
    def choose(self, context): 
        selected_arm_feats = np.zeros((self.narm, self.d))
        # print selected_arm_feats.shape
        # Sample arm feature for each arm
        for k in range(self.narm):
            selected_feature_index = random.randint(0, self.B-1)
            # print selected_arm_feats.shape
            # print self.theta_all.shape
            # print selected_feature_index
            selected_arm_feats[k,:] = self.theta_all[k, selected_feature_index, : ] # Randomly select the feature

        # Select the arm
        # print context.shape
        # print selected_arm_feats.shape
        exp_rewards = np.matrix(context)*np.transpose(np.matrix(selected_arm_feats))
        
        optarm = np.argmax(exp_rewards)
        self.selected_arm = optarm
        # print optarm
        exp_rewards = np.array(exp_rewards)

        # print optarm
        # print exp_rewards.shape

        return optarm, exp_rewards[0][optarm] 

    def update(self, context, reward, exp_reward):
        # print "here"
        for j in range(self.B):
            p = np.random.poisson(lam=1)
            for z in range(1,p+1):
                eta = 1.0 / math.sqrt(z+1)
                # print eta
                self.theta_all[self.selected_arm, j , :] += eta*(reward - exp_reward)*context / 700 # derivative of log-likelihood

        # print "arm pulled : " + str(self.selected_arm)
        # print self.theta_all
        # time.sleep(1)

    def get_random_arm(self, context):
        arm = random.randint(0, self.narm - 1)
        self.selected_arm = arm 

        r = random.randint(0, self.B - 1)
        exp_reward = np.array(np.matrix(context)*np.transpose(np.matrix(self.theta_all[arm, r, :])))

        # print exp_reward[0][0]
        return arm, exp_reward[0][0]

    def get_params(self):
        return self.theta_all[:,self.B-1,:]

    def name(self):
        return "Online Bootstrap"



class OnlineCollaborativeBootstrap():
    def __init__(self, B=1, narm=10, D=10, M=10):     
        self.B = B
        self.D = D
        self.M = M
        self.narm = narm

        mean = np.zeros((self.D))
        cov = 15*np.identity((self.D))

        self.theta_basis = []
        for i in range(self.M):          
            theta = np.random.multivariate_normal(mean=mean,cov=cov)
            self.theta_basis.append(theta)

        self.theta_basis = np.array(self.theta_basis)
        
        # print self.theta_all

        mean = np.zeros((self.M))
        cov = 15*np.identity((self.M))

        self.Z = []
        for i in range(self.narm):
            z =  np.random.multivariate_normal(mean=mean,cov=cov)
            self.Z.append(z)

        self.Z = np.array(self.Z)

        self.theta_all = np.array(np.matrix(self.Z)*np.matrix(self.theta_basis))

    
    def choose(self, context): 
        selected_arm_feats = np.zeros((self.narm, self.D))
        # print selected_arm_feats.shape
        # Sample arm feature for each arm
        # for k in range(self.narm):
            # selected_feature_index = random.randint(0, self.B-1)
            # print selected_arm_feats.shape
            # print self.theta_all.shape
            # print selected_feature_index
            # selected_arm_feats[k,:] = self.theta_all[k, selected_feature_index, : ] # Randomly select the feature

        # Select the arm
        # print context.shape
        # print selected_arm_feats.shape
        # exp_rewards = np.matrix(context)*np.transpose(np.matrix(selected_arm_feats))
        exp_rewards = np.matrix(context)*np.transpose(np.matrix(self.theta_all))        
        optarm = np.argmax(exp_rewards)
        self.selected_arm = optarm
        # print optarm
        exp_rewards = np.array(exp_rewards)

        # print optarm
        # print exp_rewards.shape

        return optarm, exp_rewards[0][optarm] 

    def update(self, context, reward, exp_reward):
        self.update_Z(context, reward, exp_reward)
        self.update_theta(context, reward, exp_reward)
        self.theta_all = np.array(np.matrix(self.Z)*np.matrix(self.theta_basis))

    def update_Z(self, context, reward, exp_reward):
        eta = 0.0001
        modified_context = np.array(np.matrix(self.theta_basis)*np.transpose(np.matrix(context)))
        # print modified_context.shape
        # print self.Z[self.selected_arm, : ].shape
        # print reward
        # print exp_reward
        self.Z[self.selected_arm, : ] += eta*(reward - exp_reward)*np.squeeze(modified_context)

    def update_theta(self, context, reward, exp_reward):
        for i in range(self.M):
            eta = 0.0001
            modified_context = self.Z[self.selected_arm][i]*context
            exp_pseudo_reward = int(np.array(np.matrix(self.theta_basis[i,:])*np.matrix(modified_context).transpose()))
            # print np.matrix(self.Z[self.selected_arm, :])*np.matrix(self.theta_basis)*np.matrix(context).transpose()
            pseudo_reward = reward + exp_pseudo_reward - int(np.matrix(self.Z[self.selected_arm, :])*np.matrix(self.theta_basis)*np.matrix(context).transpose())
            # print self.theta_basis[i, : ].shape
            # print np.array(modified_context).shape
            self.theta_basis[i, : ] += eta*(pseudo_reward - exp_pseudo_reward)*np.array(modified_context)


    def get_random_arm(self, context):
        arm = random.randint(0, self.narm - 1)
        self.selected_arm = arm 

        r = random.randint(0, self.B - 1)
        exp_reward = np.array(np.matrix(context)*np.transpose(np.matrix(self.theta_all[arm, :])))

        # print exp_reward[0][0]
        return arm, exp_reward[0][0]

    def get_params(self):
        return self.theta_all[:,self.B-1,:]

    def name(self):
        return "Online Collaborative Bootstrap"