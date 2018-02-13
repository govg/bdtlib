'''
Describes agents for MAB with and without context

Each agent provides the following two methods :
    choose ()   :   choose an arm from the given contexts. Returns
                    the index of the arm to be played
    update()    :   Updates the model parameters given the context
                    and the reward
'''
import sys
import time
import math
import random
import numpy as np
from numpy.random import multivariate_normal
from Derivative import derivative_real, derivative_binary

'''
Will contain the following policies :   
    Thompson Sampling   :   Updates using Bayes rule, generates weight
                            vector using distribution
    LinUCB              :   Similar to normal LinUCB rule
    Random              :   Randomly select an arm
    Eps-Greedy          :   Epsilon greedy selection of arm
    Bootstrap           :   Bootstrap method for arm selection

'''

def sigmoid(x):
    sig = 1.0 / (1+np.exp(-x))
    return sig

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
    def __init__(self, alpha=1, d=10, sigma=5, narm=10):
        self.d = d
        self.sigma_sq = sigma**2
        self.narm = narm
        self.A = []
        for i in range(narm):
            temp = np.matrix(np.identity(d))          
            self.A.append(temp)

        self.A = np.array(self.A)                  # K Identity Matrices
        self.b = np.matrix(np.random.randn(narm, d))     # K 'b' vectors

        self.alpha = alpha
             
    def choose(self, context):
        context = np.matrix(context).transpose()

        ucb = np.zeros(self.narm)
        for i in range(self.narm):
            A = np.matrix(self.A[i,:,:])
            b = np.matrix(self.b[i,:]).transpose()
            Ainv = np.linalg.inv(A)

            mu = Ainv * b
            sig = Ainv * self.sigma_sq

            ucb[i] = float((mu.transpose() * context) + self.alpha*(np.sqrt(mu.transpose()*sig*mu))) 

        # print self.narm - np.count_nonzero(ucb)
        optarm = np.argmax(ucb)
        self.selected_arm = optarm
        exp_rewards = np.array(ucb)

        return optarm, exp_rewards[optarm] 

    def update(self, context, reward, exp_reward, reward_type, factor):
        x = np.matrix(context)
        self.A[self.selected_arm,:,:] += x.transpose()*x
        # print self.b[self.selected_arm,:].shape
        # print reward
        # print x.shape
        self.b[self.selected_arm,:] += reward*x

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

    def choose(self, context):
        # N = contexts.shape[0]
        # if self.flag:
        optarm = random.randint(0, self.narm - 1)
        # else:
            # optarm = self.arm
        return optarm, 0

    def update(self, context, reward, exp_reward, reward_type, factor):
        return

    def name(self):
        return "Random"


class OnlineBootstrap():
    def __init__(self, B=1, narm=10, d=10, reward_type='real'):
        self.B = B
        self.d = d
        self.narm = narm
        self.l = 1
        self.reward_type = reward_type

        mean = np.zeros((self.d))
        cov = np.identity((self.d))

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
        if self.reward_type == 'binary':
            exp_rewards = sigmoid(exp_rewards)
        
        optarm = np.argmax(exp_rewards)
        self.selected_arm = optarm
        # print optarm
        exp_rewards = np.array(exp_rewards)

        # print optarm
        # print exp_rewards.shape

        return optarm, exp_rewards[0][optarm] 

    def update(self, context, reward, exp_reward, reward_type, factor):
        # print "here"
        for j in range(self.B):
            p = np.random.poisson(lam=1)
            for z in range(1,p+1):
                eta = 1.0 / math.sqrt(z+1)
                self.l += 1
                # print eta
                # self.theta_all[self.selected_arm, j , :] += eta*(reward - exp_reward)*context / 400 # derivative of log-likelihood
                if reward_type == "real":
                    self.theta_all[self.selected_arm, j , :] += eta*derivative_real(reward, exp_reward, context, factor)
                else:
                    self.theta_all[self.selected_arm, j , :] += eta*derivative_binary(reward, exp_reward, context, factor)

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
    def __init__(self, B=1, narm=10, D=10, M=10, reward_type='real'):
        self.B = B
        self.D = D
        self.M = M
        self.narm = narm
        self.reward_type = reward_type

        mean = np.zeros((self.D))
        cov = np.identity((self.D))
      
        self.theta_basis = []
        for i in range(self.M):          
            theta = np.random.multivariate_normal(mean=mean,cov=cov)
            self.theta_basis.append(theta)

        self.theta_basis = np.array(self.theta_basis)
        
        # print self.theta_all

        mean = np.zeros((self.M))
        cov = np.identity((self.M))
     
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
        exp_rewards = np.matrix(context) * np.transpose(np.matrix(self.theta_all))
        if self.reward_type == 'binary':
            exp_rewards = sigmoid(exp_rewards)


        optarm = np.argmax(exp_rewards)
        self.selected_arm = optarm
        # print optarm
        exp_rewards = np.array(exp_rewards)

        # print optarm
        # print exp_rewards.shape

        return optarm, exp_rewards[0][optarm] 

    def update(self, context, reward, exp_reward, reward_type, factor):
        self.update_Z(context, reward, exp_reward, reward_type, factor)
        self.update_theta(context, reward, exp_reward, reward_type, factor)
        self.theta_all = np.array(np.matrix(self.Z)*np.matrix(self.theta_basis))

    def update_Z(self, context, reward, exp_reward, reward_type, factor):
        eta = 0.006
        modified_context = np.squeeze(np.array(np.matrix(self.theta_basis)*np.transpose(np.matrix(context))))
        # print np.transpose(np.matrix(context)).shape
        # print modified_context.shape
        # print self.Z[self.selected_arm, : ].shape
        # print reward
        # print exp_reward
        # self.Z[self.selected_arm, : ] += eta*(reward - exp_reward)*np.squeeze(modified_context)
        if reward_type == "real":
            self.Z[self.selected_arm, : ] += eta*derivative_real(reward, exp_reward, modified_context, factor)
        else:
            self.Z[self.selected_arm, : ] += eta*derivative_binary(reward, exp_reward, modified_context, factor)


    def update_theta(self, context, reward, exp_reward, reward_type, factor):
        for i in range(self.M):
            eta = 0.006
            modified_context = np.array(self.Z[self.selected_arm][i]*context)
            exp_pseudo_reward = int(np.array(np.matrix(self.theta_basis[i,:])*np.matrix(modified_context).transpose()))
            # print np.matrix(self.Z[self.selected_arm, :])*np.matrix(self.theta_basis)*np.matrix(context).transpose()
            pseudo_reward = reward + exp_pseudo_reward - int(np.matrix(self.Z[self.selected_arm, :])*np.matrix(self.theta_basis)*np.matrix(context).transpose())
            # print self.theta_basis[i, : ].shape
            # print np.array(modified_context).shape
            # self.theta_basis[i, : ] += eta*(pseudo_reward - exp_pseudo_reward)*np.array(modified_context)
            if reward_type == "real":
                # print self.Z[self.selected_arm, : ].shape
                # print derivative_real(pseudo_reward, exp_pseudo_reward, modified_context, factor).shape   
                self.theta_basis[i, : ] += eta*derivative_real(pseudo_reward, exp_pseudo_reward, modified_context, factor)
            else:
                self.theta_basis[i, : ] += eta*derivative_binary(pseudo_reward, exp_pseudo_reward, modified_context, factor)
            


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


# class OnlineCollaborativeBootstrap():
#     def __init__(self, B=1, narm=10, D=10, M=10):
#         self.B = B
#         self.D = D
#         self.M = M
#         self.narm = narm
#         self.l1 = 1
#         self.l2 = 1
#
#         mean = np.zeros((self.D))
#         cov = 1*np.identity((self.D))
#
#         self.theta_basis_all = []
#         for i in range(self.M):
#             theta_basis = []
#             for j in range(self.B):
#                 theta = np.random.multivariate_normal(mean=mean,cov=cov)
#                 theta_basis.append(theta)
#
#             self.theta_basis_all.append(theta_basis)
#
#         self.theta_basis_all = np.array(self.theta_basis_all)
#
#         # print self.theta_all
#
#         mean = np.zeros((self.M))
#         cov = 1*np.identity((self.M))
#
#         self.Z = []
#         for i in range(self.narm):
#             Z_temp = []
#             for j in range(self.B):
#                 z =  np.random.multivariate_normal(mean=mean,cov=cov)
#                 Z_temp.append(z)
#
#             self.Z.append(Z_temp)
#
#         self.Z = np.array(self.Z)
#
#         # self.theta_all = np.array(np.matrix(self.Z)*np.matrix(self.theta_basis))
#
#
#     def choose(self, context):
#         self.selected_arm_Z_feats = np.zeros((self.narm, self.M))
#         self.selected_arm_theta_basis_feats = np.zeros((self.M, self.D))
#
#         for m in range(self.M):
#             selected_theta_index = random.randint(0, self.B-1)
#             self.selected_arm_theta_basis_feats[m,:] = self.theta_basis_all[m,selected_theta_index,:]
#
#         for k in range(self.narm):
#             selected_Z_index = random.randint(0, self.B-1)
#             self.selected_arm_Z_feats[k,:] = self.Z[k,selected_Z_index,:]
#
#         self.theta_all = np.matrix(self.selected_arm_Z_feats)*np.matrix(self.selected_arm_theta_basis_feats)
#
#         exp_rewards = np.matrix(context)*np.transpose(self.theta_all)
#         optarm = np.argmax(exp_rewards)
#         self.selected_arm = optarm
#         # print optarm
#         exp_rewards = np.array(exp_rewards)
#
#         # print optarm
#         # print exp_rewards.shape
#
#         return optarm, exp_rewards[0][optarm]
#
#     def update(self, context, reward, exp_reward, reward_type, factor):
#         self.update_Z(context, reward, exp_reward, reward_type, factor)
#         self.update_theta(context, reward, exp_reward, reward_type, factor)
#         # self.theta_all = np.array(np.matrix(self.Z)*np.matrix(self.theta_basis))
#
#     def update_Z(self, context, reward, exp_reward, reward_type, factor):
#         # eta = 0.0005
#         modified_context = np.squeeze(np.array(np.matrix(self.selected_arm_theta_basis_feats)*np.transpose(np.matrix(context))))
#         for j in range(self.B):
#             p = np.random.poisson(lam=1)
#             for z in range(1,p+1):
#                 eta = 1.0 / math.sqrt(z+1)
#                 self.l1 += 1
#                 eta = 0.00008
#                 # print eta
#                 # self.theta_all[self.selected_arm, j , :] += eta*(reward - exp_reward)*context / 400 # derivative of log-likelihood
#                 if reward_type == "real":
#                     self.Z[self.selected_arm, j , :] += eta*derivative_real(reward, exp_reward, modified_context, factor)
#                 else:
#                     self.Z[self.selected_arm, j , :] += eta*derivative_binary(reward, exp_reward, modified_context, factor)
#         # print np.transpose(np.matrix(context)).shape
#         # print modified_context.shape
#         # print self.Z[self.selected_arm, : ].shape
#         # print reward
#         # print exp_reward
#         # self.Z[self.selected_arm, : ] += eta*(reward - exp_reward)*np.squeeze(modified_context)
#         # if reward_type == "real":
#         #     self.Z[self.selected_arm, : ] += eta*derivative_real(reward, exp_reward, modified_context, factor)
#         # else:
#         #     self.Z[self.selected_arm, : ] += eta*derivative_binary(reward, exp_reward, modified_context, factor)
#
#
#     def update_theta(self, context, reward, exp_reward, reward_type, factor):
#         # print "here"
#         for m in range(self.M):
#             # eta = 0.0005
#
#
#             # print np.matrix(self.Z[self.selected_arm, :])*np.matrix(self.theta_basis)*np.matrix(context).transpose()
#
#             for j in range(self.B):
#                 modified_context = np.squeeze(np.array(float(self.Z[self.selected_arm][j][m])*context))
#                 if np.isnan(self.selected_arm_theta_basis_feats[m,:]).any():
#                     sys.exit(0)
#                 # print self.selected_arm_theta_basis_feats[m,:]
#                 exp_pseudo_reward = float(np.matrix(self.selected_arm_theta_basis_feats[m,:])*np.matrix(modified_context).transpose())
#                 pseudo_reward = reward + exp_pseudo_reward - float(np.matrix(self.selected_arm_Z_feats[self.selected_arm, :])*np.matrix(self.selected_arm_theta_basis_feats)*np.matrix(context).transpose())
#                 # print "theta_basis", np.matrix(self.theta_basis_all[m,j,:])
#                 # print "Z", self.Z[self.selected_arm][j][m]
#                 # exp_pseudo_reward = int(np.matrix(self.selected_arm_theta_basis_feats[m,:])*np.matrix(modified_context).transpose())
#                 # pseudo_reward = reward + exp_pseudo_reward - int(np.matrix(self.selected_arm_Z_feats[self.selected_arm, :])*np.matrix(self.selected_arm_theta_basis_feats)*np.matrix(context).transpose())
#
#                 p = np.random.poisson(lam=1)
#                 for z in range(1,p+1):
#                     eta = 1.0 / math.sqrt(z+1)
#                     self.l2 += 1
#                     eta = 0.00008
#                     # print eta
#                     # self.theta_all[self.selected_arm, j , :] += eta*(reward - exp_reward)*context / 400 # derivative of log-likelihood
#                     if reward_type == "real":
#                         self.theta_basis_all[m, j , :] += eta*derivative_real(pseudo_reward, exp_pseudo_reward, modified_context, factor)
#                     else:
#                         self.theta_basis_all[m, j , :] += eta*derivative_binary(reward, exp_reward, modified_context, factor)
#             # print self.theta_basis[i, : ].shape
#             # print np.array(modified_context).shape
#             # self.theta_basis[i, : ] += eta*(pseudo_reward - exp_pseudo_reward)*np.array(modified_context)
#             # if reward_type == "real":
#             #     # print self.Z[self.selected_arm, : ].shape
#             #     # print derivative_real(pseudo_reward, exp_pseudo_reward, modified_context, factor).shape
#             #     self.theta_basis[i, : ] += eta*derivative_real(pseudo_reward, exp_pseudo_reward, modified_context, factor)
#             # else:
#             #     self.theta_basis[i, : ] += eta*derivative_binary(pseudo_reward, exp_pseudo_reward, modified_context, factor)
#
#         # sys.exit(0)
#
#     def get_random_arm(self, context):
#         arm = random.randint(0, self.narm - 1)
#         self.selected_arm = arm
#
#         # r = random.randint(0, self.B - 1)
#
#         self.selected_arm_Z_feats = np.zeros((self.narm, self.M))
#         self.selected_arm_theta_basis_feats = np.zeros((self.M, self.D))
#
#         for m in range(self.M):
#             selected_theta_index = random.randint(0, self.B-1)
#             self.selected_arm_theta_basis_feats[m,:] = self.theta_basis_all[m,selected_theta_index,:]
#
#         # for k in range(self.narm):
#         selected_Z_index = random.randint(0, self.B-1)
#         self.selected_arm_Z_feats[0,:] = self.Z[arm,selected_Z_index,:]
#
#         self.theta_all = np.matrix(self.selected_arm_Z_feats)*np.matrix(self.selected_arm_theta_basis_feats)
#         exp_reward = np.array(np.matrix(context)*np.transpose(np.matrix(self.theta_all[arm, :])))
#         # print exp_reward[0][0]
#         return arm, exp_reward[0][0]
#
#     def get_params(self):
#         return self.theta_all[:,self.B-1,:]
#
#     def name(self):
#         return "Online Collaborative Bootstrap"
