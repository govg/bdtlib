import os
import sys
import numpy as np
from random import randint

def create_data_independent(d = 0, U = 200, N = 100000, K = 200):
	X = np.zeros((N,U+d+1))
	
	mean = np.zeros((U+d+1))
	cov = 100*np.identity((U+d+1))
	theta_true = []

	# print theta.shape
	for i in range(K):
		theta = np.random.multivariate_normal(mean=mean,cov=cov)
		theta_true.append(theta)

	theta_true = np.array(theta_true).transpose()

	Y = []

	for i in range(N):	
		user = randint(0, U - 1)
		X[i][user+d] = 1
		x = np.random.randn(1, d)	
		X[i, : d] = x
		X[i][U+d] = 1   
		true_rating = np.array(np.matrix(X[i,:]) * np.matrix(theta_true)) + np.random.randn(1, K)
		Y.append(np.squeeze(true_rating))


	Y = np.array(Y)

	return X, Y, theta_true

def create_data_dependent(d = 0, U = 200, N = 100000, K = 200, M = 20):
	D = U+d+1
	X = np.zeros((N,D))

	mean = np.zeros((D))
	cov = 20*np.identity((D))

	theta_basis = []
	for i in range(M):          
	    theta = np.random.multivariate_normal(mean=mean,cov=cov)
	    theta_basis.append(theta)

	theta_basis = np.array(theta_basis)

	mean = np.zeros((M))
	cov = 20*np.identity((M))

	Z = []
	for i in range(K):
	    z =  np.random.multivariate_normal(mean=mean,cov=cov)
	    Z.append(z)

	Z = np.array(Z)
	theta_true = np.array(np.matrix(Z)*np.matrix(theta_basis)).transpose()

	Y = []

	for i in range(N):	
		user = randint(0, U - 1)
		X[i][user+d] = 1
		x = np.random.randn(1, d)	
		X[i, : d] = x
		X[i][U+d] = 1   
		true_rating = np.array(np.matrix(X[i,:]) * np.matrix(theta_true)) + np.random.randn(1, K)
		Y.append(np.squeeze(true_rating))


	Y = np.array(Y)

	return X, Y, theta_true