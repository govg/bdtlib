import gzip
from numpy import zeros, uint8
import numpy as np
from struct import unpack
from readMNIST import read_MNIST
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

trainImageFile = 'train-images-idx3-ubyte.gz'
trainLabelFile = 'train-labels-idx1-ubyte.gz'
testImageFile = 't10k-images-idx3-ubyte.gz'
testLabelFile = 't10k-labels-idx1-ubyte.gz'

X, y, _ = read_MNIST(trainImageFile, trainLabelFile, train=True)
XTest, yTest, _ = read_MNIST(testImageFile, testLabelFile, train=False)

clf = Perceptron(verbose=0, n_jobs=3)
print "Training"
clf.fit(X, y)
print "Testing"
print "Mean Accuracy", clf.score(XTest, yTest)

layers = (500, 500)
clf = MLPClassifier(hidden_layer_sizes=layers)
print "Training"
clf.fit(X, y)
print "Testing"
print "Mean Accuracy", clf.score(XTest, yTest)