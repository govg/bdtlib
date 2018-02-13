import gzip
import numpy as np
from struct import unpack
from numpy import zeros, uint8
from sklearn import preprocessing

def return_pointers(file_context, file_label, data_dir='MNIST_data/'):
    images = gzip.open(data_dir + file_context, 'rb')
    labels = gzip.open(data_dir + file_label, 'rb')

    images.read(4)
    NumImages = images.read(4)
    NumImages = unpack('>I', NumImages)[0]
    # NumImages = 10000
    NumRows = images.read(4)
    NumRows = unpack('>I', NumRows)[0]
    NumColumns = images.read(4)
    NumColumns = unpack('>I', NumColumns)[0]

    labels.read(4)
    NumLabels = labels.read(4)
    NumLabels = unpack('>I', NumLabels)[0]

    return images, labels, NumLabels, NumRows, NumColumns

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  # labels = np.arange(num_labels) == labels[:,None]

  return dataset, labels

def read_MNIST(file_context, file_label, train=True, data_dir='MNIST_data/'):
    try:
        if train == True:
            X = np.load(data_dir + 'trainMNIST_feat.npy')
            y = np.load(data_dir + 'trainMNIST_label.npy')
            y_onehot = np.load(data_dir + 'trainMNIST_label_onehot.npy')
        else:
            X = np.load(data_dir + 'testMNIST_feat.npy')
            y = np.load(data_dir + 'testMNIST_label.npy')
            y_onehot = np.load(data_dir + 'testMNIST_label_onehot.npy')

    except:
        images, labels, NumLabels, NumRows, NumColumns = return_pointers(file_context, file_label)
        X = np.zeros((NumLabels, NumRows, NumColumns), dtype=uint8)
        y = np.zeros((NumLabels, 1), dtype=uint8)
        y_onehot = np.zeros((NumLabels, 10), dtype=uint8)
        for i in range(NumLabels):
            for row in range(NumRows):
                for col in range(NumColumns):
                    pixelValue = images.read(1)
                    pixelValue = unpack('>B', pixelValue)[0]
                    X[i][row][col] = pixelValue
            labelValue = labels.read(1)
            y[i] = unpack('>B', labelValue)[0]
            y_onehot[i][int(y[i])] = 1

        y = np.ravel(y)
        # print "unrolling X"
        X = np.reshape(X, (NumLabels, NumRows * NumColumns))


        permutation = np.random.permutation(NumLabels)
        X = X[permutation]
        y = y[permutation]

        # images_set = preprocessing.scale(images_set)

        if train == True:
            np.save(data_dir + 'trainMNIST_feat', X)
            np.save(data_dir + 'trainMNIST_label', y)
            np.save(data_dir + 'trainMNIST_label_onehot', y_onehot)
        else:
            np.save(data_dir + 'testMNIST_feat', X)
            np.save(data_dir + 'testMNIST_label', y)
            np.save(data_dir + 'testMNIST_label_onehot', y)

    return X, y, y_onehot

