import numpy as np
import sys
import time
import gzip
from struct import unpack


def return_pointers(file_context, file_label, data_dir='MNIST_data'):
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

def read_MNIST(file_context, file_label, data_dir='MNIST_data'):
    # trainImageFile = 'train-images-idx3-ubyte.gz'
    # trainLabelFile = 'train-labels-idx1-ubyte.gz'
    # testImageFile = 't10k-images-idx3-ubyte.gz'
    # testLabelFile = 't10k-labels-idx1-ubyte.gz'

    images, labels, NumLabels, NumRows, NumColumns = return_pointers(file_context, file_label)
    images_set = []
    labels_set = []
    for i in range(NumLabels):
        CurrImage = np.zeros((1, NumRows + NumColumns), dtype=np.float32)
        for row in range(NumRows):
            for col in range(NumColumns):
                pixelValue = images.read(1)
                pixelValue = unpack('>B', pixelValue)[0]
                # print (pixelValue)
                CurrImage[0][row+col] = pixelValue * 1.0

        labelValue = labels.read(1)
        labelValue = unpack('>B', labelValue)[0]
        current_label = np.zeros((1, 10), dtype=np.float32)
        current_label[int(labelValue)] = 1

        images_set.append(CurrImage)
        labels_set.append(current_label)


    images_set = np.array(images_set)
    labels_set = np.array(labels_set)

    permutation = np.random.permutation(NumLabels)
    images_set = images_set[permutation]
    labels_set = labels_set[permutation]

    return images_set, labels_set

