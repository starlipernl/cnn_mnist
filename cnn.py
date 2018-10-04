# ECE 542
# Project 3: CNN
# October 2018

# dataset: MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labesl (digits 0-9)

# Notes:


################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

import os
from datetime import datetime
import matplotlib.pyplot as plt

################################################################################
print("TF version: {}".format(tf.__version__))
print("GPU available: {}".format(tf.test.is_gpu_available()))


################################################################################
# loads MNIST data and
# returns (train_images, train_labels), (test_images, test_labels)
def load_mnist_data():
    # training images
    with open(os.path.join(os.getcwd(), "train-images-idx3-ubyte.gz"), "rb") as f:
        train_images = extract_images(f)

    # training labels
    with open(os.path.join(os.getcwd(), "train-labels-idx1-ubyte.gz"), "rb") as f:
        train_labels = extract_labels(f)

    # testing images
    with open(os.path.join(os.getcwd(), "t10k-images-idx3-ubyte.gz"), "rb") as f:
        test_images = extract_images(f)

    # testing labels
    with open(os.path.join(os.getcwd(), "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        test_labels = extract_labels(f)

    return (train_images, train_labels), (test_images, test_labels)


################################################################################
# loading dataset
(train_images, train_labels), (test_images, test_labels) = load_mnist_data()