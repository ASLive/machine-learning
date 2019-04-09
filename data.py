import os
import numpy as np
import math
import pickle

def load_data():
    images, labels, class_names = read_data()
    (train_images, train_labels), (test_images, test_labels) = split_data(images, labels)
    return [train_images, train_labels, test_images, test_labels, class_names]

def read_data():
    """read data from files (run setup_asl.py to generate)"""
    images = pickle.load( open("pickle/images.pickle","rb") )
    labels = pickle.load( open("pickle/labels.pickle","rb") )
    classes = pickle.load( open("pickle/classes.pickle","rb") )
    return np.array(images), np.array(labels), classes

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_data(images, labels):
    """split training and testing data"""
    train_percent = 0.7
    count = math.floor(len(images)*train_percent)
    images, labels = unison_shuffled_copies(images, labels)
    train_images, test_images = images[:count,:], images[count:,:]
    train_labels, test_labels = labels[:count], labels[count:]
    return (train_images, train_labels), (test_images, test_labels)
