import os, keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, NumpyArrayIterator, array_to_img
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge
from keras.layers.advanced_activations import PReLU
from keras import backend as K

split = 0.9
root = '../input'
np.random.seed(1337)

def train_data():
    """
    Loads training data from data file into a Pandas dataframe. Returns labels (whether or not a hydrangea is presetn)
    as well as ID numbers.
    """
    data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/train_labels.csv'))
    sample = data.pop('name')
    invasive = data.pop('invasive')
    return sample, invasive

def resize_image(img, max_dim=96):
    """
    Rescale the image so that the longest axis has dimensions max_dim
    """
    bigger, smaller = float(max(img.size)), float(min(img.size))
    scale = max_dim / bigger
    return img.resize((int(bigger*scale), int(smaller*scale)))

def image_train(ids, max_dim=96, center=True):
    """
    Loads images from train dataset. Converts each image into an array and places the arrays into a matrix. The dimensions
    of the images are designated by "max_dim" argument, and whether or not the image is centered is designated by "center"
    argument.
    """
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, idee in enumerate(ids):
            x = resize_image(load_img(os.path.join(root,'/Users/dylanrutter/Downloads/train', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
            x = img_to_array(x)
            length = x.shape[0]
            width = x.shape[1]

            if center:
                h1 = int((max_dim - length) / 2)
                h2 = h1 + length
                w1 = int((max_dim - width) / 2)
                w2 = w1 + width
            else:
                h1, w1 = 0, 0
                h2, w2 = (length, width)
            X[i, h1:h2, w1:w2, 0:1] = x
    return np.around(X / 255.0)

def image_test(ids, max_dim=96, center=True):
    """
    Loads images from test dataset. Converts each image into an array and places the arrays into a matrix. The dimensions
    of the images are designated by "max_dim" argument, and whether or not the image is centered is designated by "center"
    argument.
    """
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, idee in enumerate(ids):
            x = resize_image(load_img(os.path.join(root,'/Users/dylanrutter/Downloads/test', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
            x = img_to_array(x)
            length = x.shape[0]
            width = x.shape[1]

            if center:
                h1 = int((max_dim - length) / 2)
                h2 = h1 + length
                w1 = int((max_dim - width) / 2)
                w2 = w1 + width
            else:
                h1, w1 = 0, 0
                h2, w2 = (length, width)
            X[i, h1:h2, w1:w2, 0:1] = x
    return np.around(X / 255.0)
        
