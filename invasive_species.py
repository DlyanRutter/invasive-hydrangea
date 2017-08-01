import os, keras
import numpy as np
import pandas as pd
import tensorflow as tf
#from keris.utils.visualize_util import plot
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing.image import img_to_array, load_img, \
     ImageDataGenerator, NumpyArrayIterator, array_to_img
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, \
     MaxPooling2D, Flatten, Input, merge
from keras.layers.advanced_activations import PReLU
from keras import backend as K

split = 0.9
root = '../input'
np.random.seed(1337)

def train_data():
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

def image_data(ids, max_dim=96, center=True):
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, idee in enumerate(ids):
            x = resize_image(load_img(os.path.join(root,'/Users/dylanrutter/Downloads/images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)

        

    
    
    
