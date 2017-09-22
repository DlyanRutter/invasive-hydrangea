
import os, keras, cv2, sys, inspect, keras, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, \
     MaxPooling2D, Flatten, Input, merge
from keras.layers.advanced_activations import PReLU
from keras import backend as K

split = 0.9
root = '../input'

np.random.seed(1337)
LR = 1e-3
pixel_depth = 255.0

train_dir = '/Users/dylanrutter/Downloads/train'
test_dir = '/Users/dylanrutter/Downloads/test'
MODEL_NAME = 'invasiveplants -{}-{}.model'.format(LR, '2conv')

def load_train_df():
    """
    loads all train data into a pandas dataframe then 
    """
    path = '../input/train/'
    all_train = pd.read_csv(os.path.join\
                         (root,'/Users/dylanrutter/Downloads/train_labels.csv'))
    labels = np.array(all_train['invasive'].iloc[: ])
    files = []
    for i in range(len(all_train)):
        files.append(path + str(int(all_train.iloc[i][0])) + '.jpg')
    all_train['name'] = files
    return files, all_train, labels

def get_train_df_vals():
    """
    converts df segments into numpy arrays
    """
    files, train_df, labels = load_train_df()
    name = np.array(train_df.pop('name'))
    invasive = np.array(train_df.pop('invasive'))
    return name, invasive

def label_img(img):
    """
    labels an image with one hot encoding
    """
    label = img.split('.')[-2]
    if label == 0: return [1, 0]
    if label == 1: return [0, 1]

def augment(img):
    """
    augments image
    """
    random = random.randint(0,6)
    if random == 0:
        img = np.rot90(img, 1)
    if random == 1:
        img = np.rot90(img, 2)
    if random == 2:
        img = np.flipr(img)
    if random == 3:
        img = np.flipud(img)
    if choice == 4:
        img = np.rot90(img, 3)
    if choice == 5:
        img = np.rot90(img, 2)
        img = np.flipr(img)
    if choice == 6:
        img = np.rot90(img, 2)
        img = np.flipud(img)
    return img
    
def use_keras(height, width, color=True):
    """
    use this function to load imgs if using keras rather than tensorflow.
    height = pixel height, width = pixel width, color=True for colored
    images, color=False for grayscale. returns an array of shape
    [# images, height, width, color_channel] if color = True. shape is
    [# images, height, width] if color = False
    """
    train_dir = '/Users/dylanrutter/Downloads/train'
    image_files = os.listdir(train_dir)
    num_imgs = 0
    
    if color == True:
        dataset = np.ndarray(shape = (len(image_files),height,width,3),
                             dtype=np.float32) 
        for img in image_files:
            path = os.path.join(train_dir, img)
            img = cv2.cvtColor((cv2.resize(cv2.imread(
                path, cv2.IMREAD_COLOR),(height, width))), cv2.COLOR_BGR2RGB)
            img = augment(img)
            dataset[num_images, :, :, :] = img
            num_images = num_imags + 1
        dataset = dataset[0:num_images, :, :, :]
        np.save('img_train_color_keras.npy', dataset)
        return dataset

    else:
        dataset = np.ndarray(shape = (len(image_files),height,width),
                             dtype=np.float32)
        for img in image_files:
            path = os.path.join(train_dir, img)
            img = cv2.resize(cv2.imread(
                path, cv2.IMREAD_GRAYSCALE),(height, width))
            img = augment(img)
            dataset[num_images, :, :] = img
            num_images = num_imags + 1
        dataset = dataset[0:num_images, :, :]
        np.save('img_train_gray_keras.npy', dataset)
        return dataset

def use_TensorFlow(height, width):
    """
    used to load images if images will be analyzed using TensorFlow. height
    is pixel height, and width is pixel width. returns an array representing
    image of shape [#images, height, width]
    """
    train_dir = '/Users/dylanrutter/Downloads/train'
    image_files = os.listdir(train_dir)
    num_imgs = 0
    dataset = np.ndarray(shape = (len(image_files),height,width),
                         dtype=np.float32)
    for img in image_files:
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(
            path, cv2.IMREAD_GRAYSCALE),(height, width))
        img = augment(img)
        dataset[num_images, :, :, :] = img
        num_images = num_images + 1
    dataset = dataset[0:num_images, :, :]
    np.save('img_train_gray_keras.npy', dataset)
    return dataset
