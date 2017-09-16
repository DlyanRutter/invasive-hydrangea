import os, keras, cv2
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

root = '../input'
np.random.seed(1337)
LR = 1e-3

def load_train():
    """
    one_hot will return [1,0] if the species is not invasive, [0,1] if it is.
    """
    path = '../input/train/'
    all_labels_set = pd.read_csv(os.path.join\
                         (root,'/Users/dylanrutter/Downloads/train_labels.csv'))
    train_labels = np.array(all_labels_set['invasive'].iloc[: ])
    train_files = []
    one_hot = []

    for i in range(len(all_labels_set)):
        train_files.append(path + str(int(all_labels_set.iloc[i][0])) + '.jpg')
        
        if all_labels_set.iloc[i][1] == 0:
            one_hot.append([1,0])
        else:
            one_hot.append([0,1])
            
    all_labels_set['name'] = train_files
    all_labels_set['one_hot'] = one_hot
    return train_files, train_labels, all_labels_set
    
train_files, train_labels, all_labels_set = load_train()

def get_train_arrays():
    """
    converts df segments into numpy arrays
    """
    files, labels, df = load_train()
    name = np.array(df.pop('name'))
    invasive = np.array(df.pop('invasive'))
    one_hot_label = np.array(df.pop('one_hot'))
    return name, invasive, one_hot_label

def label_img(img):
    label = img.split('.')[-2]
    if label == 0: return [1, 0]
    if label == 1: return [0, 1]
    
train_name, train_invasive, train_one_hot = get_train_arrays()

MODEL_NAME = 'invasiveplants -{}-{}.model'.format(LR, '2conv')

def load_imgs(IMG_SIZE=96):
    train_data = []
    train_dir = '/Users/dylanrutter/Downloads/train'
    for img in os.listdir(train_dir):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_data.append([np.array(img), np.array(label)])
 #   shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data


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

def center_image(img):
    """
    Centers image (img) and resizes it
    """
    size = [96, 96]
    img_size = img.shape[:2]

    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized
    

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
        
