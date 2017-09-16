import os, keras, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

IMG_SIZE = 96
MODEL_NAME = 'invasiveplants -{}-{}.model'.format(LR, '2conv')

def load_train_df():
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

def get_train_df_vals():
    """
    converts df segments into numpy arrays
    """
    files, labels, df = load_train_df()
    name = np.array(df.pop('name'))
    invasive = np.array(df.pop('invasive'))
    one_hot_label = np.array(df.pop('one_hot'))
    return name, invasive, one_hot_label

def label_img(img):
    """
    labels an image with one hot encoding
    """
    label = img.split('.')[-2]
    if label == 0: return [1, 0]
    if label == 1: return [0, 1]

def img_train_data(angle=360.0):
    """
    loads images from training data. resizes images according to dim IMG_SIZE.
    centers img and rotates img according to angle
    """
    img_data = []
    train_dir = '/Users/dylanrutter/Downloads/train'
    
    for img in os.listdir(train_dir):
        
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w,h))
        
        img_data.append(np.array(img))
        
    np.save('img_train_data.npy', img_data)
    return img_data

def create_test_data(angle=360.0):
    test_dir = '/Users/dylanrutter/Downloads/test'
    testing_data = []
    
    for img in os.listdir(test_dir):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

#all_training_imgs = img_train_data()

train_name, train_label, train_one_hot = get_train_df_vals()
train_img = np.load('img_train_data.npy')

def shuffle(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_imgs, train_labels = shuffle(train_img[:1400], train_label[:1400])
valid_imgs, valid_labels = shuffle(train_img[1400:1850], train_label[1400:1850])
test_imgs, test_labels = shuffle(train_img[1850:], train_label[1850:])
