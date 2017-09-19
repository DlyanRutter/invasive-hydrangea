
import os, keras, cv2, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
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
pixel_depth = 255.0

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

def img_train_data(IMG_SIZE=800, keras=False):
    """
    loads images from training data. resizes images according to dim IMG_SIZE.
    centers img and rotates img according to angle. If keras is set to true,
    shape of output will be (#images, pixel height, pixel width, color channel).
    otherwise, will be set to TensorFlow standards (#images, height, width)
    """
    img_data = []
    train_dir = '/Users/dylanrutter/Downloads/train'
    image_files = os.listdir(train_dir)
    dataset = np.ndarray(shape = (len(image_files),IMG_SIZE,IMG_SIZE),
                         dtype=np.float32)                                    
    num_images=0
    for img in image_files:
        path = os.path.join(train_dir, img)
        
        if keras == True:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),
                             (IMG_SIZE,IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_data.append(np.array(img))

        else:
            image_data = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                                    (IMG_SIZE,IMG_SIZE))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1

    dataset = dataset[0:num_images, :, :]

    if keras == True:
        np.save('img_train_data_keras.npy', img_data)
        return img_data
    else:
        np.save('img_train_data_tf.npy', dataset)
        return dataset

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
tf_img_train = img_train_data(keras=True)
#train_img = np.load('img_train_data_keras.npy')
train_name, train_label, train_one_hot = get_train_df_vals()
train_img = np.load('img_train_data_keras.npy')


def accuracy(predictions, labels):
     return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
             / predictions.shape[0])

def classify_using_TensorFlow():
     #since we're using tensor flow which requires grayscale
     train_img = train_img[:,:,:,0]
     num_labels=2
     
     def mix(dataset, labels, image_size=800,):
       """
       makes one-hot
       """
       dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
       labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
       return dataset, labels

     train_dataset, train_labels = mix(train_img[:1850], train_label[:1850])
     valid_dataset, valid_labels = mix(train_img[1850:2000], train_label[1850:2000])
     test_dataset, test_labels = mix(train_img[2000:], train_label[2000:])

     print('Training set', train_dataset.shape, train_labels.shape)
     print('Validation set', valid_dataset.shape, valid_labels.shape)
     print('Test set', test_dataset.shape, test_labels.shape)

     batch_size = 5
     img_size = 800


     graph = tf.Graph()
     with graph.as_default():

      # Input data.
      # Load the training, validation and test data into constants that are
      # attached to the graph.
          tf_train_data = tf.placeholder(tf.float32,shape=(batch_size,img_size*img_size))
          tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
          tf_valid_dataset = tf.constant(valid_dataset)
          tf_test_data = tf.constant(test_dataset)
  
       # Variables.
       # These are the parameters that we are going to be training. The weight
       # matrix will be initialized using random values following a (truncated)
       # normal distribution. The biases get initialized to zero.
          weights = tf.Variable(
          tf.truncated_normal([img_size * img_size, num_labels]))
          biases = tf.Variable(tf.zeros([num_labels]))
  
      # Training computation.
      # We multiply the inputs with the weight matrix, and add biases. We compute
      # the softmax and cross-entropy (it's one operation in TensorFlow, because
      # it's very common, and it can be optimized). We take the average of this
      # cross-entropy across all training examples: that's our loss.
          logits = tf.matmul(tf_train_data, weights) + biases
          loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                            logits=logits))
  
     # Optimizer.
     # We are going to find the minimum of this loss using gradient descent.
          optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
       # Predictions for the training, validation, and test data.
       # These are not part of training, but merely here so that we can report
       # accuracy figures as we train.
          train_prediction = tf.nn.softmax(logits)
          valid_prediction = tf.nn.softmax(
          tf.matmul(tf_valid_dataset, weights) + biases)
          test_prediction = tf.nn.softmax(tf.matmul(tf_test_data, weights) + biases)

     with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print("Initialized")
          for step in range(num_steps):
                                   
          # Pick an offset within the training data, which has been randomized.
          # Note: we could use better randomization across epochs.
                                   
               offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
               # Generate a minibatch.
               batch_data = train_dataset[offset:(offset + batch_size), :]
               batch_labels = train_labels[offset:(offset + batch_size), :]
                                   
         # Prepare a dictionary telling the session where to feed the minibatch.
         # The key of the dictionary is the placeholder node of the graph to be fed,
         # and the value is the numpy array to feed to it.
               feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
               _, l, predictions = session.run(
               [optimizer, loss, train_prediction], feed_dict=feed_dict)
                                   
               if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
          print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
