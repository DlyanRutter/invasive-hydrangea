import os, keras, cv2, sys, inspect, keras, random, math, time, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.examples.tutorials.mnist import mnist

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, \
     MaxPooling2D, Flatten, Input, merge
from keras.layers.advanced_activations import PReLU
from keras import backend as K

split = 0.9
root = '../input'
FLAGS = None

np.random.seed(1337)
LR = 1e-3
pixel_depth = 255.0
learning_rate = 0.01

height = 25
width = 25
batch_size = 6
num_labels = 2

train_dir = '/Users/dylanrutter/Downloads/train'
test_dir = '/Users/dylanrutter/Downloads/test'
MODEL_NAME = 'invasiveplants -{}-{}.model'.format(LR, '2conv')

def load_train_df():
    """
    loads all train data into a pandas dataframe then returns filename, df,
    whether or not invasive (value of 0 or 1) , and a one hot label where
    one_hot is [0,1] if not invasive, [1, 0] if invasive
    """
    path = '../input/train/'
    all_train = pd.read_csv(os.path.join\
                         (root,'/Users/dylanrutter/Downloads/train_labels.csv'))
    labels = np.array(all_train['invasive'].iloc[: ])
    
    files = []
    one_hot = []
    
    for i in range(len(all_train)):
        files.append(path + str(int(all_train.iloc[i][0])) + '.jpg')
        if all_train['invasive'][i] == 0:
            one_hot.append([1, 0])
        elif all_train['invasive'][i] == 1:
            one_hot.append([0, 1])
        else:
            one_hot.append(None)
            
    all_train['one_hot'] = one_hot
    all_train['name'] = files
        
    return files, all_train, labels

def get_train_df_vals():
    """
    converts df segments into numpy arrays
    """
    files, train_df, labels = load_train_df()
    name = np.array(train_df.pop('name'))
    invasive = np.array(train_df.pop('invasive'))
    one_hot = np.array(train_df.pop('one_hot'))       
    return name, invasive, one_hot

def augment(img):
    """
    augments image
    """
    random = np.random.randint(6)
    if random == 0:
        img = np.rot90(img, k=1, axes=(1,0))
    if random == 1:
        img = np.flipud(img)
    if random == 2:
        img = np.rot90(src,2)
    if random == 3:
        img = np.fliplr(img)
    if random == 4:
        img = np.rot90(img, k=1, axes=(0,1))
    if random == 5:
        img = np.rot90(img, 2, axes=(1,0))
        img = np.fliplr(img)
    if random == 6:
        img = np.rot90(img, 2)
        img = np.flipud(img)
    return img

class DataSet(object):
    def __init__(self,images,labels, fake_data=False, dtype=dtypes.float32,
                 one_hot=False, seed=None, reshape=True):
        """
        construct batches. dtype can be either dtypes.float32 to rescale
        image into [0,1] or unit8 to leave input as [0, 255]. 
        """
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        
        if fake_data==True:
            self._num_examples = images.shape[0]
            self.one_hot = one_hot

        else:
            
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, \
                                                       labels.shape))
            
            self._num_examples = images.shape[0]

            if dtype == dtypes.float32:
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0/250)
                           
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.fake_data = fake_data

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """
        returns the next "batch_size" examples from DataSet object
        """
        if self.fake_data == True:
            fake_image = [1] * height * width
            if self.one_hot:
                fake_label = [1] + [0]*num_labels
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
                ]
                        
        start = self._index_in_epoch

        #shuffling for first epoch
        if self._epochs_completed == 0 and start == 0:
            perm_start = np.arange(self._num_examples)
            np.random.shuffle(perm_start)
            self._images = self.images[perm_start]
            self._labels = self.labels[perm_start]
        
        #going to next epoch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rem_num_ex = self._num_examples - start
            im_rem_part = self._images[start:self._num_examples]
 #           augmented = []
            
#            for img in im_rem_part:
#                img = augment(img)
#                augmented.append(img)
#            im_rem_part = augmented
            labels_rem_part = self._labels[start:self._num_examples]

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]

            start = 0
            self._index_in_epoch = batch_size - rem_num_ex
            end = self._index_in_epoch
            im_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]

            return np.concatenate((im_rem_part, im_new_part), axis=0),\
                   np.concatenate((labels_rem_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
#            augmented = []
 #           for  img in self._images[start:end]:
            
            #for img in total:
#                img = augment(img)
#                augmented.append(img)
#                
#            self._images[start:end] = augmented
            return self._images[start:end], self._labels[start:end]
                   
def use_keras(height, width):
    """
    use this function to load imgs if using keras rather than tensorflow.
    height = pixel height, width = pixel width, returns an array of shape
    [# images, height, width, color_channel]. Fur use with colored images.
    """
    image_files = os.listdir(train_dir)
    num_imgs = 0
    dataset = np.ndarray(shape = (len(image_files),height,width,3),
                         dtype=np.float32)
    
    for img in image_files:
        path = os.path.join(train_dir, img)
        img = cv2.cvtColor((cv2.resize(cv2.imread(
            path, cv2.IMREAD_COLOR),(height, width))), cv2.COLOR_BGR2RGB)
        dataset[num_images, :, :, :] = img
        num_images = num_imags + 1
        
    dataset = dataset[0:num_images, :, :, :]
    np.save('img_train_color_keras.npy', dataset)
    return dataset

def use_TensorFlow(height, width):
    """
    used to load images if images will be analyzed using TensorFlow. height
    is pixel height, and width is pixel width. returns an array representing
    image of shape [#images, height, width]
    """
    image_files = os.listdir(train_dir)
    num_imgs = 0
    dataset = np.ndarray(shape = (len(image_files),height,width),
                         dtype=np.float32)
    
    for img in image_files:
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(
            path, cv2.IMREAD_GRAYSCALE),(height, width))
        dataset[num_imgs, :, :] = img
        num_imgs = num_imgs + 1
        
    dataset = dataset[0:num_imgs, :, :]
    np.save('img_train_tf', dataset)
    return dataset
       
#tf_img_train = use_TensorFlow(25, 25)
files, train_name, train_label = load_train_df()
tf_train_img = np.load('img_train_tf.npy')

def mix(dataset, labels):
    """
    makes one-hot
    """
    dataset = dataset.reshape((-1, height * width)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_imgs, train_labels = mix(tf_train_img[:1850], train_label[:1850])
valid_imgs, valid_labels = mix(tf_train_img[1850:2000], train_label[1850:2000])
test_imgs, test_labels = mix(tf_train_img[2000:], train_label[2000:])

train_ds = DataSet(train_imgs, train_labels)
valid_ds = DataSet(valid_imgs, valid_labels)
test_ds = DataSet(test_imgs, test_labels)


#print('Training set', train_imgs.shape, train_labels.shape)
#print('Validation set', valid_imgs.shape, valid_labels.shape)
#print('Test set', test_imgs.shape, test_labels.shape)

def losses(logits, labels):
    """
    calculate loss from lotgits and labels. logits is a logits tensor, float -
    [batch_size, num_labels]. labels is labels tensor, int32 - [batch_size].
    returns float loss tensor
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')
        
def placeholder_inputs(batch_size):
    """
    Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Shape of placeholders matches shape of full image and label tensors
    except the first dimension is now the batch size rather than the full
    size of the train or test dataset
      Args:
        batch_size: The batch size will be baked into both placeholders.
      Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=([None,
                                                           height*width]))    
    labels_placeholder = tf.placeholder(tf.int32,)
                                                       
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, im_placeholder, labels_placeholder):
    """
    feed dict = {placeholder: tensor of values to be passed for placeholder.
    data_set = object from input_data.read_data_sets(), im_placeholder = the
    image placeholder from placeholder_inputs(), labels placeholder = the
    labels placeholder from placeholder_inputs(). returns feed_dict mapping
    from placeholders to values
    """
    im_feed,labels_feed = data_set.next_batch(FLAGS.batch_size,
                                              FLAGS.fake_data)

    feed_dict = {
        im_placeholder: im_feed,
        labels_placeholder: labels_feed,
    }
    return feed_dict

def start_eval(sess,
            eval_correct,
            imgs_placeholder,
            labels_placeholder,
            data_set):
    """
    Runs one evaluation against the full epoch of data. sess is the session in
    which the model was strained. eval_correct is the tensor that returns the
    number of correct predictions. imgs_placeholder is the image placeholder.
    labels_placeholder is the labels placeholder. data set is a DataSet object
    """  
    true_count = 0   
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
      feed_dict = fill_feed_dict(data_set,
                                 imgs_placeholder,
                                 labels_placeholder)
      true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def evaluation(logits, labels, k=1):
    """
    Evaluate how well logits predict the label. lots is a Logits tensor,
    float - [batch_size, number_of_classes]. labels is Labels tensor,
    int32 - [batch_size], w/values in range [0:number_of_classes]. Returns a
    scalar int32 tensor with the number of examples correctly predicted in the
    batch. classifier model used is in_top_k Op. It returns a tensor w/shape
    [batch_size] that is True for examples where label is in the top k of
    all logits for that example
    """
    correct = tf.nn.in_top_k(logits, labels, k)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def make_logits(images, initial, hidden1, hidden2):
    """
    build up a model for dataset. Images is an images placeholder. hidden1 is
    the size of the first hidden layer. AKA size of number of labels.
    hidden2 is the size of the second hidden layer. returns an output
    tensor with computed logits
    """
    with tf.name_scope('first'):
        print type((height*width))
        weights = tf.Variable(
            tf.truncated_normal([height*width, FLAGS.initial],
                               stddev=1.0 / math.sqrt(float(height*width))),
            name='weights')
        biases = tf.Variable(tf.zeros([FLAGS.initial]),
                             name='biases')
        f1 = tf.nn.relu(tf.add(tf.matmul(images, weights), biases))
                                                     
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([initial, hidden1],
                                stddev=1.0 / math.sqrt(float(initial))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1]),
                             name='biases')
        h1 = tf.nn.relu(tf.add(tf.matmul(f1, weights), biases))

    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1, hidden2],
                                stddev=1.0 / math.sqrt(float(hidden1))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2]),
                             name='biases')
        h2 = tf.nn.relu(tf.add(tf.matmul(h1, weights), biases))

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2, num_labels],
                                stddev=1.0 / math.sqrt(float(hidden2))),
            name='weights')
        biases = tf.Variable(tf.zeros([num_labels]),
                             name='biases')
        logits = tf.matmul(h2, weights) + biases
    
    return logits

def training_ops(loss, learning_rate=learning_rate):
    """
    makes training ops including a summarizer to track loss over time in
    TensorBoard, makes an optimizer and applies gradients to all training
    variables. loss is a loss tensor, and learning_rate is the learning rate
    used for gradient_descent. returns the Op for training. note that
    tf.summary.scalar adds a scalar summary for the snapshot loss
    """
    tf.summary.scalar('loss', loss) 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
                                   
def run_sess():
    """
    train data for a number of steps, get the sets of images and labels for
    training, validation, and test sets
    """
    with tf.Graph().as_default(): 

        images_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                               height*width])
                                                        
        labels_placeholder = tf.placeholder(tf.int32,)

        logits = make_logits(images_placeholder, FLAGS.initial,
                             FLAGS.hidden1,FLAGS.hidden2)
               
        loss = losses(logits, labels_placeholder)
        train_op = training_ops(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)        
        summary = tf.summary.merge_all()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in xrange(FLAGS.max_steps):
            epoch_loss = 0
            start_time = time.time()
            for _ in range(int(train_ds.num_examples/batch_size)):

               feed_dict = fill_feed_dict(train_ds,
                                            images_placeholder,
                                            labels_placeholder)

               _, loss_value = sess.run(
                   [train_op, loss], feed_dict=feed_dict)
               
               duration = time.time() - start_time
               epoch_loss += loss_value
               
            print('Epoch', step, ' completed out of',
                  FLAGS.max_steps, ' loss', epoch_loss)
            """
            print('Training Data Eval:')
            start_eval(sess,
                       eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       train_ds)
            print('Validation Data Eval:')
            start_eval(sess,
                       eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       valid_ds)
            print('Test Data Eval:')
            start_eval(sess,
                       eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       test_ds)
            """

"""
            if (step % 30 == 0):
                  print("Minibatch loss at step %d: %f" % (step, loss_value))
                  print("Minibatch accuracy: %.1f%%" % accuracy(
                      predictions, batch_labels))
                  print("Validation accuracy: %.1f%%" % accuracy(
                      valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(
        ), test_labels))

           
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' %\
                      (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
"""
 
"""
with tf.Graph().as_default():
  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_data = tf.placeholder(tf.float32, shape=(batch_size, height*width))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_data = tf.constant(test_dataset)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_data, weights) + biasesdef accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  session.run(tf.global_variables_initializer())

  for step in range(num_steps):
                                   
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
                                   
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
"""



def main(_):
  #if tf.gfile.Exists(FLAGS.log_dir):
  #    tf.gfile.DeleteRecursively(FLAGS.log_dir)
 # tf.gfile.MakeDirs(FLAGS.log_dir)
  run_sess()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=300,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--initial',
      type=int,
      default=128,
      help='Number of units in initial layer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=batch_size,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'input_data'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

