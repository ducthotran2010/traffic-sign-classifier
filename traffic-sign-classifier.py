# Load pickled data
from skimage import transform
from tqdm import tqdm
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_in, y_train = train['features'], train['labels']
X_valid_in, y_valid = valid['features'], valid['labels']
X_test_in, y_test = test['features'], test['labels']

n_train = X_train_in.shape[0]
n_validation = X_valid_in.shape[0]
n_test = X_test_in.shape[0]

image_shape = X_train_in[0].shape

n_classes = len(list(set(y_train)))

print("Number of training examples =", n_train)
print("Number of training validation =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# free memory
del n_train
del n_validation
del n_test
del image_shape

# Data exploration visualization code goes here.
# Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
%matplotlib inline

visual = collections.Counter(y_train)
fig = plt.figure(figsize=(16, 9))
plt.bar(list(visual.keys()), visual.values(), align='center')
plt.xticks(range(len(visual)), visual.keys())
plt.ylabel('Occurrence')
plt.xlabel('Traffic Sign ID')
plt.title("Visualization of training data set")
plt.minorticks_off()

visual = collections.Counter(y_valid)
fig = plt.figure(figsize=(16, 9))
plt.bar(list(visual.keys()), visual.values(), align='center')
plt.xticks(range(len(visual)), visual.keys())
plt.ylabel('Occurrence')
plt.xlabel('Traffic Sign ID')
plt.title('Visualization of valid dataset')
plt.minorticks_off()

visual = collections.Counter(y_valid)
fig = plt.figure(figsize=(16, 9))
plt.bar(list(visual.keys()), visual.values(), align='center')
plt.xticks(range(len(visual)), visual.keys())
plt.ylabel('Occurrence')
plt.xlabel('Traffic Sign ID')
plt.title('Visualization of testing dataset')
plt.minorticks_off()


def rgb2gray(rgb):
    if (rgb.shape[2] == 3):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    else:
        return rgb


def Grayscale(x):
    out = []
    for i in range(x.shape[0]):
        x[i] = np.array(x[i])
        out.append(rgb2gray(x[i]).reshape([32, 32, 1]))
    return np.array(out)


X_train_gray = Grayscale(X_train_in)
X_valid_gray = Grayscale(X_valid_in)
X_test_gray = Grayscale(X_test_in)

# Plot image before gray scale
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
a.set_title('Before')
plt.imshow(X_train_in[0].squeeze())

# Plot image after gray scale
a = fig.add_subplot(1, 2, 2)
a.set_title('After')
plt.imshow(X_train_gray[0].squeeze(), cmap='gray')
plt.savefig('examples/grayscale.png')
a.grid(b=False)

# Normalization

# def Normalization(x):
#    return np.divide(x - 128, 128)


def Normalization(x):
    if (x.max() > 1):
        temp = x-128
        return np.divide(temp, 128)
    else:
        return x


X_train = Normalization(X_train_gray)
X_valid = Normalization(X_valid_gray)
X_test = Normalization(X_test_gray)

# Plot image before normalization
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
a.set_title('Before')
plt.imshow(X_train_gray[0].squeeze(), cmap='gray')

# Plot image after normalization
a = fig.add_subplot(1, 2, 2)
a.set_title('After')
plt.imshow(X_train[0].squeeze(), cmap='gray')
plt.savefig('examples/normalization.png')
a.grid(b=False)


num_data_train = X_train.shape[0]
left_training_file = 'traffic-signs-data/train_left.p'
right_training_file = 'traffic-signs-data/train_right.p'

if os.path.exists(left_training_file) == False:
    print('Rotate 15 degree to left')

    X_train_left = []
    for i in tqdm(range(num_data_train)):
        rotate = transform.rotate(X_train[i].reshape([32, 32]), 15)
        X_train_left = np.append(X_train_left, rotate.reshape([32, 32, 1]))
    X_train_left = X_train_left.reshape([num_data_train, 32, 32, 1])
    new_train_left = {'features': X_train_left, 'labels': y_train}
    with open(left_training_file, mode='wb') as f:
        pickle.dump(new_train_left, f, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.exists(right_training_file) == False:
    print('Rotate 15 degree to right')

    X_train_right = []
    for i in tqdm(range(num_data_train)):
        rotate = transform.rotate(X_train[i].reshape([32, 32]), -15)
        X_train_right = np.append(X_train_right, rotate.reshape([32, 32, 1]))
    X_train_right = X_train_right.reshape([num_data_train, 32, 32, 1])
    new_train_right = {'features': X_train_left, 'labels': y_train}
    with open(left_training_file, mode='wb') as f:
        pickle.dump(new_train_right, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(left_training_file, mode='rb') as f:
    train_left = pickle.load(f)
with open(right_training_file, mode='rb') as f:
    train_right = pickle.load(f)

## Checkpoint for X_train
X_train_save = X_train
y_train_save = y_train
    
## Append    
X_train = np.append(X_train, train_left['features'])
X_train = np.append(X_train, train_right['features'])
y_train = np.append(y_train, train_left['labels'])
y_train = np.append(y_train, train_right['labels'])
X_train = X_train.reshape([len(y_train), 32, 32, 1])

## Show image
fig = plt.figure()
a=fig.add_subplot(1,2,1)
a.set_title('Left')
plt.imshow(train_left['features'][0].squeeze(),  cmap='gray')
a=fig.add_subplot(1,2,2)
a.set_title('Right')
plt.imshow(train_right['features'][0].squeeze(),  cmap='gray')

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 10
BATCH_SIZE = 128

keep_prob = tf.placeholder(tf.float32) 

def max_pool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME'
    )

def conv2d(x, W, b, strides=1):
    return tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(
                x, 
                W, 
                strides=[1, strides, strides, 1], 
                padding='SAME'
            ), b
        )
    )

def LeNet(x, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, X_train[0].shape[2], 32), mean = mu, stddev = sigma)),
        'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma)),
        'wc3': tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean = mu, stddev = sigma)),
        'wf1': tf.Variable(tf.truncated_normal(shape=(14336, 400), mean = mu, stddev = sigma)),
        'wf2': tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
        'wf3': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
        'out': tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros([32])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([128])),
        'bf1': tf.Variable(tf.zeros([400])),
        'bf2': tf.Variable(tf.zeros([120])),
        'bf3': tf.Variable(tf.zeros([84])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }
    
    # Input=32x32x1 --5x5--> Con1=32x32x32 --2x2--> Pool1=16x16x32
    conv1 = max_pool2d(conv2d(x, weights['wc1'], biases['bc1']))
    
    # Input=16x16x32 --5x5--> Con2=16x16x64 --2x2--> Pool2=8x8x64
    conv2 = max_pool2d(conv2d(conv1, weights['wc2'], biases['bc2']))
    
    # Input=8x8x64 --5x5--> Con3=8x8x128 --2x2--> Pool3=4x4x128
    conv3 = max_pool2d(conv2d(conv2, weights['wc3'], biases['bc3']))
    
    # Res = Conv1|Conv2|Conv3 = 16x16x32 + 8x8x64 + 4x4x128 = 14336
    flat = tf.concat([flatten(conv1), flatten(conv2), flatten(conv3)], axis=1)
    
    # Size = 14336 | Dropout = 7168
    flat = tf.nn.dropout(flat, keep_prob)
    
    # Input = 14336 --> 400
    f1 = tf.nn.relu(tf.add(tf.matmul(flat, weights['wf1']), biases['bf1']))
    
    # Input = 400 --> 120
    f2 = tf.nn.relu(tf.add(tf.matmul(f1, weights['wf2']), biases['bf2']))
    
    # Input = 120 --> 84
    f3 = tf.nn.relu(tf.add(tf.matmul(f2, weights['wf3']), biases['bf3']))
    
    # Input = 84 --> 43 (number of classes)
    return tf.add(tf.matmul(f3, weights['out']), biases['out'])

x = tf.placeholder(tf.float32, (None, 32, 32, X_train[0].shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            if end > num_examples:
                end = num_examples
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} Validation Accuracy = {:.3f}".format(i+1, validation_accuracy))

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, './lenet')
    print("Model saved")

from sklearn.utils import shuffle
from tqdm import tqdm

X_train = X_train_save
y_train = y_train_save

saver = tf.train.Saver()
# tf.reset_default_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '.\lenet')
#     saver.restore(sess, tf.train.load_checkpoint('./lenet'))
    
    num_examples = len(X_train)

    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            if end > num_examples:
                end = num_examples
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} Validation Accuracy = {:.3f}".format(i+1, validation_accuracy))
        
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, './lenet-fine-tuning')
    print("Model saved")

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Visualizations will be shown in the notebook.
%matplotlib inline

new_images_name = ['16.png',   # 16 Vehicles over 3.5 metric tons prohibited
                    '18.png',  # 18 General caution
                    '33.png',  # 33 Turn right ahead
                    '25.png',  # 25 Road work
                    '11.png']  # 11 Right-of-way at the next intersection
new_images_id = np.array([16, 18, 33, 25, 11])

new_images = []

fig = plt.figure()
total_new_image = len(new_images_name)
for i in range(total_new_image):
    image = mpimg.imread('traffic-signs-data/' + new_images_name[i])
    a=fig.add_subplot(1, total_new_image, i+1)
    a.set_title(new_images_id[i])
    plt.imshow(image)
    plt.axis('off')
    new_images.append(image)

new_images = np.array(new_images)

new_images = Grayscale(new_images)
new_images = Normalization(new_images)

fig = plt.figure()
for i in range(total_new_image):
    a=fig.add_subplot(1, total_new_image, i+1)
    a.set_title(new_images_id[i])
    plt.imshow(new_images[i].squeeze(), cmap='gray')
    plt.axis('off')

new_images = np.array(new_images)

with tf.Session() as sess:
    # saver.restore(sess, '.\lenet')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    logit = sess.run(tf.argmax(logits, 1), feed_dict={x: new_images, keep_prob: 1})
    print("Actual the Sign Type", new_images_id)
    print("Predict the Sign Type", logit)

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
accuracy = np.sum(logit == new_images_id) * 100 / len(new_images_id)
print("Accuracy = %d%%" % accuracy)