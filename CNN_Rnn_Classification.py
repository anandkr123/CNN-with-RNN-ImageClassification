# Image classification
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize

n_outputs = 15  # we are using only fifteen classes

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

learning_rate = 0.003
lstm_images = 315
num_input = 256
num_hidden_1 = 196   # 1st layer num features
num_hidden_2 = 144 # 2nd layer num features (the latent dim)
logits_layer = n_outputs

# declare weights and biases
weights = {
    'layer_w1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'layer_w2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'logits_w3': tf.Variable(tf.random_normal([num_hidden_2, logits_layer]))

}
biases = {
    'layer_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'layer_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'logits_b3': tf.Variable(tf.random_normal([logits_layer]))
}

X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
y = tf.placeholder(tf.float32, [None, n_outputs])


def reset_graph(seed = 2018):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def conv2d(input_tensor, depth, kernel, strides, padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding,
                            activation=tf.nn.relu)

# convolutional and fully connected layers for training of the CNN model
def total_conv_net(x):
    net = conv2d(x, 32, 2, strides=(2, 2)) # (128-2)/2  + 1 = 64
    net = conv2d(net, 48, 3, strides=(2, 2)) # (64-3)/2 + 1 = 31.5
    net = conv2d(net, 1, 3, strides=(2, 2)) # (32-3)/2 +1 = 15.5 , (16, 16)
    return net

def fully_connected(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['layer_w1']),
                                   biases['layer_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['layer_w2']),
                                   biases['layer_b2']))
    return layer_2

# preparing data for RNN input in single pass
# piled convolutional and FC layers with trained weights for passing data to RNN input for further training
def data_lstm(x, batch):
    net = conv2d(x, 32, 2, strides=(2, 2))  # (128-2)/2  + 1 = 64
    net = conv2d(net, 48, 3, strides=(2, 2))  # (64-3)/2 + 1 = 31.5
    net = conv2d(net, 1, 3, strides=(2, 2))  # (32-3)/2 +1 = 15.5 , (16, 16)
    trim_channel = tf.unstack(net, axis=3)
    flat_array = tf.contrib.layers.flatten(trim_channel)
    new_flat = tf.reshape(flat_array, [batch, 256])
    layer_2 = fully_connected(new_flat)
    lstm_layer = tf.reshape(layer_2, [batch, 12, 12])      # lstm with 12 steps and 12 features per step
    return lstm_layer

# passing the images
convolution_net = total_conv_net(X)

trim_channel = tf.unstack(convolution_net, axis=3)
flat_array = tf.contrib.layers.flatten(trim_channel)
new_flat = tf.reshape(flat_array, [105, 256])
layer_2 = fully_connected(new_flat)
print(layer_2.shape)
logits = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['logits_w3']),
                                   biases['logits_b3']))


# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


cnt = 1


counting1 = 1
counting2 = 1
count_same_image = 15
count_diff_image = 0
one_type_img = 21

l = 0  # for sequential labels indexing

images = np.zeros((15*21, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
val_images = np.zeros((15*21, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

images_test = np.zeros((9, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)   # test images for RNN model

image_labels = np.ndarray(shape=(21 * 15,), dtype=int)
labels_test = np.ndarray(shape=(9,), dtype=int)                                     # test labels for RNN model

init = tf.global_variables_initializer()

# pre-processing the training and validation images 
with tf.Session() as sess:
    sess.run(init)
    for i in range(count_same_image):
        for j in range(one_type_img):
            read_path = os.getcwd() + '/Train2/' + 'obj' + str(i+1) + '__' + str(count_diff_image) + '.png'
            img = imread(read_path)
            img_as_array = np.array(img)
            img2gray = sess.run(tf.image.rgb_to_grayscale(img_as_array))
            img_for_tensor = resize(img2gray, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant',
                                    preserve_range=True)
            images[cnt - 1] = img_for_tensor
            cnt += 1
            count_diff_image += 5
            image_labels[l] = i
            l += 1
        count_diff_image = 0


cnt = 1
count_same_image = 15
count_diff_image = 105
one_type_img = 21


# init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(count_same_image):
        for j in range(one_type_img):
            read_path = os.getcwd() + '/Val/' + 'obj' + str(i+1) + '__' + str(count_diff_image) + '.png'
            img = imread(read_path)
            img_as_array = np.array(img)
            img2gray = sess.run(tf.image.rgb_to_grayscale(img_as_array))
            img_for_tensor = resize(img2gray, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant',
                                    preserve_range=True)
            val_images[cnt - 1] = img_for_tensor
            cnt += 1
            count_diff_image += 5
        count_diff_image = 105

cnt = 1
path_test = '/home/smaph/PycharmProjects/autoencoder/Test/'

with tf.Session() as sess:
    sess.run(init)
    file_images = os.listdir(path_test)
    count = 0
    for i in file_images:
        x = str(i).split('_')[0][3:]
        labels_test[count] = int(x)
        count += 1
        img = imread(path_test + i)
        img_as_array = np.array(img)
        img2gray = sess.run(tf.image.rgb_to_grayscale(img_as_array))
        img_for_tensor = resize(img2gray, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant',
                                preserve_range=True)
        images_test[cnt - 1] = img_for_tensor
        cnt += 1


# prepare the labels
print(images.shape)
encode = np.zeros(shape=(21*15, 15), dtype=float)
epochs = 100
batch_count = 1
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# labels for convolution model
for i in range(n_outputs):
        for j in range(21):
            encode[j + i*21][i] = 1

# iterating over the batches of the images
def next_batch(batch_s, iters):
    count = batch_s * iters
    return images[count-batch_s:count], encode[count-batch_s: count]

def val_next_batch(batch_s, iters):
    count = batch_s * iters
    return val_images[count-batch_s:count], encode[count-batch_s: count]

# reset_graph()
# training over CNN Model to get trained weights
for i in range(3*75):
    # training on a batch of 105 images
    if (batch_count > 3):
        batch_count = 1

    batch_X, batch_Y = next_batch(105, batch_count)
    val_batch_X, val_batch_Y = val_next_batch(105, batch_count)
    batch_count += 1
    _, loss = sess.run([optimizer, cost],
                    feed_dict={
                        X: batch_X,
                        y: batch_Y
                    })
    acc = sess.run(accuracy,
                         feed_dict={
                             X: val_batch_X,
                             y: val_batch_Y
                         })
    if i % 3 == 0:
        print(i)
        print('Mini batch Loss at one epoch : {:>10.4f} Validation accuracy{:0.6f}'.format(loss, acc))
        # gr_epoch.append(i/5)
        # gr_loss.append(loss_value)

print("Done!")


encoded = np.ndarray(shape=(15*21,), dtype=int)

for i in range(15):
    for j in range(21):
        encoded[j + i*21] = i
#
batch = tf.placeholder("float", None)                           # batch size only for RNN
lstm_op = data_lstm(X, batch)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

lstm_total_images = np.concatenate((images, val_images))

# passing trained weights from CNN model to RNN model 
x_total = sess.run(lstm_op, feed_dict={X: lstm_total_images[:630], batch: 630})

# separting the training and validation images
x_train, x_val = x_total[:315], x_total[315:]
y_train, y_val = encoded[:315], encoded[:315]

# preparing the cnn represtantion of images
x_test = sess.run(lstm_op, feed_dict={X: images_test[:], batch: 9})

def lstm_next_batch(batch_s, iters):
    count_total = batch_s * iters
    return x_train[count_total-batch_s:count_total], y_train[count_total-batch_s: count_total]

def val_lstm_next_batch(batch_s, iters):
    count_total = batch_s * iters
    return x_val[count_total-batch_s:count_total], y_val[count_total-batch_s: count_total]

# Start making the Network for RNN
# reset the graph and make sure the random numbers are always the same
# reset_graph()
# hyperparameters
n_neurons = 100
learning_rate = 0.001
n_epochs = 90
n_outputs = 15
n_steps = 12
n_inputs = 12

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.int32, [None])

# make one basic RNN cell that will be unrolled in time
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
logits1 = tf.layers.dense(state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits1)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Prediction
prediction = tf.nn.in_top_k(logits1, Y, 1)                  # CHECK ONCE
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# initialize the variables
init = tf.global_variables_initializer()

# train the RNN model which has inputs as learned weights from the CNN model
sess = tf.Session()
sess.run(init)
loss_list = []
acc_list = []

for epoch in range(n_epochs):
    for batch_size in range(7):
        batch_x, batch_y = lstm_next_batch(45, batch_size+1)
        batch_x_val, batch_y_val = val_lstm_next_batch(45, batch_size + 1)
        sess.run(optimizer, feed_dict={x: batch_x, Y: batch_y})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={x: batch_x, Y: batch_y})
        loss_list.append(loss_train)
        acc_list.append(acc_train)
        loss_val, acc_val = sess.run(
            [loss, accuracy], feed_dict={x: batch_x_val, Y: batch_y_val})
    batch_size = 0
    print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(epoch + 1, loss_train, acc_train))
    print('VALIDATION Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_val, acc_val))

# init = tf.global_variables_initializer()

#  train the model
# sess = tf.Session()
# sess.run(init)
# loss_list = []
# acc_list = []

# plot train loss vs epoch
# plt.figure(figsize=(18, 5))
# plt.subplot(1, 2, 1)
# plt.title('Train Loss vs Epoch', fontsize=15)
# plt.plot(np.arange(n_epochs), loss_list, 'r-')
# plt.xlabel('Epoch')
# plt.ylabel('Train Loss')
#
# # plot train accuracy vs epoch
# plt.subplot(1, 2, 2)
# plt.title('Train Accuracy vs Epoch', fontsize=15)
# plt.plot(np.arange(n_epochs), acc_list, 'b-')
# plt.xlabel('Epoch')
# plt.ylabel('Train Accuracy')
# plt.show()

n_predict = 5   # number to display prediction

# predicting using the model and count the correct predictions made
for i in range(n_predict):
    pred = sess.run(tf.argmax(logits1[i]), feed_dict={x: x_test, Y: labels_test})
    actual = labels_test[i]-1
    plt.imshow(np.squeeze(images_test[i]))
    plt.axis('off')
    plt.show()
    print('predict: ', pred)
    print('actual: ', actual)

