from PIL import Image
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize

# dimensions of the image
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1



def reset_graph(seed = 2018):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# declaring array images for training, validation and testing
images = np.zeros((30*67, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
val_images = np.zeros((30*5, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

images_test = np.zeros((12, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)   # test images for LSTM model

# labels for the test array
labels_test = [i for i in range(1, 13)]  # 12 labels of the respective image

cnt = 1
count_same_image = 30
count_diff_image = 25
one_type_img = 67


# reading training images from the directory
for i in range(count_same_image):
    for j in range(one_type_img):
        read_path = os.getcwd() + '/Train_New/' + 'obj' + str(i+1) + '__' + str(count_diff_image) + '.jpg'
        img = imread(read_path)
        img_as_array = np.array(img)
        #
        # img_for_tensor = resize(img_as_array, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant',
        #                         preserve_range=True)
        # images.append(img_as_array)
        images[cnt - 1] = img_as_array
        cnt += 1
        count_diff_image += 5
        # image_labels[l] = i
        # l += 1
    count_diff_image = 25

cnt = 1
count_same_image = 30
count_diff_image = 0
one_type_img = 5


# reading validation images from the directory
for i in range(count_same_image):
    for j in range(one_type_img):
        read_path = os.getcwd() + '/Train_New/' + 'obj' + str(i+1) + '__' + str(count_diff_image) + '.jpg'
        img = imread(read_path)
        img_as_array = np.array(img)
        # img_for_tensor = resize(img_as_array, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant',
        #                         preserve_range=True)
        val_images[cnt - 1] = img_as_array
        # val_images.append(img_as_array)
        cnt += 1
        count_diff_image += 5
    count_diff_image = 0

# cnt = 1
path_test = '/home/smaph/PycharmProjects/autoencoder/Test/'

# reading test images from the directory
for i in range(12):
    img=imread(path_test+ str(i+1)+'.jpg')
    img_as_array = np.array(img)
    images_test[i] = img_as_array



print(images.shape)
# encode = np.zeros(shape=(72*30, 30), dtype=float)

batch_count = 1
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


encoded_train = np.ndarray(shape=(30*67,), dtype=int)
encoded_val = np.ndarray(shape=(30*5,), dtype=int)

# preparing the target labels for the training and validation set
for i in range(30):
    for j in range(67):
        encoded_train[j + i*67] = i

for i in range(30):
    for j in range(5):
        encoded_val[j + i*5] = i



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# prepare the training and validation data to feed during session
x_train, x_val = images[:], val_images[:]
y_train, y_val = encoded_train[:], encoded_val[:]

# reading training images in batches from the array
def lstm_next_batch(batch_s, iters):
    count_total = batch_s * iters
    return x_train[count_total-batch_s:count_total], y_train[count_total-batch_s: count_total]
#
# def val_lstm_next_batch(batch_s, iters):
#     count_total = batch_s * iters
#     return x_val[count_total-batch_s:count_total], y_val[count_total-batch_s: count_total]

# Start making the Network for RNN
# reset the graph and make sure the random numbers are always the same
reset_graph()
# hyperparameters
n_neurons = 600
learning_rate = 0.001
n_epochs = 152
n_outputs = 30
n_steps = 128
n_inputs = 128

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
prediction = tf.nn.in_top_k(logits1, Y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# initialize the variables
#
init = tf.global_variables_initializer()
#
# train the model
sess = tf.Session()
sess.run(init)
train_list = []
val_list = []
loss_train =0
acc_train=0
loss_val=0
acc_val=0

# saver object to save and restore the model and checkpoints
saver = tf.train.Saver(max_to_keep=2)

# returns a CheckpointState if the state was available, None otherwise
ckpt = tf.train.get_checkpoint_state('./')


if ckpt:
    print('Reading last checkpoint....')
    # saver = tf.train.import_meta_graph('u-net-50.meta')

# restoring the model from the last checkpoint
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print('Model restored')
else:
    print('Creating the checkpoint and models')

# start training the model
for epoch in range(n_epochs):
    for batch_size in range(5):
        batch_x, batch_y = lstm_next_batch(402, batch_size+1)
        sess.run(optimizer, feed_dict={x: batch_x, Y: batch_y})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={x: batch_x, Y: batch_y})

    batch_x_val, batch_y_val = x_val[:], y_val[:]
    loss_val, acc_val = sess.run( [loss, accuracy], feed_dict={x: batch_x_val, Y: batch_y_val})
    batch_size = 0
    train_list.append(loss_train)
    val_list.append(loss_val)
    if epoch % 50 == 0:
        saver.save(sess, './rnn-model', global_step=epoch, write_meta_graph=False)
    print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(epoch + 1, loss_train, acc_train))
    print('VALIDATION Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_val, acc_val))

#
# plot train loss vs epoch
plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1)
plt.title('Train Loss vs Epoch', fontsize=15)
plt.plot(np.arange(n_epochs), train_list, 'r-')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
# #
# # plot val loss vs epoch
plt.subplot(1, 2, 2)
plt.title('Validation Loss vs Epoch', fontsize=15)
plt.plot(np.arange(n_epochs), val_list, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.show()

n_predict = 12   # number to display prediction

# testing the model on the predictions
for i in range(n_predict):
    pred = sess.run(tf.argmax(logits1[i]), feed_dict={x: images_test})
    actual = labels_test[i]-1
    # plt.imshow(np.squeeze(images_test[i]))
    # plt.axis('off')
    # plt.show()
    print('predict: ', pred)
    print('actual: ', actual)

