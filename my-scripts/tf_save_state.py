# File which trains a 1-layer network on MNIST dataset and saves it in a file
import tensorflow as tf
import math as math
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# The file path to save the data
save_file = './model.ckpt'

#Set hyper parameters
learning_rate = 0.001
epochs = 100
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

#Import data
mnist = input_data.read_data_sets('.', one_hot = True)

features = tf.placeholder(tf.float32,[None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([n_input, n_classes]))
bias = tf.Variable(tf.truncated_normal([n_classes]))

#Logits = features*weights + biases\
logits = tf.add( tf.matmul(features, weights), bias)

#Define loss and optimizer for this loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))
optimizer = tf.train\
                .AdamOptimizer(learning_rate = learning_rate)\
                .minimize(loss)

#Define function to compute accuracy of the model
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()
batch_size = 128

with tf.Session() as sess:
    # Initialize all the Variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        #Create batches
        batches = math.ceil(mnist.train.num_examples/batch_size)
        for batch in range(batches):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={features:batch_features, labels: batch_labels})
        #Every 10 epochs, compute accuracy so far
        if epoch % 10 == 0:
            current_accuracy = sess.run(accuracy, feed_dict={features:mnist.validation.images, labels: mnist.validation.labels})
            print("Epoch {:<3} | validation accuracy :{}".format(epoch, current_accuracy))
    saver.save(sess, save_file)
print("Model trained and saved in {}".format(save_file))
