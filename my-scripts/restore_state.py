#Script to read from a saved checkpoint and run test
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys, math


save_file = './model.ckpt'
n_input = 784
n_classes = 10
tf.reset_default_graph()
print("input {}, classes {}, file {}".format(n_input, n_classes, save_file))
mnist = input_data.read_data_sets('.', one_hot = True)
#We want to read features, labels, weights and bias from the saved save_file
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
features = tf.placeholder(tf.float32,[None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
saver = tf.train.Saver()
#Define the accuracy
#Logits = features*weights + biases\
logits = tf.add( tf.matmul(features, weights), bias)

correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
batch_size = 1024*1024
#Now test
with tf.Session() as sess:
    saver.restore(sess, save_file)
    batches = math.ceil(mnist.test.num_examples/batch_size)
    for batch in range(batches):
        batch_features, batch_labels = mnist.test.next_batch(batch_size)
        test_accuracy = sess.run(accuracy, feed_dict = {features:mnist.test.images, labels: mnist.test.labels})
        print("Test accuracy: {}".format(test_accuracy))
