import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
learning_rate = 0.00001
epochs = 10
batch_size = 128
test_valid_size = 256
n_classes = 10
dropout = 0.75

# Patch size = 5x5
# Input image size = 28*28*1

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    # Layer 1 - input images from mnist of 28*28*1. going to 14*14*32
    cov1 = conv2d(x, weights['wc1'], biases['bc1'], 1)
    cov1 = maxpool2d(cov1, k=2)  # k=2 reduces size of output to half

    # Layer2 - input 14*14*32, going to 7*7*64
    cov2 = conv2d(cov1, weights['wc2'], biases['bc2'], 1)
    cov2 = maxpool2d(cov2, k=2)

    # Fully connected layer - 7*7*64 to 1024
    fc1 = tf.reshape(cov2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output -> class prediction from 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


if __name__ == '__main__':
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Model
    logits = conv_net(x, weights, biases, keep_prob)

    # Loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
        .minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)

        for epoch in range(epochs):
            for batch in range(mnist.train.num_examples // batch_size):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

                # Calculate batch loss and accuracy
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})

                valid_acc = sess.run(accuracy, feed_dict={
                    x: mnist.validation.images[:test_valid_size],
                    y: mnist.validation.labels[:test_valid_size],
                    keep_prob: 1.})
                print('Epoch {:>2}, Batch {:>3} -'
                      'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                    epoch + 1,
                    batch + 1,
                    loss,
                    valid_acc))
        print("Done training. Now testing")
        # Calculate Test Accuracy
        test_acc = sess.run(accuracy, feed_dict={
            x: mnist.test.images[:test_valid_size],
            y: mnist.test.labels[:test_valid_size],
            keep_prob: 1.})
        print('Testing Accuracy: {}'.format(test_acc))
