# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

def batch_norm_wrapper(inputs, is_training, decay = 0.5):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 1e-3)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 1e-3)


def build_graph(is_training):

    InputSize = 784
    OutputSize = 10
    x = tf.placeholder(tf.float32, shape=[None, InputSize])
    y_ = tf.placeholder(tf.float32, shape=[None, OutputSize])

    w1 = tf.Variable(tf.truncated_normal([InputSize, 100], stddev=1.0 / math.sqrt(float(InputSize))))
    b1 = tf.Variable(tf.zeros([100]))
    linear1 = tf.matmul(x, w1) + b1
    bn1 = batch_norm_wrapper(linear1, is_training)
    l1 = tf.nn.relu(bn1)

    w2 = tf.Variable(tf.truncated_normal([100, 100], stddev=1.0 / math.sqrt(float(100))))
    b2 = tf.Variable(tf.zeros([100]))
    linear2 = tf.matmul(l1, w2) + b2
    bn2 = batch_norm_wrapper(linear2, is_training)
    l2 = tf.nn.relu(bn2)

    w3 = tf.Variable(tf.truncated_normal([100, 10], stddev=1.0 / math.sqrt(float(10))))
    b3 = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(l2, w3) + b3)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_, train_step, accuracy, y, tf.train.Saver()

if "__main__" == __name__:
    x, y_, train_step, accuracy, _, saver = build_graph(False)
    mnist = input_data.read_data_sets("/home/yuhongsheng/TensorFlowCode/mnist_dir/mnist", one_hot=True)
    writer = tf.summary.FileWriter(logdir="/tmp/pycharm_project_141/", graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/tmp/pycharm_project_141/bn_save")
        res = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print res

    '''
    acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(1000):
            batch_x, batch_y = mnist.train.next_batch(128)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
            if i % 100 == 0:
                res = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print "%d_%s_%f"%(i, "acc:", res)
                acc.append(res)
        saved_model = saver.save(sess, "bn_save")
    '''


