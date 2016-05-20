# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:37:28 2016

@author: csi-digital4
"""


# add a fully connected layer, accuracy increased by 5 percent

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

w1 = tf.Variable(tf.truncated_normal(shape = [784,100], stddev = 0.1))
b1 = tf.Variable(tf.zeros(shape = [100]))

# add a fully connected layer
y1 = tf.nn.relu(tf.matmul(x,w1) + b1)
w2 = tf.Variable(tf.truncated_normal(shape = [100,10], stddev = 0.1))
b2 = tf.Variable(tf.zeros(shape = [10]))

## add dropout 
#keep_rate = 1.0
#y1 = tf.nn.dropout(y1, keep_rate)

y_prob = tf.nn.softmax(tf.matmul(y1,w2) + b2)

cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_prob), reduction_indices = [1]))

correct_prediction = tf.equal(tf.arg_max(y_,1),tf.arg_max(y_prob,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict = {x:batch[0], y_:batch[1]})
    
test_accuracy = accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
print test_accuracy
