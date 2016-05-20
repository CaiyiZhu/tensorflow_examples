# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:39:45 2016

@author: csi-digital4
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.InteractiveSession()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784]) # declare x and y
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10])) # declare the weight matrix and the bias
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables()) # initialize all variables

y = tf.nn.softmax(tf.matmul(x,W) + b) # define the softmax layer

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # define the cost function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # define the optimizer

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # define the prediction

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # define the accuracy




for i in range(1000): # define the training process
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})  

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # define the prediction

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # define the accuracy

print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # the testing accuracy