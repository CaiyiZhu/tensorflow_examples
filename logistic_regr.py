# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:54:17 2016

@author: csi-digital4

define a logistic regression using tensorflow,  not done yet
"""
from __future__ import division
import numpy as np
import tensorflow as tf
M = 10000
P = 10
X =  np.random.uniform(-1, 1, size=[M,P])
w_real = np.ones(shape=[P,1])
y = np.dot(X, w_real) + 0
y = np.reshape(np.array([1 if n >0 else 0 for n in y]), [M,1])

X_train = X[M//2:]
y_train = y[M//2:]
X_test = X[:M//2]
y_test = y[:M//2]

sess = tf.InteractiveSession()

x_ = tf.placeholder(tf.float32, shape = [None, P])
y_ = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.truncated_normal(shape = [10,1], stddev = 0.1))
b = tf.Variable(tf.zeros(shape = [1]))
y = tf.matmul(x_,w) + b
y = tf.nn.sigmoid(y)

#tmp = y.eval()
#y_hat = tf.convert_to_tensor(np.array([1 if n > 0.5 else 0 for n in tmp]))
#correct_prediction = tf.equal(y_hat,y_)  # still some problem

correct_prediction = tf.equal(y,y_)  # should map y to binary vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost = tf.reduce_mean(-(1-y_)*tf.log(1-y) - y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

sess.run(tf.initialize_all_variables())

for i in range(100):
    train_step.run(feed_dict = {x_:X_train, y_:y_train})
    
accuracy_test = accuracy.eval(feed_dict = {x_:X_test, y_: y_test})

print accuracy_test


