import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('weights'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([out_size]) + 0.1, name='b')
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

with tf.name_scope('hidden_layer'):
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
with tf.name_scope('output_layer'):
    prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    writer = tf.summary.FileWriter("./log", sess.graph)

    sess.run(init)