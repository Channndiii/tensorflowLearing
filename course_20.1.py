import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
# batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
batch_size = tf.placeholder(tf.int32)

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), # (28, 128)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])) # (128, 10)
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])), # 128
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes])) # 10
}

def LSTM(X, weights, biases):

    X = tf.reshape(X, [-1, n_inputs]) # (128batch, 28steps, 28inputs) --> (128*28, 28inputs)
    X_in = tf.matmul(X, weights['in']) + biases['in'] # X_in --> (128*28, 128hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) # X_in --> (128batch, 28steps, 128hidden)

    # X_in = X

    # Cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # lstm cell is divided into two parts (c_state, m_state)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    # Hidden layer for outputs as the final results
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # or unpack to list [(batch, outputs)..] * step
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2])) # transpose [0, 1, 2] --> [1, 0, 2]
    # !!!In this example, states is the last outputs!!!
    results = tf.matmul(outputs[-1], weights['out']) + biases['out'] # --> (128, 10)

    return results

def GRU(X, weights, biases):

    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    gru_cell = tf.contrib.rnn.GRUCell(n_hidden_units)
    init_state = gru_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(gru_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

pred = LSTM(x, weights, biases)
# pred = GRU(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    # step = 0
    # while step * batch_size < training_iters:
    #
    #     batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #     batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    #
    #     sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
    #
    #     if step % 20 == 0:
    #
    #         train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    #         print "training accuracy: %g" % train_accuracy
    #
    #     step += 1

    _batch_size = 128

    for step in range(2000):

        batch_xs, batch_ys = mnist.train.next_batch(_batch_size)
        batch_xs = batch_xs.reshape([-1, n_steps, n_inputs])

        sess.run(train_op, feed_dict={
            x: batch_xs,
            y: batch_ys,
            batch_size: _batch_size
        })

        if step % 200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
                batch_size: _batch_size
            })

            print "Iter %d, step %d, training accuracy %g" % (mnist.train.epochs_completed, step, train_accuracy)

    test_xs = mnist.test.images
    test_ys = mnist.test.labels

    test_xs = test_xs.reshape([-1, n_steps, n_inputs])

    print "test accuracy: %g" % sess.run(accuracy, feed_dict={
        x: test_xs,
        y: test_ys,
        batch_size: test_xs.shape[0]
    })



