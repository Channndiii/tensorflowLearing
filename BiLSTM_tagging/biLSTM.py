import tensorflow as tf
import numpy as np

timestep_size = max_len = 32
vocab_size = 5159
input_size = embedding_size = 64
class_num = 5
hidden_size = 128
layer_num = 2
max_grad_norm = 5.0

lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='softmax_weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='softmax_biases')

X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')

def bi_lstm(X_inputs): # X_inputs [128, 32]

    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32) # embedding [5159, 64]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs) # X_inputs [128, 32] --> inputs [128, 32, 64]

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    # hidden_size determines the size of cell's output, which has no relation with the size of inputs
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

    mtcell_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*layer_num, state_is_tuple=True)
    mtcell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*layer_num, state_is_tuple=True)

    init_state_fw = mtcell_fw.zero_state(batch_size, tf.float32)
    # there are #batch_size sentences fed to the model each time
    init_state_bw = mtcell_bw.zero_state(batch_size, tf.float32)

    # # [batch_size, timestep_size, embedding_size] --> timestep_size * [batch_size, embedding_size]
    # # inputs = tf.unstack(inputs, timestep_size, 1)
    # inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
    # outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mtcell_fw, mtcell_bw, inputs, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32)
    # output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size*2])

    with tf.variable_scope('bidirectional_rnn'):
        outputs_fw = list()
        state_fw = init_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = mtcell_fw(inputs[:, timestep, :], state_fw) # output_fw [128, 128]
                outputs_fw.append(output_fw) # outputs_fw [32, 128, 128]

        outputs_bw = list()
        state_bw = init_state_bw
        with tf.variable_scope('bw'):
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = mtcell_bw(inputs[:, timestep, :], state_bw) # output_bw [128, 128]
                outputs_bw.append(output_bw) # outputs_bw [32, 128, 128]

        outputs_bw = tf.reverse(outputs_bw, [0]) # outputs_bw_re  [32, 128, 128]

        output = tf.concat([outputs_fw, outputs_bw], 2) # output_con  [32, 128, 256]
        output = tf.transpose(output, [1, 0, 2]) # output_T  [128, 32, 256]
        output = tf.reshape(output, [-1, hidden_size*2]) # output  [4096, 256]

    softmax_w = weight_variable([hidden_size*2, class_num])
    softmax_b = bias_variable([class_num])
    logits = tf.matmul(output, softmax_w) + softmax_b

    return logits

# embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
# # [batch_size, timestep_size] --> [batch_size, timestep_size, embedding_size]
# inputs = tf.nn.embedding_lookup(embedding, X_inputs)
#
# lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
# lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
#
# lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
# lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
#
# mtcell_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*layer_num, state_is_tuple=True)
# mtcell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*layer_num, state_is_tuple=True)
#
# init_state_fw = mtcell_fw.zero_state(batch_size, tf.float32)
# init_state_bw = mtcell_bw.zero_state(batch_size, tf.float32)
#
# inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
# outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mtcell_fw, mtcell_bw, inputs, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32)
# output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size*2])
#
# with tf.variable_scope('bidirectional_rnn'):
#     outputs_fw = list()
#     state_fw = init_state_fw
#     with tf.variable_scope('fw'):
#         for timestep in range(timestep_size):
#             if timestep > 0:
#                 tf.get_variable_scope().reuse_variables()
#             (output_fw, state_fw) = mtcell_fw(inputs[:, timestep, :], state_fw)
#             outputs_fw.append(output_fw)
#
#     outputs_bw = list()
#     state_bw = init_state_bw
#     with tf.variable_scope('bw'):
#         inputs_bw = tf.reverse(inputs, [1])
#         for timestep in range(timestep_size):
#             if timestep > 0:
#                 tf.get_variable_scope().reuse_variables()
#             (output_bw, state_bw) = mtcell_bw(inputs_bw[:, timestep, :], state_bw)
#             outputs_bw.append(output_bw)
#
#     outputs_bw_re = tf.reverse(outputs_bw, [0])
#
#     output_con = tf.concat([outputs_fw, outputs_bw_re], 2)
#     output_t = tf.transpose(output_con, [1, 0, 2])
#     output = tf.reshape(output_t, [-1, hidden_size*2])
#
# softmax_w = weight_variable([hidden_size*2, class_num])
# softmax_b = bias_variable([class_num])
# logits = tf.matmul(output, softmax_w) + softmax_b

y_pred = bi_lstm(X_inputs) # y_pred [4096, 5]
# y_pred = logits
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]), logits=y_pred))

tvars = tf.trainable_variables()
print "tvars", [x.name for x in tvars]

grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
print 'Finish creating the bi-lstm model.'

config = tf.ConfigProto()
sess = tf.Session(config=config)

import time
import batch_generator

def test_epoch(dataset):
    _batch_size = 500
    fetches = [accuracy, cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    _costs = 0.0
    _accs = 0.0
    for i in xrange(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {
            X_inputs: X_batch,
            y_inputs: y_batch,
            lr: 1e-5,
            batch_size: _batch_size,
            keep_prob: 1.0
        }
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost

init = tf.global_variables_initializer()
sess.run(init)

data_train = batch_generator.get_data_train()
data_valid = batch_generator.get_data_valid()
data_test = batch_generator.get_data_test()
# model_save_path = './ckpt/bi-lstm.ckpt'
model_save_path = './ckpt_new/bi-lstm.ckpt'

decay = 0.85
max_epoch = 5
max_max_epoch = 12

tr_batch_size = 128
display_num = 5
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
display_batch = int(tr_batch_num / display_num)

saver = tf.train.Saver(max_to_keep=10)

for epoch in xrange(max_max_epoch):
    _lr = 1e-4
    if epoch > max_epoch:
        _lr = _lr * ((decay) ** (epoch - max_epoch))
    print 'EPOCH %d, lr=%g' % (epoch+1, _lr)

    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_costs = 0.0
    show_accs = 0.0

    for batch in xrange(tr_batch_num):
        fetches = [accuracy, cost, train_op]
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        feed_dict = {
            X_inputs: X_batch,
            y_inputs: y_batch,
            lr: _lr,
            batch_size: tr_batch_size,
            keep_prob: 0.5
        }
        _acc, _cost, _ = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost

        # if batch == 0:
        #     print "Shape:"
        #
        #     print "X_inputs ", X_batch.shape # [128, 32]
        #     print "y_inputs ", y_batch.shape # [128, 32]
        #     print "embedding_shape ", sess.run(tf.shape(embedding)) # [5159, 64]
        #     print "embedding_inputs ", sess.run(tf.shape(inputs), feed_dict) # [128, 32, 64]
        #
        #     print "inputs_fw ", sess.run(tf.shape(inputs), feed_dict) # [128, 32, 64]
        #     print "output_fw ", sess.run(tf.shape(output_fw), feed_dict) # [128, 128]
        #     print "state_fw ", sess.run(tf.shape(state_fw), feed_dict) # [2, 2, 128, 128]
        #     print "outputs_fw ", sess.run(tf.shape(outputs_fw), feed_dict) # [32, 128, 128]
        #
        #     print "inputs_bw ", sess.run(tf.shape(inputs_bw), feed_dict) # [128, 32, 64]
        #     print "outputs_bw ", sess.run(tf.shape(outputs_bw), feed_dict) # [32, 128, 128]
        #
        #     print "outputs_bw_re ", sess.run(tf.shape(outputs_bw_re), feed_dict) # [32, 128, 128]
        #     print "output_con ", sess.run(tf.shape(output_con), feed_dict) # [32, 128, 256]
        #     print "output_t ", sess.run(tf.shape(output_t), feed_dict) # [128, 32, 256]
        #     print "outputs ", sess.run(tf.shape(outputs), feed_dict) #
        #     print "output ", sess.run(tf.shape(output), feed_dict) # [4096, 256]
        #
        #     print "y_pred ", sess.run(tf.shape(y_pred), feed_dict) # [4096, 5]

        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost = test_epoch(data_valid)
            print '\ttraining acc=%g, cost=%g;  valid acc=%g, cost=%g ' % (
                show_accs / display_batch,
                show_costs / display_batch,
                valid_acc,
                valid_cost
            )
            show_accs = 0.0
            show_costs = 0.0

    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num

    if (epoch + 1) % 3 == 0:
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print 'The save path is ', save_path

    print '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost)
    print 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (
        data_train.y.shape[0],
        mean_acc,
        mean_cost,
        time.time()-start_time
    )
print '**TEST RESULT:'
test_acc, test_cost = test_epoch(data_test)
print '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)

