import tensorflow as tf

max_grad_norm = 5
vocab_size = 6110
# batch_size = 64
batch_size = 1
embedding_size = 128

keep_prob = tf.placeholder(tf.float32)
X_inputs = tf.placeholder(tf.int32, [batch_size, None], name='X_input')
# y_inputs = tf.placeholder(tf.int32, [batch_size, None], name='y_input')

def RNN(X_inputs, model='lstm', hidden_size=128, num_layers=2):

    embedding = tf.get_variable('embedding', [vocab_size, embedding_size], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)

    if model == 'rnn':
        model_cell = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        model_cell = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        model_cell = tf.contrib.rnn.BasicLSTMCell

    cell = model_cell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    mtcell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
    initial_state = mtcell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(mtcell, inputs, initial_state=initial_state, dtype=tf.float32)

    output = tf.reshape(outputs, [-1, hidden_size])

    softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size])
    softmax_b = tf.get_variable('softmax_b', [vocab_size])

    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, final_state, probs, mtcell, initial_state

_, final_state, probs, mtcell, initial_state = RNN(X_inputs)

config = tf.ConfigProto()
sess = tf.Session(config=config)

saver = tf.train.Saver()
best_model_path = './ckpt/rnnGeneration.ckpt-100'
saver.restore(sess, best_model_path)
print 'Finish restoring the RNN model.'

import numpy as np
import pickle
with open('./data.pkl', 'rb') as fr:
    X = pickle.load(fr)
    word2id = pickle.load(fr)
    id2word = pickle.load(fr)

def wordSample(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return id2word[sample]

_state = sess.run(mtcell.zero_state(1, tf.float32))
x = np.array([[word2id[u'[']]])

[_probs, _state] = sess.run([probs, final_state], feed_dict={
    X_inputs: x,
    keep_prob: 1.0,
    initial_state: _state
})

word = wordSample(_probs)
# word = id2word[np.argmax(_probs)]
# print word

poetry = ''
while word != u']':
    poetry += word
    x = np.zeros((1, 1))
    x[0, 0] = word2id[word]
    [_probs, _state] = sess.run([probs, final_state], feed_dict={
        X_inputs: x,
        keep_prob: 1.0,
        initial_state: _state
    })
    word = wordSample(_probs)
print poetry