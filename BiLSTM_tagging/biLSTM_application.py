# coding=utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import re

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

X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    # return tf.Variable(initial, name='softmax_weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    # return tf.Variable(initial, name='softmax_biases')

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

y_pred = bi_lstm(X_inputs) # y_pred [4096, 5]

config = tf.ConfigProto()
sess = tf.Session(config=config)

saver = tf.train.Saver()
best_model_path = './ckpt/bi-lstm.ckpt-12'
saver.restore(sess, best_model_path)
print 'Finish restoring the bi-lstm model.'

# data_train = batch_generator.get_data_train()
# X_tt, y_tt = data_train.next_batch(2)
# print 'X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape
# print 'X_tt = ', X_tt
# print 'y_tt = ', y_tt

transfer = {
    'be': 0.5,
    'bm': 0.5,
    'eb': 0.5,
    'es': 0.5,
    'me': 0.5,
    'mm': 0.5,
    'sb': 0.5,
    'ss': 0.5,
}

transfer = {i: np.log(transfer[i]) for i in transfer.keys()}

def viterbi(nodes):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for layer in xrange(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for node_now in nodes[layer].keys():
            sub_paths = {}
            for path_last in paths_.keys():
                if path_last[-1] + node_now in transfer.keys():
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + transfer[path_last[-1] + node_now]
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()
            node_subpath = sr_subpaths.index[-1]
            node_value = sr_subpaths[-1]
            paths[node_subpath] = node_value
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()
    return sr_paths.index[-1]

import pickle

with open('./data/data.pkl', 'rb') as fr:
    X = pickle.load(fr)
    y = pickle.load(fr)
    word2id = pickle.load(fr)
    id2word = pickle.load(fr)
    tag2id = pickle.load(fr)
    id2tag = pickle.load(fr)

def text2ids(text):
    words = list(text)
    ids = list(word2id[words])
    if len(ids) >= max_len:
        print '输出片段超过%d部分无法处理' % max_len
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    ids = np.asarray(ids).reshape([-1, max_len])
    return ids

def simple_cut(text):
    if text:
        text_len = len(text)
        X_batch = text2ids(text)
        fetches = [y_pred]
        feed_dict = {
            X_inputs: X_batch,
            lr: 1.0,
            batch_size: 1,
            keep_prob: 1.0
        }
        _y_pred = sess.run(fetches, feed_dict)[0][:text_len]
        nodes = [dict(zip(['s', 'b', 'm', 'e'], each[1:])) for each in _y_pred]
        tags = viterbi(nodes)
        words = []
        for i in range(len(text)):
            if tags[i] in ['s', 'b']:
                words.append(text[i])
            else:
                words[-1] += text[i]
        return words
    else:
        return []

def cut_word(sentence):
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!“”""]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()]))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:]))
    return result

# sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
sentence = u'天舟一号是我国自主研制的首艘货运飞船，由于它只运货，不送人，所以被形象地称为太空“快递小哥”。它采用两舱式结构，直径较小的是推进舱，直径较大的为货物舱。其最大直径达到3.35米，飞船全长10.6米，载荷能力达到了6.5吨，满载货物时重13.5吨。如果此次满载的话，它很可能将成为中国发射进入太空的质量最大的有效载荷。甚至比天宫二号空间实验室还大，后者全长10.4米，直径同为3.35米，质量为8.6吨。'
result = cut_word(sentence)
rss = ''
for each in result:
    rss = rss + each + ' / '
print rss
