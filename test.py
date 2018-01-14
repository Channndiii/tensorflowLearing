# coding=utf-8
# import pandas as pd
#
# df_data = pd.DataFrame({
#     'A': [1.0, 2.0, 3.0, 4.0],
#     'B': pd.Series(1, index=range(4))
# })
#
# # print list(df_data['A']).index(3.0)
#
# vc = pd.Series(df_data['A'].values).value_counts()
# i = vc.index
# print vc
# print i

# import tensorflow as tf
# import numpy as np
#
# # x = np.array([
# #               [[1, 2, 3],
# #                [4, 5, 6]],
# #               [[7, 8, 9],
# #                [10, 11, 12]]
# #                             ])
# # X = tf.constant(x)
#
# embedding = tf.get_variable("embedding", [5, 5], dtype=tf.float32)
#
# # y = tf.argmax(X, 0)
# # correct_prediction = tf.equal(y, [1, 1, 0, 0])
# # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     # print sess.run(tf.shape(X))
#     # print sess.run(y)
#     # print sess.run(correct_prediction)
#     # print sess.run(accuracy)
#     # print sess.run(tf.shape(X))
#     # print sess.run(X)
#     # print sess.run(tf.reverse(X, [1]))
#     print sess.run(embedding)
# import numpy as np
# transfer = {
#     'be': 0.5,
#     'bm': 0.5,
#     'eb': 0.5,
#     'es': 0.5,
#     'me': 0.5,
#     'mm': 0.5,
#     'sb': 0.5,
#     'ss': 0.5,
# }
# transfer = {i: np.log(transfer[i]) for i in transfer.keys()}
# print transfer.keys()
# print transfer.items()
# print transfer.values()

# import re
#
# not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
#
# sentence = u'人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
#
# for seg_sign in not_cuts.finditer(sentence):
#     print seg_sign.start(), seg_sign.end()

import numpy as np

# np.random.seed(111)

# print [np.random.randint(low=0, high=3) for i in range(10)]

# x = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

# y = np.copy(x)
#
#
# y[:, :-1] = x[:, 1:]

# print y
# print np.reshape(y, [-1])

# x = np.array([[0.1, 0.2, 0.3, 0.4]])
#
# t = np.cumsum(x)
# s = np.sum(x)
# tmp_1 = np.random.rand(1)
# tmp_2 = tmp_1 * s
# sample = int(np.searchsorted(t, tmp_2))
#
# print x.shape
# print t
# print s
# print tmp_1
# print tmp_2
# print sample

# print np.zeros((1, 1))
# import tensorflow as tf
# x = tf.constant([[1, 2],
#                  [3, 4]])
# with tf.Session() as sess:
#     # print sess.run(tf.reshape(tf.transpose(x), [-1]))
#     tmp = tf.transpose(x)
#     print [i for i in tmp]

import tensorflow as tf
# import numpy as np
# import pickle
# with open('/home/chandi/PycharmProjects/tensorflowLearning/rnnGeneration/data.pkl', 'rb') as fr:
#     X = pickle.load(fr)
#     word2id = pickle.load(fr)
#     id2word = pickle.load(fr)
#
# start_signal = word2id[u'['] * tf.ones([64], dtype=tf.int32, name='start_signal')
# with tf.Session() as sess:
#     print 'ss={}'.format(sess.run(start_signal))

# x = np.array([[word2id[u'[']]*64])
# print 'x.shape={}'.format(np.shape(x))

# initial_elements_finished = (0 >= 10)
# print initial_elements_finished


# elements_finished = (5 >= 10)
# finished = tf.reduce_all(elements_finished)
# input = tf.cond(finished, lambda: 0, lambda: 1)
#
# with tf.Session() as sess:
#     print sess.run(input)

a = tf.constant([1, 2, 3, 4])
with tf.Session() as sess:
    print sess.run(a-1)