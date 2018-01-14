# coding=utf-8

import re
import numpy as np

with open('./data/msr_train.txt') as fr:
    texts = fr.read().decode('gbk')
sentences = texts.split('\r\n')

def clean(s):
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

texts = u''.join(map(clean, sentences))
# print texts[:300]
sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)

def get_Xy(sentence):

    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags
    return None

data = []
label = []
print 'Start creating words and tags data ...'
for sentence in sentences:
    result = get_Xy(sentence)
    if result:
        data.append(result[0])
        label.append(result[1])
print len(data)

import pandas as pd
df_data = pd.DataFrame({'words': data, 'tags': label}, index=range(len(data)))
# df_data['sentence_length'] = df_data['words'].apply(lambda words: len(words))
# print df_data.head(1)

# import matplotlib.pyplot as plt
# df_data['sentence_length'].hist(bins=100)
# plt.xlim(0, 100)
# plt.xlabel('sentence_length')
# plt.ylabel('sentence_num')
# plt.title('Distribution of the Length of Sentence')
# plt.show()

from itertools import chain
all_words = list(chain(*df_data['words'].values))

sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index

set_ids = range(1, len(set_words)+1)
tags = ['x', 's', 'b', 'm', 'e']
tag_ids = range(len(tags))

word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)

max_len = 32
def X_padding(words):
    ids = list(word2id[words])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids

def y_padding(tags):
    ids = list(tag2id[tags])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)

X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))

print 'X.shape={}, y.shape={}'.format(X.shape, y.shape)
print 'Example of words: ', df_data['words'].values[0]
print 'Example of X: ', X[0]
print 'Example of tags: ', df_data['tags'].values[0]
print 'Example of y: ', y[0]

import pickle
with open('./data/data.pkl', 'wb') as fw:
    pickle.dump(X, fw)
    pickle.dump(y, fw)
    pickle.dump(word2id, fw)
    pickle.dump(id2word, fw)
    pickle.dump(tag2id, fw)
    pickle.dump(id2tag, fw)
print '*** Finish saving the data. ***'







