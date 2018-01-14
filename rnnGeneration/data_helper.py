# coding=utf-8

poetryList = []
with open('./poetry.txt') as fr:
    for line in fr:
        line = line.decode('utf-8')
        poetry = line.strip().split(u':')[1]
        poetry = poetry.replace(u' ', u'')
        if u'_' in poetry or u'(' in poetry or u'（' in poetry or u'《' in poetry or u'[' in poetry:
            continue
        if len(poetry) < 5 or len(poetry) > 79:
            continue
        poetry = u'[' + poetry + u']'
        poetryList.append(poetry)

poetryList = sorted(poetryList, key=lambda poetry: len(poetry))
print 'poetry_size={}'.format(len(poetryList))

import pandas as pd
df_data = pd.DataFrame({'poetry': poetryList}, index=range(len(poetryList)))
# print df_data.head()

from itertools import chain
all_words = list(chain(*df_data['poetry'].values))
all_words.append(u' ')
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(len(set_words))
# all_words = []
# for poetry in poetryList:
#     all_words.extend(list(poetry))
#
# counter = collections.Counter(all_words).most_common()
# set_words = [word[0] for word in counter] + [' ']
# print 'vocab_size={}'.format(len(set_words))
#
word2id = pd.Series(range(len(set_words)), index=set_words)
id2word = pd.Series(set_words, index=range(len(set_words)))
# print word2id.head()
vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)

def poetry2vec(poetry):
    words = list(poetry)
    ids = list(word2id[words])
    return ids

df_data['X'] = df_data['poetry'].apply(poetry2vec)
print df_data.head()

import pickle
X = list(df_data['X'].values)

with open('./data.pkl', 'wb') as fw:
    pickle.dump(X, fw)
    pickle.dump(word2id, fw)
    pickle.dump(id2word, fw)
print '** Finish saving the data. **'




