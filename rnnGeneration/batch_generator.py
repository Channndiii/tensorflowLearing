import pickle
import numpy as np

with open('./data.pkl', 'rb') as fr:
    X = pickle.load(fr)
    word2id = pickle.load(fr)
    id2word = pickle.load(fr)

class BatchGenerator(object):
    def __init__(self, data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    @property
    def data_size(self):
        return self._data_size

    def next_batch(self, batch_size):
        start = self._index_in_epoch

        if start + batch_size >= self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed += 1
            self._index_in_epoch = batch_size
            full_batch_features, full_batch_labels = self.data_batch(0, batch_size)
            return full_batch_features, full_batch_labels

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features, full_batch_labels = self.data_batch(start, end)
            # if self._index_in_epoch == self._data_size:
            #     self._index_in_epoch = 0
            #     self._epochs_completed += 1
            #     np.random.shuffle(self._data_index)
            return full_batch_features, full_batch_labels

    def data_batch(self, start, end):
        batches = []
        for i in range(start, end):
            batches.append(X[self._data_index[i]])

        length = max(map(len, batches))

        xdata = np.full((end - start, length), word2id[u' '], np.int32)
        for row in range(end - start):
            xdata[row][:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]
        return xdata, ydata

print 'Creating the data generator ...'
data_train = BatchGenerator(len(X))
print 'Finish creating the data generator. For RNN.'

def get_data_train():
    return data_train

# batch_size = 64
# batch_num = len(X) // batch_size
# for batch in range(batch_num):
#     x, y = data_train.next_batch(batch_size)