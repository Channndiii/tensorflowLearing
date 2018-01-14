import tensorflow as tf
import EncoderDecoder

model = EncoderDecoder.EncoderDecoder(infer=True)

config = tf.ConfigProto()
sess = tf.Session(config=config)

saver = tf.train.Saver()
best_model_path = './ckpt-encoder-decoder/encoder-decoder.ckpt-100'
saver.restore(sess, best_model_path)
print 'Finish restoring the encoder-decoder model.'

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

_state = sess.run(model.mtcell_encoder.zero_state(1, tf.float32))
x = np.array([[word2id[u'[']]])

[_probs, _state] = sess.run([model.probs, model.encoder_final_state], feed_dict={
    model.encoder_inputs: x,
    model.decoder_inputs: x,
    model.keep_prob: 1.0,
    model.initial_state_encoder: _state
})

word = wordSample(_probs)
# word = id2word[np.argmax(_probs)]
# print word

poetry = ''
while word != u']':
    poetry += word
    x = np.zeros((1, 1))
    x[0, 0] = word2id[word]
    [_probs, _state] = sess.run([model.probs, model.encoder_final_state], feed_dict={
        model.encoder_inputs: x,
        model.decoder_inputs: x,
        model.keep_prob: 1.0,
        model.initial_state_encoder: _state
    })
    word = wordSample(_probs)
print poetry