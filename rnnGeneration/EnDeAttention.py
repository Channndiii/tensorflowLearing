import tensorflow as tf
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

_start = word2id[u'[']
_pad = word2id[u' ']

class EDA(object):
    def __init__(self, model='lstm', vocab_size=6110, embedding_size=128, hidden_size=128, batch_size=64, num_layers=2, infer=False):

        self.batch_size = batch_size
        if infer:
            self.batch_size = None

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = model

        if self.model == 'rnn':
            model_cell = tf.contrib.rnn.BasicRNNCell
        elif self.model == 'gru':
            model_cell = tf.contrib.rnn.GRUCell
        elif self.model == 'lstm':
            model_cell = tf.contrib.rnn.BasicLSTMCell

        self.encoder_inputs = tf.placeholder(tf.int32, [self.batch_size, None], name='encoder_inputs')
        # self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [self.batch_size], name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(tf.int32, [self.batch_size, None], name='decoder_targets')
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            encoder_inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs)

        self.encoder_cell = model_cell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        self.encoder_cell = tf.contrib.rnn.DropoutWrapper(cell=self.encoder_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        self.mtcell_encoder = tf.contrib.rnn.MultiRNNCell([self.encoder_cell]*self.num_layers, state_is_tuple=True)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=self.mtcell_encoder,
                                            cell_bw=self.mtcell_encoder,
                                            inputs=encoder_inputs,
                                            sequence_length=self.encoder_inputs_length,
                                            dtype=tf.float32)
        )

        self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        def get_encoder_final_state(num_layers):
            final_state = []
            for i in range(num_layers):
                layer_final_state_c = tf.concat((encoder_fw_final_state[i].c, encoder_bw_final_state[i].c), 1)
                layer_final_state_h = tf.concat((encoder_fw_final_state[i].h, encoder_bw_final_state[i].h), 1)
                layer_final_state = tf.contrib.rnn.LSTMStateTuple(c=layer_final_state_c, h=layer_final_state_h)
                final_state.append(layer_final_state)
            return tuple(final_state)
        self.encoder_final_state = get_encoder_final_state(self.num_layers)

        self.decoder_cell = model_cell(self.hidden_size*2, forget_bias=1.0, state_is_tuple=True)
        self.mtcell_decoder = tf.contrib.rnn.MultiRNNCell([self.decoder_cell]*self.num_layers, state_is_tuple=True)

        _batch_size, encoder_max_time = tf.unstack(tf.shape(self.encoder_inputs))

        # decoder_lengths = self.encoder_inputs_length - 1
        decoder_lengths = self.encoder_inputs_length
        with tf.variable_scope('decoder_softmax'):
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size*2, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [self.vocab_size])

        start_signal = _start * tf.ones([_batch_size], dtype=tf.int32, name='start_signal')
        start_signal = tf.nn.embedding_lookup(embedding, start_signal)
        pad_signal = _pad * tf.ones([_batch_size], dtype=tf.int32, name='pad_signal')
        pad_signal = tf.nn.embedding_lookup(embedding, pad_signal)

        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_lengths)
            initial_input = start_signal
            initial_cell_state = self.encoder_final_state
            initial_cell_output = None
            initial_loop_state = None
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)
        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            def get_next_input():
                output_logits = tf.matmul(previous_output, softmax_w) + softmax_b
                # prediction = tf.argmax(output_logits, axis=1)
                prediction = tf.argmax(tf.nn.softmax(output_logits), axis=1)
                next_input = tf.nn.embedding_lookup(embedding, prediction)
                return next_input
            elements_finished = (time >= decoder_lengths)
            finished = tf.reduce_all(elements_finished)
            input = tf.cond(finished, lambda: pad_signal, get_next_input)
            state = previous_state
            output = previous_output
            loop_state = None
            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)
        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.mtcell_decoder, loop_fn)
        decoder_outputs = tf.transpose(decoder_outputs_ta.stack(), [1, 0, 2])
        print decoder_outputs
        decoder_outputs = tf.reshape(decoder_outputs, (-1, self.hidden_size*2))

        self.decoder_logits = tf.matmul(decoder_outputs, softmax_w) + softmax_b
        print self.decoder_logits
        # self.decoder_predictions = tf.argmax(self.decoder_logits, axis=1)
        self.decoder_predictions = tf.nn.softmax(self.decoder_logits)

        # def loop(prev, _):
        #     prev = tf.matmul(prev, softmax_w) + softmax_b
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return tf.nn.embedding_lookup(embedding, prev_symbol)
        #
        # self.attn_length = 5
        # self.attn_size = 32
        # self.attention_states = tf.Variable(tf.truncated_normal(shape=[self.batch_size, self.attn_length, self.attn_size], dtype=tf.float32, stddev=0.1, name='attention'))
        #
        # encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.mtcell_encoder, inputs, initial_state=self.initial_state_encoder, dtype=tf.float32)
        # decoder_outputs, self.final_state = tf.contrib.legacy_seq2seq.attention_decoder(inputs, self.encoder_final_state, self.attention_states, self.mtcell_decoder, loop_function=loop)
        #
        # decoder_outputs, self.decoder_final_state = tf.contrib.legacy_seq2seq.embedding_attention_decoder(
        #     decoder_inputs=[i for i in tf.transpose(self.X_inputs)],
        #     initial_state=self.encoder_final_state,
        #     attention_states=self.attention_states,
        #     embedding_size=self.embedding_size,
        #     cell=self.mtcell_decoder,
        #     num_symbols=self.vocab_size,
        #     num_heads=1,
        #     feed_previous=True,
        #     dtype=tf.float32
        # )

        self.targets = tf.reshape(self.decoder_targets, [-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.decoder_logits],
            [self.targets],
            [tf.ones([tf.shape(self.targets)[0]], dtype=tf.float32)]
        )
        self.cost = tf.reduce_mean(loss)
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        max_grad_norm = 5
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        self.saver = tf.train.Saver(max_to_keep=10)

model = EDA()
print 'Finish creating the EDA model.'

import time
import batch_generator
data_train = batch_generator.get_data_train()

config = tf.ConfigProto()
sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)

batch_num = int(data_train.data_size / model.batch_size)

max_epoch =100
display_num = 5
display_batch = int(batch_num / display_num)

print 'data_size={}, vocab_size={}'.format(data_train.data_size, model.vocab_size)
print 'batch_size={}, embedding_size={}'.format(model.batch_size, model.embedding_size)
print 'batch_num={}, display_batch={}'.format(batch_num, display_batch)

model_save_path = './ckpt-eda/eda.ckpt'
saver = tf.train.Saver(max_to_keep=10)

for epoch in xrange(max_epoch):

    _lr = 0.002 * (0.97 ** epoch)
    sess.run(tf.assign(model.lr, _lr))

    print 'EPOCH %d, lr=%g' % (epoch+1, _lr)
    start_time = time.time()
    all_loss = 0.0
    show_loss = 0.0
    for batch in xrange(batch_num):
        X, y = data_train.next_batch(model.batch_size)
        train_loss, _ = sess.run([model.cost, model.train_op],
                                 feed_dict={
                                     model.encoder_inputs: X,
                                     model.encoder_inputs_length: map(len, X),
                                     model.decoder_targets: y,
                                     model.keep_prob: 0.5,
                                 })
        all_loss += train_loss
        show_loss += train_loss

        if (batch+1) % display_batch == 0:
            print "\tbatch %d, training loss=%g" % (batch+1, show_loss / display_batch)
            show_loss = 0.0

    mean_loss = all_loss *1.0 / batch_num
    if (epoch + 1) % 4 == 0:
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print 'the save path is ', save_path

    print "Epoch training %d, loss=%g, speed=%g s/epoch" % (data_train.data_size, mean_loss, time.time()-start_time)


