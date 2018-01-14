import tensorflow as tf

class EncoderDecoder(object):
    def __init__(self, model='lstm', vocab_size=6110, embedding_size=128, hidden_size=128, batch_size=64, num_layers=2, infer=False):

        self.batch_size = batch_size
        if infer is True:
            self.batch_size = 1

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
        self.decoder_inputs = tf.placeholder(tf.int32, [self.batch_size, None], name='decoder_inputs')
        self.decoder_targets = tf.placeholder(tf.int32, [self.batch_size, None], name='decoder_targets')
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            encoder_inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs)

        with tf.variable_scope('encoder'):
            self.encoder_cell = model_cell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.mtcell_encoder = tf.contrib.rnn.MultiRNNCell([self.encoder_cell]*self.num_layers, state_is_tuple=True)
            self.initial_state_encoder = self.mtcell_encoder.zero_state(self.batch_size, tf.float32)
            encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.mtcell_encoder, encoder_inputs, initial_state=self.initial_state_encoder, dtype=tf.float32)
            del encoder_outputs

        with tf.variable_scope('decoder'):
            self.decoder_cell = model_cell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.mtcell_decoder = tf.contrib.rnn.MultiRNNCell([self.decoder_cell] * self.num_layers, state_is_tuple=True)
            decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(self.mtcell_decoder, decoder_inputs, initial_state=self.encoder_final_state, dtype=tf.float32)

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [self.vocab_size])

        output = tf.reshape(decoder_outputs, [-1, self.hidden_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.targets = tf.reshape(self.decoder_targets, [-1])

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
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

model = EncoderDecoder()
print 'Finish creating the encoder-decoder model.'

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

model_save_path = './ckpt-encoder-decoder/encoder-decoder.ckpt'
saver = tf.train.Saver(max_to_keep=10)
print 'Start Training ...\n'

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
                                     model.decoder_inputs: X,
                                     model.decoder_targets: y,
                                     model.keep_prob: 0.5,
                                 })
        logit, target = sess.run([model.logits, model.targets],
                                 feed_dict={
                                     model.encoder_inputs: X,
                                     model.decoder_inputs: X,
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

