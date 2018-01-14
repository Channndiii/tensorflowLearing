import tensorflow as tf

max_grad_norm = 5
vocab_size = 6110
batch_size = 64
embedding_size = 128

lr = tf.Variable(0.0, trainable=False)
keep_prob = tf.placeholder(tf.float32)
X_inputs = tf.placeholder(tf.int32, [batch_size, None], name='X_input')
y_inputs = tf.placeholder(tf.int32, [batch_size, None], name='y_input')

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

    return logits

logits = RNN(X_inputs)
targets = tf.reshape(y_inputs, [-1])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [logits],
    [targets],
    [tf.ones([tf.shape(targets)[0]], dtype=tf.float32)]
)
cost = tf.reduce_mean(loss)

tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

print 'Finish creating the model.'

import time
import batch_generator
data_train = batch_generator.get_data_train()

config = tf.ConfigProto()
sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)

batch_num = int(data_train.data_size / batch_size)

max_epoch =100
display_num = 5
display_batch = int(batch_num / display_num)

print 'data_size={}, vocab_size={}'.format(data_train.data_size, vocab_size)
print 'batch_size={}, embedding_size={}'.format(batch_size, embedding_size)
print 'batch_num={}, display_batch={}'.format(batch_num, display_batch)

# model_save_path = './ckpt/rnnGeneration.ckpt'
model_save_path = './ckpt-new/rnnGeneration.ckpt'
saver = tf.train.Saver(max_to_keep=10)

for epoch in xrange(max_epoch):

    _lr = 0.002 * (0.97 ** epoch)
    sess.run(tf.assign(lr, _lr))

    print 'EPOCH %d, lr=%g' % (epoch+1, _lr)
    start_time = time.time()
    all_loss = 0.0
    show_loss = 0.0
    for batch in xrange(batch_num):
        X, y = data_train.next_batch(batch_size)
        train_loss, _ = sess.run([cost, train_op],
                                 feed_dict={
                                     X_inputs: X,
                                     y_inputs: y,
                                     keep_prob: 0.5,
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