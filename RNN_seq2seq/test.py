import tensorflow as tf
import data_helper

encoder_sentence_length = 10
decoder_sentence_length = 10

print 'Building vocab ...'
all_input_sentences = data_helper.get_input_sentence()
all_target_sentences = data_helper.get_target_sentence()
encoder_vocab, encoder_reverse_vocab, encoder_vocab_size = data_helper.build_vocab(all_input_sentences)
decoder_vocab, decoder_reverse_vocab, decoder_vocab_size = data_helper.build_vocab(all_target_sentences, is_target=True)

max_epoch = 2000
layer_num = 2
hidden_size = 30
embedding_size = 30

encoder_inputs = tf.placeholder(tf.int32, [None, encoder_sentence_length], name='encoder_inputs')
encoder_sequence_length = tf.placeholder(tf.int32, [None], name='encoder_sequence_length')
decoder_inputs = tf.placeholder(tf.int32, [None, decoder_sentence_length+1], name='decoder_inputs')

with tf.variable_scope('embedding'):
    encoder_embedding = tf.get_variable('encoder_embedding', initializer=tf.random_uniform([encoder_vocab_size+1, embedding_size]))
    decoder_embedding = tf.get_variable('decoder_embedding', initializer=tf.random_uniform([decoder_vocab_size+2, embedding_size]))

with tf.variable_scope('encoder'):
    encoder_inputs_T = tf.transpose(encoder_inputs, [1, 0])
    encoder_inputs_emb = tf.nn.embedding_lookup(encoder_embedding, encoder_inputs_T)
    # encoder_inputs_emb = tf.nn.embedding_lookup(encoder_embedding, encoder_inputs)
    encoder_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    mtcell_encoder = tf.contrib.rnn.MultiRNNCell([encoder_cell]*layer_num, state_is_tuple=True)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        cell=mtcell_encoder,
        inputs=encoder_inputs_emb,
        sequence_length=encoder_sequence_length,
        dtype=tf.float32,
        time_major=True
    )
    del encoder_outputs

with tf.variable_scope('decoder'):
    decoder_inputs_T = tf.transpose(decoder_inputs, [1, 0])
    decoder_inputs_emb = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs_T)
    # decoder_inputs_emb = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
    decoder_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    mtcell_decoder = tf.contrib.rnn.MultiRNNCell([decoder_cell]*layer_num, state_is_tuple=True)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        cell=mtcell_decoder,
        inputs=decoder_inputs_emb,
        initial_state=encoder_final_state,
        dtype=tf.float32,
        time_major=True
    )

with tf.variable_scope('softmax'):
    softmax_w = tf.get_variable('softmax_w', initializer=tf.random_uniform([hidden_size, decoder_vocab_size + 2]))
    softmax_b = tf.get_variable('softmax_b', initializer=tf.random_uniform([decoder_vocab_size + 2]))

output = tf.reshape(decoder_outputs, [-1, hidden_size])
logits = tf.reshape(tf.matmul(output, softmax_w) + softmax_b, [decoder_sentence_length+1, -1, decoder_vocab_size+2])
logits_T = tf.transpose(logits, [1, 0, 2])
predictions = tf.argmax(logits_T, axis=2)
# logits = tf.reshape(tf.matmul(output, softmax_w) + softmax_b, [-1, decoder_sentence_length+1, decoder_vocab_size+2])
# predictions = tf.argmax(logits, axis=2)

labels = tf.one_hot(decoder_inputs_T, decoder_vocab_size+2)
# labels = tf.one_hot(decoder_inputs, decoder_vocab_size+2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels,
    logits=logits
))

train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

input_batches = data_helper.get_input_batches()
target_batches = data_helper.get_target_batches()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(max_epoch):
        all_preds = []
        epoch_loss = 0
        for input_batch, target_batch in zip(input_batches, target_batches):
            input_token_ids = []
            target_token_ids = []
            input_sentence_lengths = []

            for input_sentence in input_batch:
                input_sentence, sentenceLength = data_helper.sentence2id(input_sentence, vocab=encoder_vocab, max_sentence_length=encoder_sentence_length)
                input_token_ids.append(input_sentence)
                input_sentence_lengths.append(sentenceLength)

            for target_sentence in target_batch:
                target_sentence = data_helper.sentence2id(target_sentence, vocab=decoder_vocab, max_sentence_length=decoder_sentence_length, is_target=True)
                target_token_ids.append(target_sentence)

            batch_preds, batch_loss, _ = sess.run(
                [predictions, loss, train_op],
                feed_dict={
                    encoder_inputs: input_token_ids,
                    encoder_sequence_length: input_sentence_lengths,
                    decoder_inputs: target_token_ids,
                }
            )

            epoch_loss += batch_loss
            all_preds.append(batch_preds)

        if epoch % 400 == 0:
            print 'Epoch={}'.format(epoch)
            print 'Epoch loss={}'.format(epoch_loss)
            for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                for input_sentence, target_sentence, pred in zip(input_batch, target_batch, batch_preds):
                    print '\t' + input_sentence
                    print '\t => ' + data_helper.id2sentence(pred, reverse_vocab=decoder_reverse_vocab)
                    print '\tCorrect answer:' + target_sentence