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
sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
decoder_inputs = tf.placeholder(tf.int32, [None, decoder_sentence_length+1], name='decoder_inputs')

encoder_inputs_T = tf.transpose(encoder_inputs, [1, 0])
decoder_inputs_T = tf.transpose(decoder_inputs, [1, 0])

with tf.variable_scope('embedding_rnn_seq2seq'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    decoder_outputs, decoder_final_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
        encoder_inputs=tf.unstack(encoder_inputs_T),
        decoder_inputs=tf.unstack(decoder_inputs_T),
        cell=rnn_cell,
        num_encoder_symbols=encoder_vocab_size+1,
        num_decoder_symbols=decoder_vocab_size+2,
        embedding_size=embedding_size,
        feed_previous=True
    )

logits = tf.stack(decoder_outputs)
predictions = tf.transpose(tf.argmax(tf.stack(decoder_outputs), axis=-1), [1, 0])

labels = tf.one_hot(decoder_inputs_T, decoder_vocab_size+2)

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
            sentence_lengths = []

            for input_sentence in input_batch:
                input_sentence, sentenceLength = data_helper.sentence2id(input_sentence, vocab=encoder_vocab, max_sentence_length=encoder_sentence_length)
                input_token_ids.append(input_sentence)
                sentence_lengths.append(sentenceLength)

            for target_sentence in target_batch:
                target_token_ids.append(data_helper.sentence2id(target_sentence, vocab=decoder_vocab, max_sentence_length=decoder_sentence_length, is_target=True))

            batch_preds, batch_loss, _ = sess.run(
                [predictions, loss, train_op],
                feed_dict={
                    encoder_inputs: input_token_ids,
                    sequence_length: sentence_lengths,
                    decoder_inputs: target_token_ids
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