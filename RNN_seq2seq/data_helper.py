import re
import collections

input_batches = [
    ['Hi What is your name?', 'Nice to meet you!'],
    ['Which programming language do you use?', 'See you later.'],
    ['Where do you live?', 'What is your major?'],
    ['What do you want to drink?', 'What is your favorite food?']]

target_batches = [
    ['Hi this is Chandi.', 'Nice to meet you too!'],
    ['I like Python.', 'Bye Bye.'],
    ['I live in Shenzhen, China.', 'My major is computer science.'],
    ['Coke please!', 'Seafood!']]

def get_input_batches():
    return input_batches

def get_target_batches():
    return target_batches

def get_input_sentence():
    all_input_sentences = []
    for input_batch in input_batches:
        all_input_sentences.extend(input_batch)
    return all_input_sentences

def get_target_sentence():
    all_target_sentences = []
    for target_batch in target_batches:
        all_target_sentences.extend(target_batch)
    return all_target_sentences

def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens

def build_vocab(sentences, is_target=False, max_vocab_size=None):
    word_counter = collections.Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)
    if max_vocab_size is None:
        max_vocab_size = len(word_counter)
    if is_target:
        vocab['_GO'] = 0
        vocab['_PAD'] = 1
        vocab_idx = 2
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    else:
        vocab['_PAD'] = 0
        vocab_idx = 1
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    for key, value in vocab.items():
        reverse_vocab[value] = key
    return vocab, reverse_vocab, max_vocab_size

# encoder_vocab, encoder_reverse_vocab, encoder_vocab_size = build_vocab(all_input_sentences)
# decoder_vocab, decoder_reverse_vocab, decoder_vocab_size = build_vocab(all_target_sentences, is_target=True)

def token2id(word, vocab):
    return vocab[word]

encoder_sentence_length = 10
decoder_sentence_length = 10
batch_size = 4

def sentence2id(sentence, vocab, max_sentence_length, is_target=False):
    tokens = tokenizer(sentence)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2id(token, vocab) for token in tokens] + [token2id('_PAD', vocab)] * pad_length
    else:
        return [token2id(token, vocab) for token in tokens] + [token2id('_PAD', vocab)] * pad_length, current_length

def id2token(id, reverse_vocab):
    return reverse_vocab[id]

def id2sentence(ids, reverse_vocab):
    return ' '.join([id2token(id, reverse_vocab) for id in ids])

