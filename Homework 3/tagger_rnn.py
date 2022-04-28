import os
import sys

import numpy as np

from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input, Bidirectional, LSTM, Activation, TimeDistributed
from keras import backend as backend


def load_corpus(path):
    if not os.path.isdir(path):
        sys.exit("Input path is not a directory")

    sentence_list = []
    max_length = 0
    for filename in os.listdir(path):
        # Iterates over files in directory
        with open(path + filename, 'r') as file:
            for line in file:
                words = line.lower().split()
                word_tags = []
                for word in words:
                    word_tags += [tuple(word.split('/'))]
                if word_tags:
                    sentence_list += [word_tags]
                max_length = max(max_length, len(words))
    return sentence_list, max_length


def create_dataset(sentences, max_length):
    word_to_index = {"PAD": 0, "DNE": 1}
    tag_to_index = {"PAD": 0}
    train_X = []
    train_y = []
    for sentence in sentences:
        word_list = []
        tag_list = []
        for word, tag in sentence:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
            if tag not in tag_to_index:
                tag_to_index[tag] = len(tag_to_index)
            word_list.append(word_to_index[word])
            tag_list.append(tag_to_index[tag])
        train_X.append(np.pad(word_list, (0, max_length - len(word_list)), constant_values=0))
        train_y.append(np.pad(tag_list, (0, max_length - len(tag_list)), constant_values=0))
    return np.array(train_X), np.array(train_y), word_to_index, tag_to_index


def index_test_sentence(sentence, word_to_index, max_length):
    int_sentence = np.zeros((1, max_length))
    for i, word in enumerate(sentence):
        if word in word_to_index:
            int_sentence[0, i] = word_to_index[word]
        else:
            int_sentence[0, i] = word_to_index["DNE"]
    return int_sentence


def to_categorical(tags, tag_count):
    shape = *tags.shape, tag_count
    one_hot_tags = np.zeros(shape=shape)
    for i, sentence in enumerate(tags):
        for j, tag in enumerate(sentence):
            one_hot_tags[i, j, tag] = 1

    return one_hot_tags


def from_categorical(one_hot_tags, tag_to_index):
    tag_list = []
    reverse_tag_dict = {i: t for t, i in tag_to_index.items()}
    for one_hot in one_hot_tags[0]:
        index = np.argmax(one_hot)
        if index == 0:
            return tag_list
        tag_list.append(reverse_tag_dict[index])
    return tag_list


def padless_accuracy(y_true, y_pred):
    true_class = backend.argmax(y_true, axis=-1)
    pred_class = backend.argmax(y_pred, axis=-1)

    ignore_mask = (pred_class != 0)
    matches = backend.cast((true_class == pred_class) & ignore_mask, 'int32')
    ignore_mask = backend.cast(ignore_mask, 'int32')
    accuracy = backend.sum(matches)/backend.maximum(backend.sum(ignore_mask), 1)
    return accuracy


def define_model(input_size, word_count, tag_count):
    model = Sequential()
    model.add(Input(shape=(input_size, )))
    model.add(Embedding(word_count, 128))
    model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_count)))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam_v2.Adam(0.001),
        metrics=['accuracy']
    )

    return model


def test(model, test_sentences, word_to_index, tag_to_index, max_length):
    for sentence in test_sentences:
        if not isinstance(sentence, list):
            sentence = sentence.split()
        int_sentence = index_test_sentence(sentence, word_to_index, max_length)
        one_hot_tags = model.predict(int_sentence)
        tag_list = from_categorical(one_hot_tags, tag_to_index)
        print(' '.join(sentence))
        print(tag_list[:len(sentence)])


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage tagger_rnn.py <path to corpus dir>")
    corpus_dir = sys.argv[1]
    if corpus_dir[-1] != '/':
        corpus_dir += '/'
    sentence_list, max_length = load_corpus(corpus_dir)
    train_X, train_y, word_to_index, tag_to_index = create_dataset(sentence_list, max_length)
    word_count = len(word_to_index)
    tag_count = len(tag_to_index)
    one_hot_train = to_categorical(train_y, tag_count)
    model = define_model(max_length, word_count, tag_count)
    model.fit(train_X, one_hot_train, verbose=1, batch_size=128, epochs=40, validation_split=0.2)

    test_sentences = [
        "the planet jupiter and its moons are in effect a mini solar system .",
        "computers process programs accurately .",
    ]
    test(model, test_sentences, word_to_index, tag_to_index, max_length)


if __name__ == "__main__":
    main()
