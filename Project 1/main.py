import re
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM


ALLOWED_RELATIONS = [
    'Cause-Effect',
    'Component-Whole',
    'Entity-Destination',
    'Entity-Origin',
    # 'Instrument-Agency',
    # 'Member-Collection',
    # 'Message-Topic',
    # 'Product-Producer',
    'Other',
]
NUM_WORDS = 20000


def load_data(data_file):
    with open(data_file) as f:

        data_x = []
        data_y = []
        for i in f:
            line = i.strip()
            # every 3rd or 4th line can be a ignored
            if line == '' or line.startswith('Comment:'):
                continue
            # if line starts with a number, it is a sentence
            elif re.match(r'^\d+', line):
                # extract sentence surrounded by quotes
                sentence = re.search(r'"(.*)"', line).group(1)
                cleaned_sentence = re.sub(r'<([^>]+)>', '', sentence)
                data_x.append(cleaned_sentence)
            # otherwise, it is a label
            else:
                cleaned_line = re.sub(r'\(.*', '', line)
                data_y.append(cleaned_line)
        return remove_invalid_relations(data_x, data_y)


def remove_invalid_relations(data_x, data_y):
    # loop though data_y
    # set invalid relations to Other
    for i in range(len(data_y)):
        if data_y[i] not in ALLOWED_RELATIONS:
            data_y[i] = 'Other'

    # covert to pandas dataframe
    data_x = pd.DataFrame(data_x, columns=['sentence'])
    data_y = pd.Series(data_y, name='label')
    return data_x, data_y


def process_data(raw_x, raw_y, raw_x_test, raw_y_test):
    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(raw_y)
    y = to_categorical(y, num_classes=len(ALLOWED_RELATIONS))
    y_test = le.transform(raw_y_test)
    y_test = to_categorical(y_test, num_classes=len(ALLOWED_RELATIONS))

    # tokenize sentences with keras tokenizer
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;=?@[]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(raw_x['sentence'].values)
    x = tokenizer.texts_to_sequences(raw_x['sentence'].values)
    x = pad_sequences(x, maxlen=100)

    x_test = tokenizer.texts_to_sequences(raw_x_test['sentence'].values)
    x_test = pad_sequences(x_test, maxlen=100)

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    return x, y, x_test, y_test


def data_statistics(data_x, data_y):
    print(data_y.describe())
    print()
    print(data_y.value_counts())
    print()
    print(data_y.value_counts(normalize=True))
    print()


def main():
    x_data, y_data = load_data('./dataset/SemEval2010_task8_training/TRAIN_FILE.TXT')
    x_test_data, y_test_data = load_data('./dataset/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
    print("Training data:")
    data_statistics(x_data, y_data)
    print("Test data:")
    data_statistics(x_test_data, y_test_data)
    x_train, y_train, x_test, y_test = process_data(x_data, y_data, x_test_data, y_test_data)

    # Make Model
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(NUM_WORDS, embedding_dim, input_length=x_train.shape[1]))
    model.add(Bidirectional(LSTM(128, dropout=0.7, recurrent_dropout=0.7)))
    model.add(Dense(len(ALLOWED_RELATIONS), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=ALLOWED_RELATIONS, y=y_data)
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        # tf.tensorboard.TensorBoard(log_dir='./logs')
    ]

    # model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2,
    #           callbacks=callbacks, class_weight=class_weights)

    # Save Model
    # model.save('model.h5')

    # Load Model
    model.load_weights('model.h5')

    # model.evaluate(x_test, y_test)

    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=ALLOWED_RELATIONS)
    y_pred = y_pred.idxmax(axis=1)
    y_pred.name = 'predicted_label'

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_data, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ALLOWED_RELATIONS)
    disp.plot()
    # plt.show()

    combined_data = pd.concat([x_test_data, y_test_data, y_pred], axis=1)

    incorrect_predictions = combined_data[(combined_data['label'] != combined_data['predicted_label']) &
                                          (combined_data['label'] != 'Other') &
                                          (combined_data['predicted_label'] != 'Other')]
    random_samples = incorrect_predictions.sample(n=50)
    pd.set_option('display.max_colwidth', None)
    random_samples.to_csv('incorrect_predictions.csv', index=False)


if __name__ == "__main__":
    main()
