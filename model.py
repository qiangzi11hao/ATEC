# -*- coding:utf8 -*-

import pickle
from keras.models import Model
from keras.layers import Input, merge, Dense, Lambda, concatenate, Dropout, BatchNormalization,LSTM,Conv1D, Add
from keras.layers.embeddings import Embedding
from keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score
from layers import  Attention, Position_Embedding


def get_f1(y_true, y_pred):
    return  f1_score(y_true, y_pred)



def cnn_lstm_f1():
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    question1 = Input(shape=(20,))
    question2 = Input(shape=(20,))

    q1 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=20,
                   trainable=False)(question1)

    q2 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=20,
                   trainable=False)(question2)

    f_rnn = LSTM(30, return_sequences=True, implementation=1)
    b_rnn = LSTM(30, return_sequences=True, implementation=1, go_backwards=True)

    pos = Position_Embedding(mode='concat')
    att = Attention(20)

    q1 = BatchNormalization()(q1)
    qf_rnn = f_rnn(q1)
    qb_rnn = b_rnn(q1)

    q1_rnn = concatenate([qf_rnn, qb_rnn], axis=-1)
    q1_rnn = pos(q1_rnn)
    q1_rnn = concatenate([q1_rnn, att(q1_rnn)])

    q2 = BatchNormalization()(q2)
    af_rnn = f_rnn(q2)
    ab_rnn = b_rnn(q2)
    q2_rnn = concatenate([af_rnn, ab_rnn], axis=-1)
    q2_rnn = pos(q2_rnn)
    q2_rnn = concatenate([q2_rnn, att(q2_rnn)])

    # cnn
    cnns = [Conv1D(kernel_size=kernel_size,
                   filters=100,
                   activation='tanh',
                   padding='same') for kernel_size in [1, 2, 3, 5]]
    # qq_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
    q1_cnn = concatenate([cnn(q1_rnn) for cnn in cnns], axis=-1)
    # q2_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')
    q2_cnn = concatenate([cnn(q2_rnn) for cnn in cnns], axis=-1)

    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    maxpool.supports_masking = True
    q1_pool = Dropout(0.05)(maxpool(q1_cnn))
    q2_pool = Dropout(0.05)(maxpool(q2_cnn))
    merged1 = Dense(100, activation='relu')(q1_pool)
    merged2 = Dense(100, activation='relu')(q2_pool)
    merged = concatenate([merged1, merged2])

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def attention_lstm():
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    question1 = Input(shape=(15,))
    question2 = Input(shape=(15,))

    q1 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=15,
                   trainable=False)(question1)

    q2 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=15,
                   trainable=False)(question2)

    pos = Position_Embedding()
    f_rnn = LSTM(256, return_sequences=True, consume_less='mem')
    b_rnn = LSTM(256, return_sequences=True, consume_less='mem', go_backwards=True)

    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    maxpool.supports_masking = True

    q1 = pos(q1)
    q2 = pos(q2)
    qf_rnn = f_rnn(q1)
    qb_rnn = b_rnn(q1)
    # q1_rnn = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
    q1_rnn =  concatenate([qf_rnn, qb_rnn], axis=-1)


    af_rnn = f_rnn(q2)
    ab_rnn = b_rnn(q2)
    # q2_rnn = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
    q2_rnn =  concatenate([af_rnn, ab_rnn], axis=-1)

    att = Attention(20)


    q1_att = maxpool(att([q1_rnn, q1_rnn, q1_rnn]))
    q1 = Dense(200, activation='relu')(q1_att)

    q2_att = maxpool(attention([q2_rnn, q2_rnn, q2_rnn]))
    q2 = Dense(200, activation='relu')(q2_att)

    merged = concatenate([q1, q2])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(0)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
