# -*- coding:utf8 -*-

import numpy as np
import io
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from utils.langconv import *

FEATURE_WORDS = set([u'花呗', u'借呗'])
HUA_BEI = set([u'花贝', u'花吧', u'花臂', u'发倍', u'好呗', u'花被', u'花坝', u'花宝贝', u'画吧'])
JIE_BEI = set([u'借吧', u'借贝', u'戒备', u'接呗', u'借本'])
WANG_SHANG_DAI = set([u'网上贷'])

MAX_SEQUENCE_LENGTH = 20
BALANCED = None

jieba.load_userdict('data/user_dict.txt')


class Vocab(object):
    def __init__(self, file, simplified=True, correct=True):
        _, self.q1_char, self.q2_char, self.q1_word, self.q2_word, self.coll, self.label = self.get_data(file, simplified, correct, BALANCED)
        self.q_word = self.q1_word + self.q2_word
        # self.analyze(self.q1_word, self.q2_word)
        self.embedding = 0
        self.embedding_c = {}
        self.word_index = {}
        self.char_index = {}
        self.nb_chars = 0
        self.nb_words = 0
        self.tokenizer_word = 'add'
        self.tokenizer_char = None

    def get_data(self, file, simplified=True, corrected=True, balanced=None):
        df = pd.read_csv(file, header=None, sep='\t')
        df = df.sort_index(by=3, ascending=False)
        if balanced == 'delete':
            df = df[:37370]
            df = shuffle(df)

        index, q1, q2, label = df[0].tolist(), df[1].tolist(), df[2].tolist(), map(float, df[3].tolist())
        if balanced == 'add':
            q1 = q1 + q1[:18685] * 3
            q2 = q2 + q2[:18685] * 3
            label = label + label[:18685] * 3
            for _ in range(9052):
                index = np.random.randint(18685)
                q1.append(q1[index])
                q2.append(q2[index])
                label.append(label[index])

        if simplified:
            q1 = list(map(self.cht_to_chs, q1))
            q2 = list(map(self.cht_to_chs, q2))
        if corrected:
            q1 = list(map(self.correction, q1))
            q2 = list(map(self.correction, q2))
        q1_word = map(list, map(jieba.cut, q1))
        q2_word = map(list, map(jieba.cut, q2))
        q1_char = map(list, q1)
        q2_char = map(list, q2)

        def join_(l):
            return ' '.join(l).encode("utf-8").strip()

        def collocate(q1, q2):
            common_words = []
            for q1_w, q2_w in zip(q1, q2):
                ret = [single_word for single_word in q1_w if single_word in q2_w]
                common_words.append(ret)
            return  common_words

        coll = collocate(q1_word, q2_word)
        q1_word = map(join_, q1_word)
        q2_word = map(join_, q2_word)
        coll = map(join_, coll)
        return index, q1_char, q2_char, q1_word, q2_word, coll, label

    def cht_to_chs(self, line):
        line = Converter('zh-hans').convert(line.decode("utf-8"))
        line.encode('utf-8')
        return line

    def correction(self, q):
        for word in FEATURE_WORDS:
            if word in q:
                return q
        for word in HUA_BEI:
            q = q.replace(word, u'花呗')
        for word in JIE_BEI:
            q = q.replace(word, u'借呗')
        for word in WANG_SHANG_DAI:
            q = q.replace(word, u'网商贷')
        return q

    def load_embedding_word(self, path):
        self.tokenizer_word = Tokenizer()
        self.tokenizer_word.fit_on_texts(self.q_word)
        self.word_index = self.tokenizer_word.word_index
        print("Words in index: %d" % len(self.word_index))
        embeddings_index = {}
        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for i, line in enumerate(fin):
            if i == 1200000:
                break
            tokens = line.rstrip().split(' ')
            embeddings_index[tokens[0]] = list(map(float, tokens[1:]))
        self.nb_words = len(self.word_index)
        self.embedding = np.random.rand(self.nb_words + 1, 300)
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word.decode('utf-8'))
            if embedding_vector is not None:
                self.embedding[i] = embedding_vector
        # print('Null word embeddings: %d' % np.sum(np.sum(self.embedding, axis=1) == 0))

    def to_sequence_word(self, question, padding=True, max_len=MAX_SEQUENCE_LENGTH):
        seq = self.tokenizer_word.texts_to_sequences(question)
        if padding:
            seq = pad_sequences(seq, maxlen=max_len)
        return seq

    def load_embedding_char(self, path):
        self.tokenizer_char = Tokenizer()
        self.tokenizer_char.fit_on_texts(self.q1_char+self.q2_char)
        self.char_index = self.tokenizer_char.word_index
        print("Chars in index: %d" % len(self.char_index))
        embeddings_index = {}
        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for i, line in enumerate(fin):
            if i == 1200000:
                break
            tokens = line.rstrip().split(' ')
            embeddings_index[tokens[0]] = list(map(float, tokens[1:]))
        self.nb_chars = len(self.char_index)
        self.embedding_c = np.random.rand(self.nb_chars + 1, 300)
        for word, i in self.char_index.items():
            embedding_vector = embeddings_index.get(word.encode('utf-8'))
            if embedding_vector is not None:
                self.embedding_c[i] = embedding_vector

    def to_sequence_char(self, question, padding=True, max_len=MAX_SEQUENCE_LENGTH):
        seq = self.tokenizer_char.texts_to_sequences(question)
        if padding:
            seq = pad_sequences(seq, maxlen=max_len)
        return seq

    def analyze(self, q1, q2):
        both = []
        either = []
        neither = []
        for i in range(len(q1)):
            set1 = set(q1[i].decode('utf-8').split())
            set2 = set(q2[i].decode('utf-8').split())
            if set1 & set2 & FEATURE_WORDS:
                both.append({'q1': q1[i], 'q2': q2[i]})
            elif (set1 | set2) & FEATURE_WORDS:
                either.append({'q1': q1[i], 'q2': q2[i]})
            else:
                neither.append({'q1': q1[i], 'q2': q2[i]})
        print(len(both), len(either), len(neither))

        q = q1+q2
        with open('candidate.txt', 'w') as fin:
            for i in range(len(q)):
                if not (set(q[i].decode('utf-8').split()) & FEATURE_WORDS):
                    fin.write(q[i]+'\n')


if __name__ == '__main__':
    vocab = Vocab('data/data_all.csv')
    vocab.load_embedding_word('data/sgns.merge.word')
    label = vocab.label
    print(len(label), sum(label))
    # 18685 102477
    # either 4717   3877
