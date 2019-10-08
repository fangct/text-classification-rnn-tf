import os
from collections import Counter
import numpy as np
import jieba
from tqdm import tqdm

def segment_data(path, save_path):

    with open(path, 'r', encoding='utf-8') as f_r:
        with open(save_path, 'w', encoding='utf-8') as f_w:
            for line in f_r:
                words = []
                line = line.strip().split('\t')
                label = line[0]
                context = line[1]
                words.extend(jieba.cut(context))

                f_w.write(label + '\t')
                for word in words:
                    if len(word.strip()) != 0:
                        f_w.write(word + ' ')
                f_w.write('\n')


def load_stopwords(path='./data/stopwords.txt'):
    stopwords = []
    with open(path, 'r', encoding='utf-8') as f_r:
        for word in f_r:
            stopwords.append(word.strip())
    return stopwords

def create_vocab(data_paths, vocab_path, vocab_size):
    print('*' * 80)
    if not os.path.exists(vocab_path):
        print('create vocabulary...')
        vocab = {}
        vocab['PAD'] = 0

        stopwords = load_stopwords('./data/stopwords.txt')
        words = []
        for path in data_paths:
            with open(path, 'r', encoding='utf-8') as f_r:
                for line in tqdm(f_r.readlines()):
                    label, content = line.strip().split('\t')
                    words.extend(content.split(' '))

        remove_stopwords = [word for word in words if word not in stopwords]
        count = Counter(remove_stopwords)
        word_counts = count.most_common(vocab_size-2)

        for word, _ in word_counts:
            vocab[word] = len(vocab)

        vocab['UNK'] = len(vocab)
        print('{} words totally, but save {} words in vocabulary.'.format(len(words), vocab_size))

        print('save vocab into {}'.format(vocab_path))
        with open(vocab_path, 'w', encoding='utf-8') as f_w:
            for word, idx in vocab.items():
                f_w.write(word + '\t' + str(idx) + '\n')
    else:
        print('vocabulary file is already exists.')

def load_vocab(path):
    print('*' * 80)
    if os.path.exists(path):

        print('load vocabulary from file {}'.format(path))
        vocab = {}
        with open(path, 'r', encoding='utf-8') as f_r:
            for line in f_r:
                word, idx = line.strip().split('\t')
                vocab[word] = idx
        return vocab
    else:
        raise ValueError('vocabulary file not exists, please create first.')


def create_embedding_mat(vocab, pre_train_path, embed_mat_path):
    print('*' * 80)
    if not os.path.exists(embed_mat_path):
        print('create embedding mat from {}'.format(pre_train_path))
        embed_mat = np.zeros(shape=(len(vocab), 100), dtype=float)
        count = 0
        with open(pre_train_path, 'r', encoding='utf-8') as f_r:
                for line in f_r:
                    word = line.strip().split(' ')[0]
                    vector = list(map(float, line.strip().split(' ')[1:]))
                    if word in vocab.keys():
                        count += 1
                        embed_mat[int(vocab[word]), :] = vector

        print('{}/{} words found in pre_train embedding file {}.'.format(count, len(vocab), pre_train_path))
        np.savez_compressed(embed_mat_path, embedding=embed_mat)

    else:
        print('embedding mat is already exists.')

def load_embedding_mat(path):
    print('*' * 80)
    print('loading embedding mat from file {}'.format(path))
    return np.load(path)['embedding']


def word_to_ids(vocab, data_path, target_path):
    print('*' * 80)
    if not os.path.exists(target_path):
        print('transform word into index.')

        with open(data_path, 'r' , encoding='utf-8') as f_r:
            with open(target_path, 'w', encoding='utf-8') as f_w:
                for line in f_r:
                    label, content = line.strip().split('\t')
                    f_w.write(label + '\t')
                    for word in content.strip().split(' '):
                        if word in vocab.keys():
                            f_w.write(vocab[word] + ' ')
                        else:
                            f_w.write(vocab['UNK'] + ' ')
                    f_w.write('\n')

    else:
        print('word2ids file {} is already exists...'.format(target_path))


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return cat_to_id


def preprocess(data_paths, vocab_path, pre_train_path, embed_mat_path, vocab_size):

    create_vocab(data_paths, vocab_path, vocab_size)
    vocab = load_vocab(vocab_path)
    create_embedding_mat(vocab, pre_train_path, embed_mat_path)
    word_to_ids(vocab, data_paths[0], target_path='./data/seg_data/train_ids.txt')
    word_to_ids(vocab, data_paths[1], target_path='./data/seg_data/dev_ids.txt')
    word_to_ids(vocab, data_paths[2], target_path='./data/seg_data/test_ids.txt')

def padding_sentence(sentence, max_len):
    new_sentence = sentence
    if len(sentence) < max_len:
        for i in range(max_len-len(sentence)):
            new_sentence.append(0)
    else:
        new_sentence = sentence[0:max_len]
    return new_sentence


def load_data(path, max_len):
    cat_to_id = read_category()
    labels = []
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label, sentence = line.strip().split('\t')

            labels.append(cat_to_id[label])
            sentence = [int(value) for value in sentence.strip().split(' ')]

            pad_sentence = padding_sentence(sentence, max_len)
            sentences.append(pad_sentence)

    x = np.array(sentences)
    y = np.zeros(shape=(len(labels), 10))
    for i in range(len(labels)):
        label = labels[i]
        y[i][label] = 1

    return x, y


if __name__ == '__main__':

    preprocess(["./data/seg_data/cnews.train.txt", "./data/seg_data/cnews.dev.txt", "./data/seg_data/cnews.test.txt"],
               "./data/vocab.txt",
               "./data/embed/vector_word.txt",
               "./data/embed/wordvecs.npz",
               vocab_size=5000
    )