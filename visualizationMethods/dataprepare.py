#encoding=utf-8
import config
import json
import os
import numpy as np
import six
from six.moves import cPickle

import jieba
import jieba.posseg as psg

def pickle_load(f):
    # six.PY3：即python版本是3.X
    # 打开kpl文件
    if six.PY3:
        # cPickle可以对任意类型的python对象进行序列化操作
        # cPickle.load（）：从字符串变量中载入python对象两个参数
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    # 生成pkl文件
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

# 将json文件存放在数组中
def read_json(filename):
    data = []
    with open('' + filename, 'rb+') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


opt = config.parse_opt()
dic = "data/百度/dict.txt"
jieba.load_userdict(dic)

vocab = np.load(opt.input_vocab)
# vocab是word.npy文件
word2id = {j: i for i, j in enumerate(vocab)}
id2word = {i: j for i, j in enumerate(vocab)}
# rel2id是rel2id.json文件
rel2id = json.load(open(opt.input_rel2id, 'r'))
# label2id是label2id.json文件
label2id = json.load(open(opt.input_label2id, 'r'))
# pos2id是pos2id.json文件
pos2id = json.load(open(opt.input_pos2id, 'r'))
# char2id是char2id.json文件
char2id = json.load(open(opt.input_char2id, 'r'))
# train_data是train.json,test_data是test.json,dev_data是dev.json
test_data = read_json(opt.input_test_vis)


def prepare():
    # 将test_f, test_l的内容保存到npy文件中
    test_f, test_l = process_dev_test(test_data)
    np.save(os.path.join('' + opt.root_vis, 'test_features.npy'), test_f, allow_pickle=True)
    np.save(os.path.join('' + opt.root_vis, 'test_len.npy'), test_l, allow_pickle=True)


def process_dev_test(dataset):
    features = []
    sen_len = []
    for i, data in enumerate(dataset):
        sent_text = data['sentText']
        sent_words, sent_ids, pos_ids, sent_chars, cur_len = process_sentence(sent_text)
        entities = data['entityMentions']
        raw_triples_ = data['relationMentions']
        # 去重
        triples_list = []
        for t in raw_triples_:
            triples_list.append((t['em1Text'], t['em2Text'], t['label']))
        triples_ = list(set(triples_list))
        triples_.sort(key=triples_list.index)

        triples = []
        for triple in triples_:
            head, tail, relation = triple
            try:
                if triple[2] != 'None':

                    head_index = sent_text.index(head)
                    head_pos = range(head_index, (head_index + len(head)))

                    tail_index = sent_text.index(tail)
                    tail_pos = range(tail_index, (tail_index + len(tail)))

                    h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                    t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                    print((relation, rel2id[relation]))
                    triples.append((h_chunk, t_chunk, rel2id[relation]))
            except:
                continue
        features.append([sent_text, sent_ids, pos_ids, sent_chars, triples])
        sen_len.append(cur_len)
        if (i + 1) * 1.0 % 10000 == 0:
            print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))

    return np.array(features), np.array(sen_len)

def process_sentence(sent_text):

    sen_len = min(len(sent_text), opt.max_len)
    sent_text = sent_text[:sen_len]
    sent_pos_ = psg.lcut(sent_text)
    sent_words_ = list(sent_text)   # 分字，包括标点符号
    sent_pos = []
    for i in sent_pos_:
        str(i).replace('/', ',')
        i = tuple(i)
        sent_pos.append(i)  # 分词 带词性
    sent_words = []
    for j in sent_pos:
        sent_words.append(j[0])  # 分词

    sent_pos_ids = []
    for pos in sent_pos:
        p = pos2id.get(pos[1], 1)
        p_l = len(pos[0])
        for a in range(0, p_l):
            sent_pos_ids.append(p)
    sent_ids = []
    for w in sent_words:
        ww = word2id.get(w, 1)
        w_l = len(w)
        for a in range(0, w_l):
            sent_ids.append(ww)
    sent_chars = []
    for w in sent_words:
        tokens = [char2id.get(token, 1) for token in list(w)]
        word_len = min(len(w), opt.max_word_len)
        for _ in range(opt.max_word_len - word_len):
            tokens.append(0)
        for o in range(0, len(w)):
            sent_chars.append(tokens[: opt.max_word_len])

    for _ in range(sen_len, opt.max_len):
        sent_ids.append(0)
        sent_pos_ids.append(0)
        sent_chars.append([0] * opt.max_word_len)

    if len(sent_ids) != 120:
        print(sent_text)
        print(len(sent_ids))
    if len(sent_pos_ids) != 120:
        print(sent_text)
        print(len(sent_pos_ids))
    if len(sent_chars) != 120:
        print(sent_text)
        print(len(sent_chars))

    return sent_words_[:sen_len], sent_ids, sent_pos_ids, sent_chars, sen_len
