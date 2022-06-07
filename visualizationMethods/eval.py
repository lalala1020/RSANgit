import torch
import numpy as np
from misc.utils import tag_mapping,attn_mapping
import sys
import config
import json

opt = config.parse_opt()
rel2id = json.load(open(opt.input_rel2id, 'r'))

def trans_index_to_entity(text, triples):
    f_triples = list()
    for triple in triples:
        h, t, r = triple
        _, h1, h2 = h
        head = text[h1:h2]
        _, t1, t2 = t
        tail = text[t1:t2]
        rs = "NONE"
        for rel in rel2id.items():
            rel1, rel2 = rel
            if rel2 == r:
                rs = rel1
                break
        f_triples.append((head, tail, rs))
    return f_triples

def evaluate(model, loader, label2id, batch_size, rel_num, prefix):
    model.eval()
    loader.reset(prefix)
    n = 0
    predictions = []
    if prefix == 'dev':
        val_num = loader.dev_len
    else:
        val_num = loader.test_len
    # while True:
    #     with torch.no_grad():
    texts, sents, gts, poses, chars, sen_lens, wrapped = loader.get_batch_dev_test(batch_size, prefix)
    sents = sents.cuda()
    sen_lens = sen_lens.cuda()
    mask = torch.zeros(sents.size()).cuda()
    poses = poses.cuda()
    chars = chars.cuda()
    n = n + batch_size
    for i in range(sents.size(0)):
        mask[i][:sen_lens[i]] = 1
    sents = sents.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
    poses = poses.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
    chars = chars.repeat([1, rel_num - 1, 1]).view(batch_size * (rel_num - 1), opt.max_len, -1)
    mask = mask.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
    sen_lens = sen_lens.unsqueeze(1).repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1))
    rel = torch.arange(1, rel_num).repeat(batch_size).cuda()
    if not opt.use_char:
        chars = None
    if not opt.use_pos:
        poses = None
    predict, attention_score = model(sents, sen_lens, rel, mask, poses, chars)   # (batch * rel_num-1) * max_sen_len * label_num
    predict = torch.softmax(predict, -1)


    for i in range(predict.size(0)):
        predict[i][:sen_lens[i], -1] = -1e9
        predict[i][sen_lens[i]:, -1] = 1e9
    decode_tags = np.array(predict.max(-1)[1].data.cpu())
    current_relation = [k for k in range(1, rel_num)]
    # print(decode_tags)

    for i in range(batch_size):
        triple = tag_mapping(decode_tags[i * (rel_num - 1):(i + 1) * (rel_num - 1)], current_relation, label2id)
        #att = attn_mapping(attention_score[i * (rel_num - 1):(i + 1) * (rel_num - 1)], gts[i])
        target = gts[i]
        text = texts[i]
        f_triple = trans_index_to_entity(text, triple)
        if f_triple not in predictions:
            predictions.append(f_triple)

    # model.train()

    return predictions
