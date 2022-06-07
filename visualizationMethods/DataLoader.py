from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import random
from misc.utils import pickle_load

class Data(Dataset):

    def __init__(self, root, prefix):
        self.prefix = prefix
        self.features = np.load(os.path.join(root, prefix+'_features.npy'), allow_pickle=True)
        self.sen_len = np.load(os.path.join(root, prefix+'_len.npy'), allow_pickle=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sen_len = self.sen_len[idx]
        feature = self.features[idx]
        return feature[0], feature[1], feature[2], feature[3], feature[4], sen_len

def dev_test_collate(features):
    text = []
    sent = []
    triples = []
    poses = []
    chars = []
    sen_len = []
    for feature in features:
        text.append(feature[0])
        sent.append(torch.tensor(feature[1]))
        poses.append(torch.tensor(feature[2]))
        chars.append(torch.tensor(feature[3]))
        triples.append(feature[4])
        sen_len.append(feature[5])
    sent = torch.stack(sent)
    poses = torch.stack(poses)
    chars = torch.stack(chars)
    sen_len = torch.tensor(sen_len)
    return text, sent, triples, poses, chars, sen_len

class Loader():
    def __init__(self, opt):
        self.opt = opt

        self.test_data = Data(opt.root_vis, 'test')
        self.test_len = self.test_data.__len__()
        self.loader = {}
        self.reset('test')


    def reset(self, prefix):
        if prefix == 'test':
            self.loader[prefix] = iter(DataLoader(self.test_data, batch_size=1, collate_fn=dev_test_collate,
                                                  shuffle=False))


    def get_batch_dev_test(self, batch_size, prefix):
        wrapped = False
        texts = []
        sents = []
        gts = []
        poses = []
        chars = []
        sen_lens = []
        for i in range(batch_size):
            try:
                text, sent, triple, pos, char, sen_len = self.loader[prefix].next()
            except:
                self.reset(prefix)
                text, sent, triple, pos, char, sen_len = self.loader[prefix].next()
                wrapped = True
            texts.append(text[0])
            sents.append(sent[0])
            gts.append(triple[0])
            poses.append(pos[0])
            chars.append(char[0])
            sen_lens.append(sen_len[0])
        sents = torch.stack(sents)
        poses = torch.stack(poses)
        chars = torch.stack(chars)
        sen_lens = torch.stack(sen_lens)
        return texts, sents, gts, poses, chars, sen_lens, wrapped

if __name__ =='__main__':
    import config
    opt = config.parse_opt()
    data_loader = Loader(opt)
    for i in range(1):
        data = data_loader.get_batch_train(1)
        print("train")
        #data = sorted(data, key=lambda x: list(x[-2].data), reverse=True)
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        print(data[4].shape)
        print(data[5].shape)


    # for i in range(1):
    #     data = data_loader.get_batch_dev_test(10, 'dev')
    #     print("dev")
    #     print(data[1])
    #     print(data[2])
    #     print(data[3].shape)
    #     print(data[4].shape)
    #     print(data[5].shape)
    #     print(data[6])

