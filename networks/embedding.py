import torch.nn as nn
import torch
from torch.nn.utils import weight_norm

class charEmbedding(nn.Module):
    '''
    Input: (max_len, max_word_len) max_len是指句子的最大长度，也就是char的batch size
    Output: (max_len, filter_num)
    '''
    def __init__(self, opt):
        super(charEmbedding, self).__init__()
        self.opt = opt
        self.emb = nn.Embedding(self.opt.char_vocab_size, self.opt.char_embedding_size)
        self.conv = weight_norm(nn.Conv1d(in_channels=self.opt.char_embedding_size, out_channels=self.opt.filter_number,
                                          kernel_size=self.opt.kernel_size))
        self.pool = torch.nn.MaxPool1d(self.opt.max_word_len - self.opt.kernel_size + 1, stride=1)
        self.drop = torch.nn.Dropout(opt.dropout_rate)
        self.init()
    def init(self):
        nn.init.kaiming_uniform_(self.emb.weight.data)
        nn.init.kaiming_uniform_(self.conv.weight.data)
    def forward(self, x):
        '''
            x: one char sequence. shape: (max_len, max_word_len)
        '''
        # 如果输入是一句话
        inp = self.drop(self.emb(x))  # (max_len, max_word_len) -> (max_len, max_word_len, hidden)
        # permute(0,2,1):将原来的tensor按照新的维度生成
        inp = inp.permute(0, 2, 1)  # (max_len, max_word_len, hidden) -> (max_len,  hidden, max_word_len)
        out = self.conv(inp)  # out: (max_len, filter_num, max_word_len - kernel_size + 1)
        return self.pool(out).squeeze()  # out: (max_len, filter_num)

class Embedding(nn.Module):
    def __init__(self, opt):
        super(Embedding, self).__init__()
        self.opt = opt
        # nn.Embedding(48*300):保存了固定字典和大小的简单查找表。输入是一个下标的表，输出是对应词的嵌入。
        self.word_embedding = nn.Embedding(self.opt.vocab_size, self.opt.word_embedding_size)
        self.relu = nn.ReLU()
        if self.opt.use_pos:
            self.pos_embedding = nn.Embedding(self.opt.pos_vocab_size, self.opt.pos_embedding_size, padding_idx=0)
        if self.opt.use_char:
            self.char_encode = charEmbedding(opt)
        # dropout:在训练过程的前向传播中，让每个神经元以一定的概率p处于不激活的状态，已达到减少过拟合的效果
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.word_embedding.weight.data)
        unk = torch.Tensor(1, self.opt.word_embedding_size).uniform_(-1, 1)
        pad = torch.zeros(1, self.opt.word_embedding_size)
        self.word_embedding.weight.data = torch.cat([pad, unk, self.word_embedding.weight.data])
        if self.opt.use_pos:
            nn.init.kaiming_uniform_(self.pos_embedding.weight.data)

    def forward(self, x, pos=None, char=None):
        '''
            x,pos : (batch, max_len)
            char : (batch, max_len, max_word_len)
        '''

        word_emb = self.dropout(self.word_embedding(x))
        if char is not None:
            char_embedding = []
            for i in range(char.shape[0]):
                one_word_char_emb = self.char_encode(char[i])
                char_embedding.append(one_word_char_emb)
            char_embedding = torch.stack(char_embedding)
            word_emb = torch.cat((word_emb, char_embedding), -1)
        if pos is not None:
            pos_emb = self.dropout(self.pos_embedding(pos))
            word_emb = torch.cat((word_emb, pos_emb), -1)
        return word_emb
