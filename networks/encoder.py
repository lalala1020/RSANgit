import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        if self.opt.use_pos and self.opt.use_char:
            self.in_width = self.opt.word_embedding_size + self.opt.pos_embedding_size + self.opt.filter_number
        elif self.opt.use_pos:
            self.in_width = self.opt.word_embedding_size + self.opt.pos_embedding_size
        elif self.opt.use_char:
            self.in_width = self.opt.word_embedding_size + self.opt.filter_number
        else:
            self.in_width = self.opt.word_embedding_size
        self.birnn = nn.GRU(self.in_width, self.opt.rnn_hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.opt.rnn_hidden_size*2, self.opt.att_hidden_size)
        self.dropout = nn.Dropout(self.opt.dropout_rate)

    def forward(self, x, sen_len):
        # sort:排序。un_idx是元素索引，descending表示倒叙
        sort_len, perm_idx = sen_len.sort(0, descending=True)
        # torch.sort:按照指定维度对tensor张量的元素进行排序,
        _, un_idx = torch.sort(perm_idx, dim=0)
        x_input = x[perm_idx]
        # pack_padded_sequence的输入的序列长度必须是从长到短排序好的，压缩掉无效的填充值
        packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)

        packed_out, _ = self.birnn(packed_input)
        # pad_packed_sequence：把压缩的序列在填充回来，便于后续处理
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.opt.max_len)
        # index_select：按照参数un_idx中的值作为索引输出out_put的值
        output = torch.index_select(output, 0, un_idx)
        # relu():激活函数（可作为独立的一层），输出与输入值的size相同，保留正值，负值用0代替
        output = torch.relu(self.linear(output))

        return output
