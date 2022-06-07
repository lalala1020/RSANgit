import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionDot(nn.Module):
    def __init__(self, opt):
        super(AttentionDot, self).__init__()
        self.opt = opt
        self.rel2att = nn.Linear(self.opt.rel_dim, self.opt.att_hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.rel2att.weight.data)

    def forward(self, sent, rel, mask):
        # sent: batch, max_len, hidden
        # rel: batch, rel_dim -> relation: batch, hidden
        relation = self.rel2att(rel).unsqueeze(1)
        # batch, max_len
        weight = torch.matmul(relation, sent.transpose(-1, -2)).squeeze()
        weight = weight * mask.float()
        weight = torch.softmax(weight, -1)
        att_res = torch.bmm(weight.unsqueeze(1), sent).squeeze(1)  # batch_size * att_hidden_size
        return att_res, weight

class AttentionNet(nn.Module):
    def __init__(self, opt):
        super(AttentionNet, self).__init__()
        self.opt = opt
        self.Wg = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.Wh = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.Wr = nn.Linear(self.opt.rel_dim, self.opt.att_hidden_size)
        self.alpha_net = nn.Linear(self.opt.att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wg.weight.data)
        nn.init.xavier_uniform_(self.Wh.weight.data)
        nn.init.xavier_uniform_(self.Wr.weight.data)
        nn.init.xavier_uniform_(self.alpha_net.weight.data)

    # pool为计算的global_sen
    def forward(self, sent_h, rel, pool, mask):
        relation = self.Wr(rel)
        sent = self.Wh(sent_h)
        global_sen = self.Wg(pool)
        # 将relation扩展为sent大小一致
        relation = relation.unsqueeze(1).expand_as(sent)
        global_sen = global_sen.unsqueeze(1).expand_as(sent)
        # tanh：用输入元素的双曲切线返回一个新的tensor
        mix = torch.tanh(relation + sent + global_sen)
        weight = self.alpha_net(mix).squeeze()
        # 将值为0的地方替换为-1e9
        weight.masked_fill_(mask == 0, -1e9)
        weight_ = torch.softmax(weight, -1)

        #weight = weight * mask.float()
        # 将sent中的每个句子的字分配权重
        att_res = torch.bmm(weight_.unsqueeze(1), sent).squeeze(1)  # batch_size * att_hidden_size
        return att_res, weight_

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.relation_matrix = nn.Embedding(self.opt.rel_num, self.opt.rel_dim)

        self.attention = AttentionNet(opt)
        self.W = nn.Linear(self.opt.att_hidden_size*3, self.opt.att_hidden_size)
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        self.bilstm = nn.LSTM(self.opt.att_hidden_size, self.opt.rnn_hidden_size,
                              bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.opt.rnn_hidden_size*2, self.opt.label_num)

        self.W1 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.W2 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.W3 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size*2)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.xavier_uniform_(self.hidden2tag.weight.data)
        #nn.init.kaiming_uniform_(self.W.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.W.weight.data)
        nn.init.xavier_uniform_(self.W1.weight.data)
        nn.init.xavier_uniform_(self.W2.weight.data)
        nn.init.xavier_uniform_(self.W3.weight.data)

    def masked_mean(self, sent, mask):
        # masked_fill(mask，value):用于attention机制中，用来mask掉当前时刻前后时刻的序列信息
        # 参数mask是一个tensor中，元素是布尔值，value是要填充的值
        # mask.masked_fill(mask == 0, -1e9)：即将mask矩阵中值为0的标量全部替换为-1e9
        mask_ = mask.masked_fill(mask == 0, -1e9)
        # torch.softmax(mask_, -1)：给每横中的元素分配权重（计算句子中每个字所分配的权重），其和=1
        score = torch.softmax(mask_, -1)
        # matmul：两个tensor矩阵相乘
        # unsqueeze()：返回一个新的tensor，其size为插入指定位置的size，若原tensor的size为（3,4）
        # unsqueeze(1)返回的tensor的size为（3,1,4）
        # squeeze(1)：即压缩维度1，若该维度大小为1，则移除
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def nn_decode(self, inputs, sen_len):
        sort_len, perm_idx = sen_len.sort(0, descending=True)
        _, un_idx = torch.sort(perm_idx, dim=0)
        x_input = inputs[perm_idx]
        packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)

        packed_out, _ = self.bilstm(packed_input)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.opt.max_len)
        output = torch.index_select(output, 0, un_idx)
        return output

    def forward(self, sent, rel, mask, sen_len):

        rel_embedding = self.relation_matrix(rel)
        # mask矩阵是sent大小的tensor，与句子重合处值为1，其余位置为0
        global_sen = self.masked_mean(sent, mask)

        sent_att, weight = self.attention(sent, rel_embedding, global_sen, mask)
        # 将每个句子全部表示global_sen和关系特征表示sent_att连接在一起
        concats = torch.cat([self.W1(global_sen), self.W2(sent_att)], -1)
        alpha = torch.sigmoid(concats)
        gate = alpha * torch.tanh(self.W3(sent_att))

        decode_input = torch.cat([sent, gate.unsqueeze(1).expand(sent.shape[0], sent.shape[1], -1)], -1)
        decode_input = self.W(decode_input)
        #decode_input = sent + (alpha * (sum_sen_rel)).unsqueeze(1).expand_as(sent)

        decode_out = self.nn_decode(decode_input, sen_len)
        project = self.hidden2tag(decode_out)
        return project, weight
