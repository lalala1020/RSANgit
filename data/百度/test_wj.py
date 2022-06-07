import json
import numpy
vocab = numpy.load("word.npy")
print(len(vocab))
# file_name = "char2id.json"
# with open(file_name, encoding='utf-8') as f:
#     vocab = json.load(f)
#     for one in vocab:
#         print(one)
    # print(vocab)




# import torch
# sents = torch.tensor([[8, 8, 8, 5, 6, 8, 8, 8, 8, 8, 2, 3, 9, 9],
#                       [2, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 6, 9],
#                       [8, 8, 2, 3, 8, 8, 8, 8, 8, 5, 6, 9, 9, 9]])
# import pickle

# f = open('dict_word.pkl', 'rb')111111111111111111
# for line in f:
#     print(line)
# word = pickle.load(open("E:\PycharmProjects\模型与对比模型\RSAN\save\infos-best.pkl", 'rb'), encoding='utf-8')
# # train = pickle.load(open("train.pkl", 'rb'), encoding='iso-8859-1')
# print(word)

# dic = []
#
# f1 = open("origin/train.json", "r", encoding="utf-8")
# for line in f1.readlines():
#     line = json.loads(line)
#     entities = line["entityMentions"]
#     for e in entities:
#         if e not in dic:
#             dic.append(e)
#
# f2 = open("origin/dev.json", "r", encoding="utf-8")
# for line in f2.readlines():
#     line = json.loads(line)
#     entities = line["entityMentions"]
#     for e in entities:
#         if e not in dic:
#             dic.append(e)
#
# f3 = open("origin/test.json", "r", encoding="utf-8")
# for line in f3.readlines():
#     line = json.loads(line)
#     entities = line["entityMentions"]
#     for e in entities:
#         if e not in dic:
#             dic.append(e)
#
#
# f4 = open("dict.txt", "w", encoding="utf-8")
# for d in dic:
#     f4.write(str(d) + "\n")


