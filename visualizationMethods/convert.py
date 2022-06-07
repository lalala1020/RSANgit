import config
import os
import json
import model
from visualizationMethods import dataprepare, eval, dataprocess
from visualizationMethods.DataLoader import Loader

def convert(opt):
    dataprepare.prepare()
    loader = Loader(opt)
    Model = model.setup(opt).cuda()
    label2id = json.load(open(opt.input_label2id, "r"))
    predictions = eval.evaluate(Model, loader, label2id, opt.eval_batch_size, opt.rel_num, 'test')
    out_file = open("save_files/tmp/origin_data.txt", "w", encoding="utf-8")
    for p in predictions:
        out_file.write(str(p) + "\n")
    out_file.close()
    dataprocess.process()
    return True
# convert(opt)s