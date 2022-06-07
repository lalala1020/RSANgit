import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_vocab', type=str, default='data/百度/word.npy')
    parser.add_argument('--input_rel2id', type=str, default='data/百度/rel2id.json')
    parser.add_argument('--input_label2id', type=str, default='data/百度/label2id.json')
    parser.add_argument('--input_pos2id', type=str, default='data/百度/pos2id.json')
    parser.add_argument('--input_char2id', type=str, default='data/百度/char2id.json')
    parser.add_argument('--input_train', type=str, default='data/百度/origin/train.json')
    parser.add_argument('--input_test', type=str, default='data/百度/origin/test.json')
    parser.add_argument('--input_test_vis', type=str, default='save_files/origin/data.json')
    parser.add_argument('--input_dev', type=str, default='data/百度/origin/dev.json')

    parser.add_argument('--model', type=str, default='Rel_based_labeling', help='model name')
    parser.add_argument('--root', type=str, default='data/百度', help='path to the dataset')
    parser.add_argument('--root_vis', type=str, default='save_files/tmp', help='path to the dataset')
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids')

    parser.add_argument('--use_pos', action='store_true', default=True, help='whether to use pos embedding')
    parser.add_argument('--pos_embedding_size', type=int, default=30, help='the dimension of pos')
    parser.add_argument('--pos_vocab_size', type=int, default=48)
    parser.add_argument('--use_char', action='store_true', default=True, help='whether to use character embedding')
    parser.add_argument('--max_word_len', type=int, default=10, help='the max length of word')
    parser.add_argument('--char_embedding_size', type=int, default=30, help='the dimension of char emb')
    parser.add_argument('--kernel_size', type=int, default=3, help='the kernel size of character CNN')
    parser.add_argument('--filter_number', type=int, default=50, help='filter number of character CNN')
    parser.add_argument('--char_vocab_size', type=int, default=5959)  # 66

    parser.add_argument('--max_len', type=int, default=120, help='the max length of input text')
    parser.add_argument('--word_embedding_size', type=int, default=300, help='the dimension of input word')
    parser.add_argument('--vocab_size', type=int, default=131591)  # 83997
    parser.add_argument('--rnn_hidden_size', type=int, default=300, help='dimension of rnn in encoder and decoder')
    parser.add_argument('--att_hidden_size', type=int, default=300, help='dimension of the attention space')
    parser.add_argument('--max_epoch', type=int, default=200, help='-1: never stop, or >0')  # 100
    parser.add_argument('--rel_dim', type=int, default=300, help='dimension of the relation')
    parser.add_argument('--rel_num', type=int, default=25, help='number of relation types')  # 25
    parser.add_argument('--label_num', type=int, default=10, help='number of target labels')
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 0.001
    parser.add_argument('--learning_rate_decay', type=float, default=0.5)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0, help='-1: not decay or >0')
    parser.add_argument('--learning_rate_decay_every', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--grad_clip', type=float, default=10., help='the maximum of gradient values')
    parser.add_argument('--neg_rate', type=float, default=0.2, help='the number of negative sampling')
    parser.add_argument('--neg_num', type=int, default=4, help='the number of negative sampling')
    parser.add_argument('--op_type', type=str, default='add', help='add or mul')
    parser.add_argument('--attention_type', type=str, default='dot', help='dot or net')

    parser.add_argument('--dump_results', type=int, default=1, help='dump predictions into prediction path')
    parser.add_argument('--dump_path', type=str, default='predictions', help='the path to save test results')
    parser.add_argument('--save_loss_every', type=int, default=140, help='how often (iterations) to save  loss history')
    parser.add_argument('--save_checkpoint_every', type=int, default=562, help='how often (iterations) to save model')
    parser.add_argument('--checkpoint_path', type=str, default='save', help='path to save model')
    parser.add_argument('--load_from', type=str, default="save", help='path to load the model parameters, not including optimizer')  # None
    parser.add_argument('--start_from', type=str, default=None, help='path to start from the last checkpoint, including weights, optimizer, performance')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)  # 16
    parser.add_argument('--eval_batch_size', type=int, default=50)  # 50
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer selected in training')

    # args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    return args
