# -*- coding: utf-8 -*-

import argparse


def get_args(data_dir, cache_dir, embedding_folder, model_dir, log_dir):

    parser = argparse.ArgumentParser(description='RACE')

    parser.add_argument("--model_name", default="CoMatch",
                        type=str, help="这批参数所属的模型的名字")

    parser.add_argument("--seed", default=1234, type=int, help="随机种子")

    # data_util
    parser.add_argument(
        "--data_path", default=data_dir, type=str, help="RACE数据集位置")

    parser.add_argument(
        "--cache_dir", default=cache_dir, type=str, help="数据缓存地址"
    )

    parser.add_argument(
        "--sequence_length", default=800, type=int, help="句子长度"
    )

    # 输出文件名
    parser.add_argument(
        "--output_dir", default=model_dir + "GAReader/", type=str, help="输出模型的保存地址"
    )
    parser.add_argument(
        "--log_dir", default=log_dir + "GAReader/", type=str, help="日志文件地址"
    )

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--print_step", default=200,
                        type=int, help="多少步存储一次模型")

    # 模型参数
    parser.add_argument("--output_dim", default=4, type=int)

    # 优化参数
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)

    parser.add_argument("--epoch_num", default=5, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    parser.add_argument("--lr", default=0.001, type=float, help="学习率")

    parser.add_argument("--clip", default=10, type=int, help="梯度裁剪")

    # LSTM 参数
    parser.add_argument("--hidden_size", default=256, type=int, help="隐层特征维度")
    parser.add_argument('--rnn_num_layers', default=1, type=int, help='RNN层数')
    parser.add_argument("--bidirectional", default=True, type=bool)

    # GAReader
    parser.add_argument('--ga_layers', default=1,
                        type=int, help='GAReader 的层数')

    # word Embedding
    parser.add_argument(
        '--glove_word_file',
        default=embedding_folder + 'glove.840B.300d.txt',
        type=str, help='path of word embedding file')
    parser.add_argument(
        '--glove_word_size',
        default=int(2.2e6), type=int,
        help='Corpus size for Glove')
    parser.add_argument(
        '--glove_word_dim',
        default=300, type=int,
        help='word embedding size (default: 300)')

    config = parser.parse_args()

    return config
