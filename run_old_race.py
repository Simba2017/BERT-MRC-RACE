import random
import time
import numpy as np
import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn


from torchtext import data
from torchtext import datasets
from torchtext import vocab

from Utils.utils import word_tokenize, get_device, epoch_time, classifiction_metric
from Utils.race_embedding_utils import load_race

def main(config, model_filename):

    device, n_gpu = get_device()

    # 设定随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True  # cudnn 使用确定性算法，保证每次结果一样
    
    # 数据准备
    id_field = data.RawField()
    id_field.is_target = False
    text_field = data.Field(tokenize='spacy', lower=True,
                            include_lengths=True, fix_length=config.sequence_length)
    label_field = data.LabelField(dtype=torch.long)

    train_iterator, dev_iterator, test_iterator = load_race(
        config.data_path, id_field, text_field, label_field, config.batch_size, device, config.glove_word_file)

    # # 词向量
    # word_emb = text_field.vocab.vectors

    # model_file = config.model_dir + model_filename

    # if config.model_name == "GAReader":
    #     pass










if __name__ == "__main__":

    model_name = "GAReader"
    data_dir = "/home/songyingxin/datasets/RACE/all"
    cache_dir = data_dir + "/cache/"
    embedding_folder = "/home/songyingxin/datasets/WordEmbedding/glove/"

    model_dir = ".models/"
    log_dir = ".log/"

    model_filename = "model1.pt"

    if model_name == "GAReader":
        from GAReader import args, GAReader
        main(args.get_args(data_dir, cache_dir, embedding_folder, model_dir, log_dir), model_filename)
