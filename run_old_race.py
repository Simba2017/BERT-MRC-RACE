# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import argparse
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.nn as nn


from torchtext import data
from torchtext import datasets
from torchtext import vocab

from Utils.utils import word_tokenize, get_device, epoch_time, classifiction_metric
from Utils.race_embedding_utils import load_race


def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, label_list, out_model_file, log_dir, print_step):

    model.train()
    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime('%H:%M:%S', time.gmtime()))
    
    global_step = 0
    best_dev_loss = float('inf')

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch+1:02} ----------')

        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            optimizer.zero_grad()

            logits = model(batch)

            loss = criterion(logits.view(-1, len(label_list)), batch.label)

            labels = batch.label.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            loss.backward()
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            train_steps += 1

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

            if global_step % print_step == 0:

                train_loss = epoch_loss / train_steps
                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, label_list, data_type)
                c = global_step // print_step

                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/dev", dev_loss, c)

                writer.add_scalar("acc/train", train_acc, c)
                writer.add_scalar("acc/dev", dev_acc, c)

                for label in label_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                print_list = ['micro avg', 'macro avg', 'weighted avg']
                for label in print_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(), out_model_file)

                model.train()

    writer.close()



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
                            include_lengths=True)
    label_field = data.LabelField(dtype=torch.long)

    train_iterator, dev_iterator, test_iterator = load_race(
        config.data_path, id_field, text_field, label_field, config.batch_size, device, config.glove_word_file, config.cache_path)

    for batch in dev_iterator:
        print("test")
    # 词向量
    word_emb = text_field.vocab.vectors

    model_file = config.model_dir + model_filename

    if config.model_name == "GAReader":
        from GAReader.GAReader import GAReader
        model = GAReader(
            config.glove_word_dim, config.output_dim, config.hidden_size,
            config.rnn_num_layers, config.ga_layers, config.bidirectional,
            config.dropout, word_emb)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if config.do_train:
        train(config.epoch_num, model, train_iterator, dev_iterator, optimizer, criterion, ['0', '1', '2', '3'], model_file, config.log_dir, config.print_step)










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
