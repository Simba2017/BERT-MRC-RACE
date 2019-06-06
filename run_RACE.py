import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob
import json

from tensorboardX import SummaryWriter
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, BertForMultipleChoice

from BertOrigin import args
from Utils.utils import get_device, classifiction_metric
from Utils.race_utils import load_data


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def train(epoch_num, n_gpu, train_dataloader, dev_dataloader, model, optimizer, criterion, gradient_accumulation_steps, device, label_list, output_model_file, output_config_file, log_dir, print_step):

    model.train()

    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime('%H:%M:%S', time.gmtime()))

    best_dev_loss = float('inf')
    global_step = 0

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch+1:02} ----------')
        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits.view(-1, len(label_list)),
                             label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            train_steps += 1
            # 反向传播
            loss.backward()
            
            epoch_loss += loss.item()

            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step != 0 and global_step % print_step == 0:
                train_loss = epoch_loss / train_steps

                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, device, label_list)
                
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

                print_list = ['macro avg', 'weighted avg']
                for label in print_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss

                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    torch.save(model_to_save.state_dict(), output_model_file)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
                
    writer.close()

def evaluate(model, dataloader, criterion, device, label_list):

    model.eval()
    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    for batch in tqdm(dataloader, desc="Eval"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()
    
    acc, report = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report
        

def main(config, bert_vocab_file, bert_model_dir):

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    output_model_file = os.path.join(config.output_dir, config.weights_name)  # 模型输出文件
    output_config_file = os.path.join(config.output_dir, config.config_name)

    # 设备准备
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    """ 设定随机种子 """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
    
    tokenizer = BertTokenizer.from_pretrained(
        bert_vocab_file, do_lower_case=config.do_lower_case)
    label_list = ["0", "1", "2", "3"]
    if config.do_train:
        
        # 数据准备
        train_file = os.path.join(config.data_dir, "train.json")
        dev_file = os.path.join(config.data_dir, "dev.json")

        train_dataloader, train_len = load_data(train_file, tokenizer, config.max_seq_length, config.train_batch_size)

        dev_dataloader, dev_len = load_data(dev_file, tokenizer, config.max_seq_length, config.dev_batch_size)

        num_train_steps = int(
            train_len / config.train_batch_size / config.gradient_accumulation_steps * config.num_train_epochs)
        
        # 模型准备
        model = BertForMultipleChoice.from_pretrained(
            bert_model_dir,
            cache_dir=config.cache_dir, num_choices=4)

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model,device_ids=gpu_ids)

        # 优化器准备
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=config.learning_rate,
                            warmup=config.warmup_proportion,
                             t_total=num_train_steps)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        train(config.num_train_epochs, n_gpu, train_dataloader, dev_dataloader, model, optimizer, criterion,
              config.gradient_accumulation_steps, device, label_list, output_model_file, output_config_file, config.log_dir, config.print_step)

    test_file = os.path.join(config.data_dir, "test.json")
    test_dataloader, _ = load_data(
        test_file, tokenizer, config.max_seq_length, config.test_batch_size)
    
    bert_config = BertConfig(output_config_file)
    model = BertForMultipleChoice(bert_config, num_choices=len(label_list))
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    """ 损失函数准备 """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    test_loss, test_acc, test_report = evaluate(
        model, test_dataloader, criterion, device, label_list)

    print("-------------- Test -------------")
    print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} %')

    for label in label_list:
        print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
            label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
    print_list = ['macro avg', 'weighted avg']

    for label in print_list:
        print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
            label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
    


if __name__ == "__main__":
    data_dir = "/search/hadoop02/suanfa/songyingxin/data/RACE/all"
    output_dir = ".bertoutput"
    cache_dir = ".bertcache"
    log_dir = ".bertlog"

    # bert-base
    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased"

    # bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased-vocab.txt"
    # bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased"

    main(args.get_args(data_dir, output_dir, cache_dir, log_dir), bert_vocab_file, bert_model_dir)
