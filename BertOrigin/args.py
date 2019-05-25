import argparse


def get_args(data_dir, output_dir, cache_dir, log_dir):

    parser = argparse.ArgumentParser(description='BERT Baseline')

    # 文件路径：数据目录， 缓存目录
    parser.add_argument("--data_dir",
                        default=data_dir,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default=output_dir,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--config_name", 
                        default="bert_config.json",
                        type=str)
    parser.add_argument("--weights_name",
                        default="pytorch_model.bin",
                        type=str)

    parser.add_argument("--cache_dir",
                        default=cache_dir,
                        type=str,
                        help="缓存目录，主要用于模型缓存")

    parser.add_argument("--log_dir",
                        default=log_dir,
                        type=str,
                        help="缓存目录，主要用于模型缓存")

    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="随机种子 for initialization")

    # 文本预处理参数
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--max_seq_length",
                        default=400,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # 训练参数
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--test_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for test.")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    # optimizer 参数
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam 的 学习率"
                        )
    
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--print_step',
                        type=int,
                        default=400,
                        help="多少步进行模型保存以及日志信息写入")


    # 解决gpu不足问题
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
                         
    config = parser.parse_args()

    return config
