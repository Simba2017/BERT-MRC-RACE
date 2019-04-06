from tqdm import tqdm
import spacy
import json
import os
from collections import Counter
import argparse
import numpy as np
import matplotlib.pyplot as plt

nlp = spacy.blank("en")

dataset = "/home/songyingxin/datasets/"
target_dir = "./data"

parser = argparse.ArgumentParser(description='RACE data')

""" Dataset """
parser.add_argument(
    '--RACE',
    default=dataset + "RACE", type=str,
    help='path of the RACE dataset')



""" word Embedding """
parser.add_argument(
    '--glove_word_file',
    default=dataset + 'glove/glove.840B.300d.txt',
    type=str, help='path of glove word embedding file')
parser.add_argument(
    '--glove_word_size',
    default=int(2.2e6), type=int,
    help='Corpus size for Glove')
parser.add_argument(
    '--glove_dim',
    default=300, type=int,
    help='word embedding size (default: 300)')

""" char Embedding """
parser.add_argument(
    '--glove_char_file',
    default=dataset + "glove/glove.840B.300d-char.txt",
    type=str, help='path of char embedding file')

parser.add_argument(
    '--glove_char_size',
    default=94, type=int,
    help='Corpus size for char embedding')

parser.add_argument(
    '--char_dim',
    default=64, type=int,
    help='char embedding size (default: 64)')

""" nums of limit """
parser.add_argument(
    '--article_limit',
    default=1500, type=int,
    help='maximum context token number')

parser.add_argument(
    '--ques_limit',
    default=80, type=int,
    help='maximum question token number')

parser.add_argument(
    '--option_limit',
    default=120, type=int,
    help='maximum option token number')

parser.add_argument(
    '--char_limit',
    default=16, type=int,
    help='maximum char number in a word')


def word_tokenize(sent):
    """ 分词 """
    doc = nlp(sent)
    return [token.text for token in doc]

def get_examples(dir_name, name, type, word_counter, char_counter):
    """ 
    将所有数据转化为json文件, 并将问题划分成多个样本
    Args:
        dir_name: 文件路径
        name: middle , high
        type: train, dev, test
        word_counter, char_counter: 计频
    
    Returns:
        examples: 分词后的样本信息, 列表类型：
            [
                {"article_tokens": ... , "article_chars": ... , 
                "question_tokens": ... , "question_chars": ... , 
                "option0_tokens": ... , "option0_chars": ...,
                "option1_tokens": option_tokens[1], "option1_chars": option_chars[1], 
                "option2_tokens": option_tokens[2], "option2_chars": option_chars[2], 
                "option3_tokens": option_tokens[3], "option3_chars": option_chars[3],
                "answer": ... , "id": ...} ... ]
        eval_example: 字典类型， eval_example[i] = {"article": ..., "question": ..., "answer": ..., "uuid": ...}
        meta: 数据集信息， 字典类型
            {
                "max_ques_len": 问题最大长度, 
                "max_article_len": 文章最大长度,
                "max_option_len": 选项最大长度,
                'total': 样本数
            }
    """
    answer_dict = {"A": 0, "B": 1, "C": 2, "D": 3}
    print("Generating {}:{} examples...".format(name, type))

    examples = []
    eval_examples = {}
    meta = {"ques_len":[], "article_len": [], "option_len": []}
    total = 0

    dir_name = dir_name + os.sep + type + os.sep + name

    for file in os.listdir(dir_name):
        data = json.load(open(os.path.join(dir_name, file)))

        article = data['article'].replace("\\newline", "\n")
        article = article.lower().strip()
        article = article.replace("''", '" ').replace("``", '" ')
        article_tokens = word_tokenize(article)  # 分词
        article_chars = [list(token) for token in article_tokens]  #　分 char

        meta['article_len'].append(len(article_tokens))

        """ 统计 article 词频， char 频 """
        for token in article_tokens:
            word_counter[token] += len(data['questions'])
            for char in token:
                char_counter[char] += len(data['questions'])
        
        for i in range(len(data['questions'])):
            total += 1

            question = data['questions'][i]
            question = question.lower().strip()
            question = question.replace("''", '" ').replace("``", '" ')
            question_tokens = word_tokenize(question)
            question_chars = [list(token) for token in question_tokens]
            
            meta['ques_len'].append(len(question_tokens))
            """ 统计 question 的词频， char 频 """
            for token in question_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1
            
            answer = answer_dict[data['answers'][i]]

            options_tokens = []
            options_chars = []
            
            for option in data['options'][i]:
                option = option.lower().strip()
                option = option.replace("''", '" ').replace("``", '" ')
                option_tokens = word_tokenize(option)
                option_chars = [list(token) for token in option_tokens]
                options_tokens.append(option_tokens)
                options_chars.append(option_chars)

                meta['option_len'].append(len(option_tokens))
                """ 统计 option 的词频， char 频 """
                for token in option_tokens:
                    word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1

            uuid = data['id'].split('.')[0] + '_' + str(i)

            example ={
                "article_tokens":article_tokens, "article_chars":article_chars, 
                "question_tokens":question_tokens, "question_chars":question_chars, 
                "option0_tokens": options_tokens[0], "option0_chars": options_chars[0],
                "option1_tokens": options_tokens[1], "option1_chars": options_chars[1], 
                "option2_tokens": options_tokens[2], "option2_chars": options_chars[2], 
                "option3_tokens": options_tokens[3], "option3_chars": options_chars[3],
                "answer":answer,  "id": str(total)}
            
            examples.append(example)
            eval_examples[str(total)] = {"article": article, "question": question, "answer": answer, "uuid": uuid}
    print("{} examples in this dataset".format(len(examples)))
    meta['total'] = total
    return examples, eval_examples, meta

def integrate_examples(high_examples, middle_examples):
    """ 将 high， middle 两大数据集整合 """
    result = high_examples.copy()
    result.extend(middle_examples)
    return result

def integrate_meta(high_meta, middle_meta):
    """ 整个 high， middle 的 meta 信息 """
    result = high_meta.copy()
    infomations = ["ques_len", "article_len", "option_len"]
    for info in infomations:
        result[info].extend(middle_meta[info])
    result['total'] = result['total'] + middle_meta['total']
    return result


def get_embedding(counter, data_type, emb_file=None, size=None, vec_size=None,limit=-1,):
    """ 建立 word: id 的映射，并建立 embedding 矩阵
    Args:
        counter: word_counter 或 char_counter
        data_type: word 或 char
        limit： 最小的句子长度
        emb_file: 词向量文件 或 char向量文件
        vec_size : 词向量或char向量维度
    Returns:
        emb_mat: 词/char向量矩阵
        token2idx_dict: word:id
    """
    print("Generating {} embedding ...".format(data_type))

    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]

    with open(emb_file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=size):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in counter and counter[word] > limit:
                embedding_dict[word] = vector
    print("{} / {} tokens have corresponding {} embedding vector".format(
        len(embedding_dict), len(filtered_elements), data_type))
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def filter_func(config, example):
    """ 过滤样本集 """
    return (len(example["article_tokens"]) > config.article_limit or
            len(example["question_tokens"]) > config.ques_limit or
            max([len(option) for option in example['options_tokens']]) > config.option_limit)

def build_features(config, examples , meta, data_type, word2idx_dict, char2idx_dict, debug=False):
    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0

    for examples in tqdm(examples):
        total_ += 1

        if filter_func(config, example):
            continue
        total += 1

        article_word_ids = np.ones([config.article_limit], data_type=np.int32)
        article_char_ids = np.ones([config.article_limit, config.char_limit], dtype=np.int32)

        question_word_ids = np.ones([config.ques_limit], dtype=int32)
        question_char_ids = np.ones([config.ques_limit, config.char_limit], dtype=int32)

        option1_word_ids = np.ones([config.option_limit], dtype=int32)



def pickle_dump_large_file(obj, filepath):
    """
    This is a defensive way to write pickle.write,
    allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def save(filepath, obj, message=None):
    """ 保存信息到文件中 """
    if message is not None:
        print("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)

def analyze_meta(meta, title):
    """ 对 meta 进行图形绘制，分析，以便确定长度参数 """
    def draw_line(data, item, title):
        x = range(len(data))
        plt.plot(x, data)
        plt.ylabel(item)
        plt.title(title)
        plt.legend()
        plt.show()

    items = ["ques_len", "article_len", "option_len"]
    for item in items:
        data = meta[item]
        draw_line(data, item, title)





def prepro(config):
    """ 进行数据预处理 """

    """ 参数 
    home: RACE 数据集的家目录
    word_emb_file: word embedding file
    word_emb_size: 词向量
    """

    word_counter, char_counter = Counter(), Counter()

    """ 数据集导入 """
    high_train_examples, high_train_eval, high_train_meta = get_examples(config.RACE, 'high', 'train', word_counter, char_counter)
    analyze_meta(high_train_meta, "high_train")

    high_dev_examples, high_dev_eval, high_dev_meta = get_examples(config.RACE, 'high', 'dev', word_counter, char_counter)
    analyze_meta(high_dev_meta, "high_dev")

    high_test_examples, high_test_eval, high_test_meta = get_examples(config.RACE, 'high', 'test', word_counter, char_counter)
    analyze_meta(high_test_meta, "high_test")

    middle_train_examples, middle_train_eval, middle_train_meta = get_examples(config.RACE, 'middle', 'train', word_counter, char_counter)
    analyze_meta(middle_train_meta, "middle_train")

    middle_dev_examples, middle_dev_eval, middle_dev_meta = get_examples(config.RACE, 'middle', 'dev', word_counter, char_counter)
    analyze_meta(middle_dev_meta, "middle_dev")

    middle_test_examples, middle_test_eval, middle_test_meta = get_examples(config.RACE, 'middle', 'test', word_counter, char_counter)
    analyze_meta(middle_test_meta, "middle_test")

    """ 整合数据集 """
    train_examples = integrate_examples(high_train_examples, middle_train_examples)
    dev_examples = integrate_examples(high_dev_examples, middle_dev_examples)
    test_examples = integrate_examples(high_test_examples, middle_test_examples)

    train_meta = integrate_meta(high_train_meta, middle_train_meta)
    analyze_meta(train_meta, "train")
    dev_meta = integrate_meta(high_dev_meta, middle_dev_meta)
    analyze_meta(dev_meta, "dev")
    test_meta = integrate_meta(high_test_meta, middle_test_meta)
    analyze_meta(test_meta, 'test')

    # word_emb_file = config.glove_word_file
    # word_emb_size = config.glove_word_size
    # word_emb_dim = config.glove_dim

    # char_emb_file = config.glove_char_file
    # char_emb_size = config.glove_char_size
    # char_emb_dim = config.glove_dim

    # word_emb_mat, word2id_dict = get_embedding(
    #     word_counter, "word", emb_file=word_emb_file, size=word_emb_size, vec_size=word_emb_dim)
    
    # char_emb_mat, char2id_dict = get_embedding(
    #     char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)


    




if __name__ == "__main__":
    prepro(parser.parse_args())
