import json
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)


class RaceExample(object):
    """ RACE 数据集的样本格式
    Args:
        race_id: data id
        article: article
        question: question
        option_0/1/2/3: option_0/1/2/3
        label: true answer
    """

    def __init__(self,
                 race_id,
                 article,
                 question,
                 option_0,
                 option_1,
                 option_2,
                 option_3,
                 label=None):
        self.race_id = race_id
        self.article = article
        self.question = question
        self.option_0 = option_0
        self.option_1 = option_1
        self.option_2 = option_2
        self.option_3 = option_3
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.article}",
            f"question: {self.question}",
            f"option_0: {self.option_0}",
            f"option_1: {self.option_1}",
            f"option_2: {self.option_2}",
            f"option_3: {self.option_3}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    """ RACE 数据集输入特征 
    Args:
        example_id: 样本id
        choice_features: 特征 ['input_ids', 'input_mask', 'segment_ids']
        label: 标签
    """

    def __init__(self,
                 example_id,
                 article_features,
                 question_features,
                 option0_features,
                 option1_features,
                 option2_features,
                 option3_features,
                 label
                 ):
        self.example_id = example_id
        self.article_features = {
            'input_ids': article_features[1],
            'input_mask': article_features[2],
            'segment_ids': article_features[3]
        }

        self.question_features = {
            'input_ids': question_features[1],
            'input_mask': question_features[2],
            'segment_ids': question_features[3]
        }

        self.option0_features = {
            'input_ids': option0_features[1],
            'input_mask': option0_features[2],
            'segment_ids': option0_features[3]
        }

        self.option1_features = {
            'input_ids': option1_features[1],
            'input_mask': option1_features[2],
            'segment_ids': option1_features[3]
        }

        self.option2_features = {
                'input_ids': option2_features[1],
                'input_mask': option2_features[2],
                'segment_ids': option2_features[3]
        }

        self.option3_features = {
                'input_ids': option3_features[1],
                'input_mask': option3_features[2],
                'segment_ids': option3_features[3]
        }

        self.label = label

def get_features(tokenizer, text, max_len):
    text_tokens = tokenizer.tokenize(text)
    _truncate_seq(text_tokens, max_len - 2)
    tokens = ["[CLS]"] + text_tokens + ["[SEP]"]

    segment_ids = [0] * (len(text_tokens) + 2) 
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len

    return (tokens, input_ids, input_mask, segment_ids)


def convert_examples_to_features(examples, tokenizer, article_len, question_len, option_len):
    """Loads a data file into a list of `InputBatch`s.
    Args:
        examples: RACEExamples 类型
        tokenizer: 分词器
        max_seq_length: 句子最大长度
    Returns:
        features:{
            example_id: 试题id
            choices_features :[
                {article + question + option1}, ... , {article + question + option2}
            ]
            label: 标签
        }
    """

    features = []
    for example_index, example in enumerate(examples):
        article_features = get_features(
            tokenizer, example.article, article_len)
        question_features = get_features(
            tokenizer, example.question, question_len)

        option0_features = get_features(
            tokenizer, example.option_0, option_len)
        option1_features = get_features(
            tokenizer, example.option_1, option_len)
        option2_features = get_features(
            tokenizer, example.option_2, option_len)
        option3_features = get_features(
            tokenizer, example.option_3, option_len)

        assert len(option0_features) == 4
        label = example.label

        features.append(
            InputFeatures(
                example_id=example.race_id,
                article_features=article_features,
                question_features=question_features,
                option0_features=option0_features,
                option1_features=option1_features,
                option2_features=option2_features,
                option3_features=option3_features, 
                label=label
            )
        )

    return features


def _truncate_seq(tokens,  max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while len(tokens) > max_length:
        tokens.pop()


def select_field(features, field):
    return [choice[field] for choice in features]

def get_tensor(features):
    all_input_ids = torch.tensor(
        select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(
        select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(
        select_field(features, 'segment_ids'), dtype=torch.long)
    
    return all_input_ids, all_input_mask, all_segment_ids
    

def convert_features_to_tensors(features, batch_size):
    article_fetures = [feature.article_features for feature in features]
    article_tensors = get_tensor(article_fetures)

    question_features = [feature.question_features for feature in features]
    question_tensors = get_tensor(question_features)

    option0_features = [feature.option0_features for feature in features]
    option0_tensors = get_tensor(option0_features)

    option1_features = [feature.option1_features for feature in features]
    option1_tensors = get_tensor(option1_features)

    option2_features = [feature.option2_features for feature in features]
    option2_tensors = get_tensor(option2_features)

    option3_features = [feature.option3_features for feature in features]
    option3_tensors = get_tensor(option3_features)

    all_label_ids = torch.tensor(
        [f.label for f in features], dtype=torch.long)

    data = TensorDataset(
        article_tensors[0], article_tensors[1], article_tensors[2],
        question_tensors[0], question_tensors[1], question_tensors[2],
        option0_tensors[0], option0_tensors[1], option0_tensors[2],
        option1_tensors[0], option1_tensors[1], option1_tensors[2],
        option2_tensors[0], option2_tensors[1], option2_tensors[2],
        option3_tensors[0], option3_tensors[1], option3_tensors[2],
        all_label_ids)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def read_race_examples(filename):

    examples = []

    with open(filename, 'r') as f:
        dataes = json.load(f)
        for data in dataes:
            examples.append(
                RaceExample(
                    race_id=data['race_id'],
                    article=data['article'],
                    question=data['question'],
                    option_0=data['option_0'],
                    option_1=data['option_1'],
                    option_2=data['option_2'],
                    option_3=data['option_3'],
                    label=data['label']
                )
            )

    return examples


def load_data(filename, tokenizer, article_len, question_len, option_len, batch_size):

    examples = read_race_examples(filename)
    features = convert_examples_to_features(
        examples, tokenizer, article_len, question_len, option_len)
    dataloader = convert_features_to_tensors(features, batch_size)
    return dataloader, len(examples)


if __name__ == "__main__":
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt", do_lower_case=True)

    filename = "/search/hadoop02/suanfa/songyingxin/data/RACE/all/test.json"

    dataes, data_len = load_data(filename, tokenizer, 120, 30, 20, 32)
    for data in dataes:
        print(1)

    # print(1)
    # out = get_features(tokenizer, 'illness', 10)
    # print(out)
