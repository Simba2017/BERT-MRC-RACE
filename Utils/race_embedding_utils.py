import torch
from torchtext import data
from torchtext import datasets
from torchtext import vocab


def load_race(path, id_field, word_field, label_field, batch_size, device, embedding_file):

    fields = {
        'race_id': ('race_id', id_field),
        'article': ('article', word_field),
        'question': ('question', word_field),
        'option_0': ('option_0', word_field),
        'option_1': ('option_1', word_field),
        'option_2': ('option_2', word_field),
        'option_3': ('option_3', word_field),
        'label': ('label', label_field)
    }

    train, dev, test = data.TabularDataset.splits(
        path=path, train='train.jsonl', validation='dev.jsonl',
        test='test.jsonl', format='json', fields=fields)
    
    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(dev.examples), len(test.examples)))
    
    vectors = vocab.Vectors(embedding_file)

    word_field.build_vocab(train, dev, test, max_size=25000,
                           vectors=vectors, unk_init=torch.Tensor.normal_)
    
    label_field.build_vocab(train, dev, test)
    
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False, shuffle=True, device=device
    )

    return train_iter, dev_iter, test_iter




if __name__ == "__main__":

    id_field = data.RawField()
    id_field.is_target = False
    text_field = data.Field(tokenize='spacy', lower=True,
                            include_lengths=True, fix_length=30)
    label_field = data.LabelField(dtype=torch.long)
    
    path = "/home/songyingxin/datasets/RACE/all"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    word_emb_file = "/home/songyingxin/datasets/WordEmbedding/glove/glove.840B.300d.txt"

    load_race(path, id_field, text_field, label_field, 32, device, word_emb_file)




