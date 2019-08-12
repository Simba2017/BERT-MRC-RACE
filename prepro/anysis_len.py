import json
import spacy

NLP = spacy.blank("en")


def word_tokenize(sent):
    doc = NLP(sent)
    return [token.text for token in doc]

def anysis_len(input_file, out_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    article_lens = {}

    for example in examples:
        article = example['article']
        article_len = (len(word_tokenize(article)) // 400 + 1) * 400

        if article_len not in article_lens.keys():
            article_lens[article_len] = 1
        else:
            article_lens[article_len] += 1
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(article_lens, f)


if __name__ == "__main__":
    
    dir = '/search/hadoop02/suanfa/songyingxin/data/RACE/all/'
    # anysis_len(dir + 'train.json', 'train_lens.json')
    anysis_len(dir + 'dev.json', 'dev_lens.json')
    anysis_len(dir + 'test.json', 'test_lens.json')

    
        


