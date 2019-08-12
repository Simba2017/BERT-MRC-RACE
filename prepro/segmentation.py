import json
import spacy

NLP = spacy.blank("en")


def word_tokenize(sent):
    doc = NLP(sent)
    return [token.text for token in doc]

def segmentation(input_file, out_file, judge):

    with open(input_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    for example in examples:
        article = example['article']
        articles = []
        segment = []
        segment_len = 0
        for seg in article:
            seg_len = len(word_tokenize(seg))
            if segment_len + seg_len < judge:
                segment.append(seg)
                segment_len += seg_len
            else:
                segment = '.'.join(segment)
                articles.append(segment)
                segment = []
                segment_len = 0
                
        segment = '.'.join(segment)
        articles.append(segment)
        example['article'] = articles
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f)


if __name__ == "__main__":
    
    dir = '/search/hadoop02/suanfa/songyingxin/data/RACE/all/segment/'
    segmentation(dir + 'train.json', dir + '400/' + 'train.json', 400)
    # segmentation(dir + 'dev.json', dir + '400/' + 'dev.json', 400)
    segmentation(dir + 'test.json', dir + '400/' + 'test.json', 400 )

