# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch

from Models.LSTM import LSTM
from Models.Linear import Linear


def gated_attention(article, question):
    """
    Args:
        article: [batch_size, article_len , dim]
        question: [batch_size, question_len, dim]
        gating_fn: 融合 article 与 question 信息的函数,有三种: mul, sum, cat
            mul: 元素相乘
            sum: 元素同维度求和
            cat: 拼接
    Returns:
        question_to_article: [batch_size, article_len, dim], 融合query信息的文章信息
    """
    question_att = question.permute(0, 2, 1)
    # question : [batch_size * dim * question_len]

    att_matrix = torch.bmm(article, question_att)
    # att_matrix: [batch_size * article_len * question_len]
    # 一行表示对于article中一个单词, 所有question中单词与该单词的相似度或相关度
    # 一列表示对于question 中的一个单词,所有 article 中单词与该单词的相似度或相关度

    att_weights = F.softmax(att_matrix.view(-1, inter.size(-1))).view_as(att_matrix)
    # att_weights: [batch_size, article_len, question_len]

    question_rep = torch.bmm(att_weights, question)
    # question_rep : [batch_size, article_len, dim]

    question_to_article = torch.mul(article, question_rep)
    # question_to_article: [batch_size, article_len, dim]

    return question_to_article
    


class GAReader(nn.Module):
    """
    RACE 数据集上所用模型与最初的GA Reader 有所不同:
    1. The query GRU is shared across hops.
    2. Dropout is applied to all hops (including the initial hop).
    3. Gated-attention is applied at the final layer as well.
    4. No character-level embeddings are used.
    """

    def __init__(self, embedding_dim, output_dim, hidden_size, rnn_num_layers, ga_layers, bidirectional, dropout, word_emb):
        super(GAReader, self).__init__()

        self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)

        self.rnn = LSTM(embedding_dim, hidden_size,
                        rnn_num_layers, bidirectional, dropout)

        self.ga_rnn = LSTM(hidden_size, hidden_size,
                           rnn_num_layers, bidirectional, dropout)
        
        self.ga_layers = ga_layers

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):

        article, article_lengths = batch.article
        # article: [article_len, batch_size], article_lengths: [batch_size]
        question, question_lengths = batch.question
        # question: [question_len, batch_size], question_lengths: [batch_size]

        option0, option0_lengths = batch.option_0
        option1, option1_lengths = batch.option_1
        option2, option2_lengths = batch.option_2
        option3, option3_lengths = batch.option_3
        # option: [option_len, batch_size]

        article_emb = self.dropout(self.word_embedding(article))
        # article_emb: [article_len, batch_size, emd_dim]

        question_emb = self.dropout(self.word_embedding(question))
        # question_emb: [question_len, batch_size, emd_dim]

        option0_emb = self.dropout(self.word_embedding(option0))
        option1_emb = self.dropout(self.word_embedding(option1))
        option2_emb = self.dropout(self.word_embedding(option2))
        option3_emb = self.dropout(self.word_embedding(option3))
        # option: [option_len, batch_size, emd_dim]

        _, question_out = self.rnn(question_emb, question_lengths)
        # question_out: [question_len, batch_size, hidden_size]

        _, option0_out = self.rnn(option0_emb, option0_lengths)
        _, option1_out = self.rnn(option1_emb, option1_lengths)
        _, option2_out = self.rnn(option2_emb, option2_lengths)
        _, option3_out = self.rnn(option3_emb, option3_lengths)
        # option_out: [option_len, batch_size, hidden_size]

        _, article_out = self.rnn(article_emb, article_lengths)
        # article_out: [article_len, batch_size, hidden_size]

        article_out.permute(1, 0, 2)
        question_out.permute(1, 0, 2)
        option0_out.permute(1, 0, 2)
        option1_out.permute(1, 0, 2)
        option2_out.permute(1, 0, 2)
        option3_out.permute(1, 0, 2)

        for layer in range(self.ga_layers):
                        
            article_emb = self.dropout(gated_attention(article_out, question_out))
            # article_emb: [batch_size, article_len, hidden_size]

            _, article_out = self.ga_rnn(article_emb, article_lengths)
            # article_out: [batch_size, article_len, hidden_size]
        
        








            









        












