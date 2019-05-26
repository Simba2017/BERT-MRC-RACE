
import torch.nn as nn
import torch.nn.functional as F
import torch

from Models.LSTM import LSTM
from Models.Linear import Linear

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
        
        self.ga_layers = ga_layers

        self.dropout = nn.Dropout(dropout)
    
    def forward(batch):

        article, article_lengths = batch.article
        question, question_lengths = batch.question
        option0, option0_lengths = batch.option0
        option1, option1_lengths = batch.option1
        option2, option2_lengths = batch.option2
        option3, option3_lengths = batch.option3

        article_emb = self.dropout(self.word_embedding(article))
        question_emb = self.dropout(self.word_embedding(question))
        option0_emb = self.dropout(self.word_embedding(option0))
        option1_emb = self.dropout(self.word_embedding(option1))
        option2_emb = self.dropout(self.word_embedding(option2))
        option3_emb = self.dropout(self.word_embedding(option3))

        _, question_out = self.rnn(question_emb, question_lengths)
        _, option0_out = self.rnn(option0_emb, option0_lengths)
        _, option1_out = self.rnn(option1_emb, option1_lengths)
        _, option2_out = self.rnn(option2_emb, option2_lengths)
        _, option3_out = self.rnn(option3_emb, option3_lengths)

        for layer in range(self.ga_layers):

            article_out = self.rnn(article_emb, article_lengths)

            # # GA Attention 的计算

            # article_inter_question = 









        












