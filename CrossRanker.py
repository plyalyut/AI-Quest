import torch
from torch import nn
from torch.nn import functional as F


class CrossRanker(nn.Module):

    def __init__(self, bert_model, seq_length, batch_size):
        super(CrossRanker, self).__init__()
        self.bert_model = bert_model
        # Used for the representation of the ranking loss.
        hidden_size = self.bert_model.config.hidden_size
        self.embedding_layer = nn.Linear(hidden_size*seq_length, hidden_size)


    def forward(self, seq, masks):
        '''
        Computes the distance between the norm between the embeddings.
        The context embedding and the input embeddings.
        :param context: context of the character as well as previous conversation history
        :param input: input from the character
        :return: loss, embeddings
        '''

        # TODO: incorporate masks in each forward pass
        context = self.bert_model(seq, attention_mask=masks)[0]
        context = context.view(seq.shape[0],-1)
        context_embedding = self.embedding_layer(context)
        return F.softmax(context_embedding, dim=0, dtype=torch.float32)[:, -1]

        





