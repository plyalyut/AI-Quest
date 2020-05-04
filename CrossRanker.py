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


    def forward(self, seq, masks, type_ids, labels=None):

        # TODO: incorporate masks in each forward pass
        if labels != None:
            loss, out = self.bert_model(seq, attention_mask=masks, token_type_ids=type_ids, next_sentence_label=labels)[:2]
            print(out)
            return loss, out
        else:
            return self.bert_model(seq, attention_mask=masks, token_type_ids=type_ids)[0]


        





