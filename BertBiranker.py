import torch
from torch import nn
from torch.nn import functional as F


class BertBiranker(nn.Module):

    def __init__(self, bert_model, seq_length):
        '''
        Implementation for the BertBiranker Model described in the paper
        https://arxiv.org/pdf/1903.03094.pdf
        :param bert_model: pretrained bert model used as the base for training
        :param seq_length: length of each sentence input
        '''
        super(BertBiranker, self).__init__()
        self.bert_model = bert_model
        hidden_size = self.bert_model.config.hidden_size

        # Used for the representation of the ranking loss.
        self.embedding_layer = nn.Linear(hidden_size*seq_length, hidden_size)
        self.loss_func = nn.HingeEmbeddingLoss(reduction='mean')


    def forward(self, context, input, context_mask, input_mask, labels=None):
        '''
        Computes the distance between the norm between the embeddings.
        The context embedding and the input embeddings.
        :param context: context of the character as well as previous conversation history
        :param input: input from the character
        :return: loss, embeddings
        '''

        # TODO: incorporate masks in each forward pass
        context = self.bert_model(context, attention_mask=context_mask)[0]
        context = context.view(context.shape[0],-1)
        context_embedding = self.embedding_layer(context)
        softmax_context = F.softmax(context_embedding, dim=-1)

        input = self.bert_model(input, attention_mask=input_mask)[0]
        input = input.view(input.shape[0],-1)
        input_embedding = self.embedding_layer(input)
        softmax_input = F.softmax(input_embedding, dim=-1)

        # Softmax and dot product?
        similarity = torch.norm(softmax_context-softmax_input, p=2, dim=1)

        # TODO: Use a ranking loss to be maximized on random labels and minimized on correct labels
        if labels != None:
            return self.loss_func(similarity, labels), (context_embedding, input_embedding)
        else:
            return (context_embedding, input_embedding)





