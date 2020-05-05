from comet_ml import Experiment
import torch
import torch.nn
import argparse
import math
import numpy as np
from torch import nn
from transformers import *
from preprocess import *
from tqdm import tqdm
from BertBiranker import BertBiranker
from CrossRanker import CrossRanker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 3,
     "num_epochs": 2,
     "learning_rate": 0.001,
     'seq_len': 512,
     'accumulation_steps': 10
 }

def train_model(model, train_loader, optimizer, experiment, model_type):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the training loop here, save trained model weights if needed
    model = model.train()
    if model_type == "bert":
        loss_func = nn.MSELoss(reduction='sum')

    with experiment.train():
        for i in range(hyper_params['num_epochs']):
            optimizer.zero_grad()
            train_count = 0
            for data in tqdm(train_loader):
                train_count += 1
                if model_type == "gpt2":
                    input = data['seq'].long().to(DEVICE)
                    lengths = data['lengths'].long().to(DEVICE)
                    masks = data['mask'].to(DEVICE)
                    loss = model(input, labels=input, attention_mask=masks)[0]
                elif model_type == "bert":
                    context = data['context'].to(DEVICE)
                    context_mask = data['context_mask'].to(DEVICE)
                    input_text = data['input'].to(DEVICE)
                    input_mask = data['input_mask'].to(DEVICE)
                    labels = data['label'].to(DEVICE)
                    context_embed, input_embed = model(context, input_text, context_mask, input_mask)
                    #loss = loss_func(similarity.float(), labels.float())
                    loss = nn.CosineEmbeddingLoss()(context_embed, input_embed, labels)
                elif model_type == "cross":
                    input = data['seq'].long().to(DEVICE)
                    masks = data['mask'].long().to(DEVICE)
                    position_ids = data["position_ids"].long().to(DEVICE)
                    labels = data['label'].long().to(DEVICE)
                    loss, prediction = model(input, masks, position_ids, labels=labels)
                    # loss = loss_func(prediction.float(), labels.float())
                loss = loss/hyper_params['accumulation_steps']
                loss.backward()  # calculate gradients
                if train_count % hyper_params['accumulation_steps'] == 0:
                    optimizer.step()  # update model weights
                    optimizer.zero_grad()


def test_model(model, test_loader, experiment, model_type):
    """
    Tests the model using the testing dataset, evaluates the perplexity.
    :param model: the initilized model to use for forward pass
    :param loader: Dataloader of test data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """

    model = model.eval()
    total_correct = 0.0
    total_predicted = 0
    total_loss = 0.0
    word_count = 0
    with experiment.validate():
        for data in tqdm(test_loader):
                if model_type == "cross":
                    input = data['seq'].long().to(DEVICE)
                    masks = data['mask'].long().to(DEVICE)
                    position_ids = data['position_ids'].long().to(DEVICE)
                    labels = data['label'].long().to(DEVICE)
                    loss, prediction = model(input, masks, position_ids, labels=labels)
                    correct = (torch.argmax(prediction) == torch.argmin(labels))
                    total_correct = total_correct + correct
                    total_predicted = total_predicted + 1
                elif model_type == "gpt2":
                    x = data["seq"].to(DEVICE)
                    masks = data['mask'].to(DEVICE)
                    lengths = data["lengths"].to(DEVICE)
                    outputs = model(x, labels=x, attention_mask=masks)
                    loss, logits = outputs[:2]
                    prediction = torch.argmax(logits, dim=-1)
                    total_correct += torch.sum(x.flatten() == prediction.flatten())
                    total_predicted += (torch.sum(masks.flatten()))
                    num_words = torch.sum(lengths)
                    total_loss = total_loss + (loss.item() * num_words.item())
                    word_count = word_count + num_words.item()
    
    if model_type == "gpt2":
        loss_per_word = total_loss / word_count
        perplexity = torch.exp(torch.tensor(loss_per_word))
        print("perplexity:", perplexity.item())
        experiment.log_metric("perplexity", perplexity.item())
    
    accuracy = total_correct / total_predicted
    print("accuracy: ", accuracy.item())
    experiment.log_metric("accuracy", accuracy.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="run in interactive mode")
    parser.add_argument("-m", "--model", type=str, default="",
    help="gpt2 or bert or cross")
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    
    if args.model == "gpt2":
        # Load the GPT2 Tokenizer, add any special token if needed
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
    elif args.model == "bert" or args.model == "cross":
        # Load the Bert Tokenizer, add any special token if needed
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',  pad_token='<PAD>', sep_token='<SEP>')



        
    # These are all the sentence types that could happen. Feel free to add more if necessary!
    SPECIAL_TOKENS = ['<|task_speech|>', '<|task_action|>', '<|task_emote|>', '<|setting_name|>', '<|setting_desc|>', '<|partner_name|>', '<|self_name|>', '<|partner_persona|>', '<self_persona>', '<|object_desc|>', '<|partner_say|>', '<|partner_act|>', '<|partner_emote|>', '<|self_say|>', '<|self_act|>', '<|self_emote|>']
    tokenizer.add_tokens(SPECIAL_TOKENS)

    if args.train_file == None and args.test_file == None:
        print("loading saved data loaders...")
        train_loader = torch.load('./train_loader.pt')
        test_loader = torch.load('./test_loader.pt')

    else:
        train_loader, test_loader = load_dataset(args.train_file, args.test_file, hyper_params['batch_size'], hyper_params['seq_len'], SPECIAL_TOKENS, tokenizer, args.model) 
        print("saving data loaders...")
        torch.save(train_loader, './train_loader.pt')
        torch.save(test_loader, './test_loader.pt')

    if args.model == "gpt2":
        # Initialized the pre-trained GPT-2 model
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
        model_embeddings = model.resize_token_embeddings(len(tokenizer))

    elif args.model == "bert":
        # Initialized the pre-trained BERT model
        pretrained_model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
        # Initializing a model from the bert-base-uncased style configuration
        pretrained_model_embeddings = pretrained_model.resize_token_embeddings(len(tokenizer) + len(SPECIAL_TOKENS))
        model = BertBiranker(pretrained_model, seq_length=hyper_params['seq_len']).to(DEVICE)

    elif args.model == "cross":
        # Initialized the pre-trained BERT model
        pretrained_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased').to(DEVICE)
        # Initializing a model from the bert-base-uncased style configuration
        pretrained_model_embeddings = pretrained_model.resize_token_embeddings(len(tokenizer) + len(SPECIAL_TOKENS))
        model = CrossRanker(pretrained_model, hyper_params['seq_len'], hyper_params['batch_size']).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train_model(model, train_loader, optimizer, experiment, args.model)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.test:
        print("running testing loop...")
        test_model(model, test_loader, experiment, args.model)


    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive_model(input_text, tokenizer, model)
