from comet_ml import Experiment
import torch
import torch.nn
import argparse
import math
import numpy as np
from transformers import *
from preprocess import *
from tqdm import tqdm
from BertBiranker import BertBiranker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")

hyper_params = {
     "batch_size": 2,
     "num_epochs": 3,
     "learning_rate": 0.001, 
     'seq_len': 512 #Actual sequence length is 1201. But model crashes when this large. Need to determine a workaround
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
    with experiment.train():
        for i in range(hyper_params['num_epochs']):
            for data in tqdm(train_loader):
                optimizer.zero_grad()
                if model_type == "gpt2":
                    input = data['seq'].long().to(DEVICE)
                    lengths = data['lengths'].long().to(DEVICE)
                    masks = data['mask'].to(DEVICE)
                    loss = model(input, labels = input, attention_mask=masks)[0]
                elif model_type == "bert":
                    context = data['context'].to(DEVICE)
                    context_mask = data['context_mask'].to(DEVICE)
                    input_text = data['input'].to(DEVICE)
                    input_mask = data['input_mask'].to(DEVICE)
                    labels = data['label'].to(DEVICE)
                    loss, (context_embedding, input_embedding) = model(context, input_text, context_mask, input_mask, labels=labels)

                loss.backward()  # calculate gradients
                optimizer.step()  # update model weights


def test_model(model, test_loader, experiment):
    """
    Tests the model using the testing dataset, evaluates the perplexity.
    :param model: the initilized model to use for forward pass
    :param loader: Dataloader of test data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """

    pass

def interactive():
    '''TODO'''
    pass 


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
    help="gpt2 or bert")
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    
    if args.model == "gpt2":
        # Load the GPT2 Tokenizer, add any special token if needed
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
    elif args.model == "bert":
        # Load the Bert Tokenizer, add any special token if needed
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',  pad_token='<PAD>', sep_token='<SEP>')



        
    # These are all the sentence types that could happen. Feel free to add more if necessary!
    SPECIAL_TOKENS = ['<|task_speech|>', '<|task_action|>', '<|task_emote|>', '<|setting_name|>', '<|setting_desc|>', '<|partner_name|>', '<|self_name|>', '<|partner_persona|>', '<self_persona>', '<|object_desc|>', '<|partner_say|>', '<|partner_act|>', '<|partner_emote|>', '<|self_say|>', '<|self_act|>', '<|self_emote|>']
    tokenizer.add_tokens(SPECIAL_TOKENS)

    if args.train_file == None and args.test_file == None:
        print("loading saved data loaders...")
        train_loader = torch.load('./train_loader.pt')
        # test_loader = torch.load('./test_loader.pt')

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
        # Load the Bert Tokenizer, add any special token if needed
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',  pad_token='<PAD>', sep_token='<SEP>')
        # Initialized the pre-trained BERT model
        pretrained_model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)

        # Initializing a model from the bert-base-uncased style configuration
        pretrained_model_embeddings = pretrained_model.resize_token_embeddings(len(tokenizer) + len(SPECIAL_TOKENS))
        model = BertBiranker(pretrained_model, seq_length=hyper_params['seq_len']).to(DEVICE)

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
        test_model(model, test_loader, experiment)


    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive_model(input_text, tokenizer, model)
