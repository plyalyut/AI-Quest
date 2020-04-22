from comet_ml import Experiment
import torch
import torch.nn
import argparse
import math
import numpy as np
from transformers import *
from preprocess import *
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 50,
     "num_epochs": 3,
     "learning_rate": 0.001
 }

def train_model(model, train_loader, optimizer, experiment):
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
            for input, length in tqdm(train_loader):
                input = input.long().to(DEVICE)
                optimizer.zero_grad()
                loss = model(input, labels = input)[0]
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
    parser.add_argument("train_file")
    parser.add_argument("test_file")
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
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer, add any special token if needed
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', eos_token='<EOS>', pad_token='<PAD>')

    # These are all the sentence types that could happen. Feel free to add more if necessary!
    # SPECIAL_TOKENS = {'prompt': '<|prompt|>', 'response': '<|response|>', 'action': '<|action|>', 'emote': '<|emote|>'}
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # Initialized the pre-trained GPT-2 model and optimizer
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)


    # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer
    # Need toggle for seen and unseen test dataset
    train_loader, test_loader, vocab_size = load_dataset((args.train_file, args.test_file), tokenizer, hyper_params['batch_size'], False) # TODO: feel free to change this up
    model_embeddings = model.resize_token_embeddings(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        # run train loop here
        print("running training loop...")
        train_model(model, train_loader, optimizer, experiment)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test_model(model, test_loader, experiment)


    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive_model(input_text, tokenizer, model)
