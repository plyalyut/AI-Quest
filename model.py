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
     "num_epochs": 1,
     "learning_rate": 0.01,
     "window_size": 40,
 }


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
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # These are all the sentence types that could happen. Feel free to add more if necessary!
    SPECIAL_TOKENS = {'prompt': '<|prompt|>', 'response': '<|response|>', 'action': '<|action|>', 'emote': '<|emote|>'}
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # Initialized the pre-trained GPT-2 model and optimizer
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)


    # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer

    train_loader, test_loader, vocab_sz = load_dataset((args.train_file, args.test_file), tokenizer, hyper_params['batch_size'], hyper_params['window_size'])
    model_embeddings = model.resize_token_embeddings(vocab_sz)
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
