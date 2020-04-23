import pickle
from transformers import *
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset, Subset
import torch
from torch.nn.utils.rnn import pad_sequence

def load_dataset(files, tokenizer, batch_size, seq_len, unseen_data):
    if unseen_data:
        total_dataset = LightDataset(files[0], files[1], tokenizer, seq_len)
        train_indices = list(range(0, total_dataset.num_training_examples))
        test_indices = list(range(total_dataset.num_training_examples, len(total_dataset)))
        train_dataset = Subset(total_dataset, train_indices)
        test_dataset = Subset(total_dataset, test_indices)
    else:
        total_dataset = LightDataset(files[0], None, tokenizer, seq_len)
        split_size = int(len(total_dataset) * 0.9)
        train_dataset, test_dataset = random_split(total_dataset, [split_size, len(total_dataset) - split_size])
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader, total_dataset.vocab_size


class LightDataset(Dataset):
    def __init__(self, train_file, test_file, tokenizer, seq_len):
        self.tokenizer = tokenizer

        self.seq = []
        self.lengths = []
        self.masks = []
        self.max_seq_len = seq_len
        
        self.convert_file(train_file, tokenizer)
        if test_file != None:
            self.num_train_examples = len(self.lengths)
            self.convert_file(test_file, tokenizer)
        self.vocab_size = len(tokenizer)
                    
        self.seq = pad_sequence(self.seq, batch_first=True, padding_value=tokenizer.pad_token_id)

    def convert_file(self, file_name, tokenizer):
        with open(file_name, 'rb') as fp:
            data = pickle.load(fp)
            for episode in data:
                num_responses = len(episode['character'])
                for i in range(1, num_responses):
                    # we may need a separate dictionary for personas, actions, and emotes
                    # sorry if the added tokens are messy, we can modify them later as we go
                    first_persona = "<" + episode['character'][i - 1] + ">"
                    second_persona = "<" + episode['character'][i]
                    second_persona_action = episode['action'][i]
                    second_persona_emote = episode['emote'][i]

                    first_persona_speech = tokenizer.convert_tokens_to_ids(episode['speech'][i - 1].split())
                    second_persona_speech = tokenizer.convert_tokens_to_ids(episode['speech'][i].split())

                    tokenizer.add_tokens([first_persona, second_persona + ">"])
                    exchange = [tokenizer.convert_tokens_to_ids(first_persona)] + first_persona_speech + [tokenizer.convert_tokens_to_ids(second_persona + ">")] + second_persona_speech + [tokenizer.eos_token_id]
                    total_tokens = len(exchange)
                    self.seq.append(torch.tensor(exchange))
                    self.lengths.append(total_tokens)
                    ones = torch.ones(total_tokens)
                    zeros = torch.zeros(self.max_seq_len - total_tokens)
                    self.masks.append(torch.cat((ones, zeros)))

                    if second_persona_action is not None:
                        tokenizer.add_tokens(second_persona + "_action>")
                        exchange = [tokenizer.convert_tokens_to_ids(first_persona)] + first_persona_speech + [tokenizer.convert_tokens_to_ids(second_persona + "_action>")] + tokenizer.convert_tokens_to_ids(second_persona_action.split()) + [tokenizer.eos_token_id]
                        total_tokens = len(exchange)
                        self.seq.append(torch.tensor(exchange))
                        self.lengths.append(total_tokens)
                        ones = torch.ones(total_tokens)
                        zeros = torch.zeros(self.max_seq_len - total_tokens)
                        self.masks.append(torch.cat((ones, zeros)))

                    if second_persona_emote is not None:
                        tokenizer.add_tokens(second_persona + "_emote>")
                        exchange = [tokenizer.convert_tokens_to_ids(first_persona)] + first_persona_speech + [tokenizer.convert_tokens_to_ids(second_persona + "_emote>")] + tokenizer.convert_tokens_to_ids(second_persona_emote.split()) + [tokenizer.eos_token_id]
                        total_tokens = len(exchange)
                        self.seq.append(torch.tensor(exchange))
                        self.lengths.append(total_tokens)
                        ones = torch.ones(total_tokens)
                        zeros = torch.zeros(self.max_seq_len - total_tokens)
                        self.masks.append(torch.cat((ones, zeros)))

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.lengths)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {
            "seq": self.seq[idx],
            "lengths": self.lengths[idx],
            "mask":  self.masks[idx]
        }
        return item
