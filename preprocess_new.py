import pickle
from transformers import *
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset, Subset
import torch
from torch.nn.utils.rnn import pad_sequence


def load_dataset(files, tokenizer, batch_size, seq_len, unseen_data, special_tokens):
    # Builds data loaders
    if unseen_data:
        total_dataset = GPTDataset(files[0], files[1], tokenizer, seq_len)
        train_indices = list(range(0, total_dataset.num_training_examples))
        test_indices = list(range(total_dataset.num_training_examples, len(total_dataset)))
        train_dataset = Subset(total_dataset, train_indices)
        test_dataset = Subset(total_dataset, test_indices)
    else:
        total_dataset = GPTDataset(files[0], None, tokenizer, seq_len)
        split_size = int(len(total_dataset) * 0.9)
        train_dataset, test_dataset = random_split(total_dataset, [split_size, len(total_dataset) - split_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader


def convert_episode_context_to_data_point(episode):
    # Gets context for the entire scene
    data_point = ' <|setting_name|> ' + episode['setting']['name'] + ", " + episode['setting']['category']
    data_point += ' <|setting_desc|> ' + episode['setting']['description']
    data_point += ' <|partner_name|> ' + episode['agents'][1]['name']
    data_point += ' <|self_name|> ' + episode['agents'][0]['name']
    data_point += ' <|partner_persona|> ' + episode['agents'][1]['persona']
    data_point += ' <|self_persona|> ' + episode['agents'][0]['persona']
    partner = episode['agents'][1]['name']
    self_character = episode['agents'][0]['name']

    # We can join the data_points together
    return data_point, partner, self_character


def get_previous_utterances(episode, response_num, partner):
    # builds speech over conversation
    res = ''
    if episode['character'][response_num] == partner:
        res += ' <|partner_say|> ' + episode['speech'][response_num]
        if episode['action'][response_num] is not None:
            res += ' <|partner_act|> ' + episode['action'][response_num]
        if episode['emote'][response_num] is not None:
            res += ' <|partner_emote|> ' + episode['emote'][response_num]
    else:
        res += ' <|self_say|> ' + episode['speech'][response_num]
        if episode['action'][response_num] is not None:
            res += ' <|self_act|> ' + episode['action'][response_num]
        if episode['emote'][response_num] is not None:
            res += ' <|self_emote|> ' + episode['emote'][response_num]
    return res


# Uses GPT2 Tokenizer. Is very slow as is
class GPTDataset(Dataset):
    def __init__(self, train_file, test_file, tokenizer, seq_len):
        self.tokenizer = tokenizer

        self.seq = []
        self.lengths = []
        self.masks = []
        self.max_seq_len = seq_len

        self.convert_file(train_file)
        if test_file != None:
            self.num_train_examples = len(self.lengths)
            self.convert_file(test_file)
        # self.vocab_size = len(tokenizer)

        self.seq = pad_sequence(self.seq, batch_first=True, padding_value=tokenizer.pad_token_id)

    def convert_file(self, file_name):
        with open(file_name, 'rb') as fp:
            data = pickle.load(fp)
            for episode in data:
                context, partner, self_character = convert_episode_context_to_data_point(episode)
                num_responses = len(episode['character'])
                previous_text = ''
                for i in range(1, num_responses):

                    # objects for current call and response
                    current_input = context
                    for obj in episode['room_objects'][i]:
                        current_input += '<|object_desc|> ' + obj + " : " + episode['all_descriptions'][obj]

                    # previous text in conversation
                    previous_text += get_previous_utterances(episode, i - 1, partner)
                    current_input += previous_text

                    speech_input = current_input + ' <|task_speech|> ' + episode['speech'][i]
                    # if i == num_responses - 1:
                    #    #self.tokenizer.add_tokens(speech_input)
                    #    pass
                    total_tokens = len(speech_input)
                    self.seq.append(torch.tensor(self.tokenizer.encode(speech_input)))
                    self.lengths.append(total_tokens)
                    # Check masking once window size is found
                    ones = torch.ones(total_tokens)
                    # zeros = torch.zeros(self.max_seq_len - total_tokens)
                    # self.masks.append(torch.cat((ones, zeros)))

                    second_persona_action = episode['action'][i]
                    second_persona_emote = episode['emote'][i]
                    if second_persona_action is not None:
                        action_input = current_input + ' <|task_action|> ' + second_persona_action
                        # self.tokenizer.add_tokens(second_persona_action.split())
                        total_tokens = len(action_input)
                        self.seq.append(torch.tensor(self.tokenizer.encode(action_input)))
                        self.lengths.append(total_tokens)
                        # self.tokenizer.add_tokens(action_input)
                        # Check masking once window size is found
                        ones = torch.ones(total_tokens)
                        # zeros = torch.zeros(self.max_seq_len - total_tokens)
                        # self.masks.append(torch.cat((ones, zeros)))

                    if second_persona_emote is not None:
                        emote_input = current_input + ' <|task_emote|> ' + second_persona_emote
                        # self.tokenizer.add_tokens(second_persona_emote.split())
                        total_tokens = len(emote_input)
                        self.seq.append(torch.tensor(self.tokenizer.encode(emote_input)))
                        self.lengths.append(total_tokens)
                        # self.tokenizer.add_tokens(emote_input)
                        # Check masking once window size is found
                        ones = torch.ones(total_tokens)
                        # zeros = torch.zeros(self.max_seq_len - total_tokens)
                        # self.masks.append(torch.cat((ones, zeros)))

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        item = {
            "seq": self.seq[idx],
            "lengths": self.lengths[idx],
            # "mask":  self.masks[idx]
        }
        return item


# Converts data using a word2id dictionary
class LightDataset(Dataset):
    def __init__(self, train_file, test_file, seq_len, special_tokens):
        # self.tokenizer = tokenizer

        self.seq, self.lengths, self.masks = [], [], []
        self.max_seq_len = seq_len
        self.word2id = {}
        self.word2id["<|PAD|>"] = 0
        self.vocab_size = 1
        self.add_words(special_tokens)

        self.convert_file(train_file)
        if test_file != None:
            self.num_train_examples = len(self.lengths)
            self.convert_file(test_file)

        self.seq = pad_sequence(self.seq, batch_first=True, padding_value=0)

    def add_words(self, sentence):
        '''
        Add words to the vocab.
        :param sentence: the sentence to be encoded
        '''
        res = []
        for word in sentence:
            if not self.word2id.get(word):
                self.word2id[word] = self.vocab_size
                self.vocab_size = self.vocab_size + 1
            res.append(self.word2id[word])
        return res

    def convert_file(self, file_name):
        '''
        Reads in pickle file and extracts meaningful information.
        '''

        with open(file_name, 'rb') as fp:
            data = pickle.load(fp)
            for episode in data:
                context, partner, self_character = convert_episode_context_to_data_point(episode)
                num_responses = len(episode['character'])
                previous_text = []
                for i in range(1, num_responses):

                    # objects for current call and response
                    current_input = list(context)
                    for obj in episode['room_objects'][i]:
                        current_input += ['<|object_desc|>'] + (obj + " : " + episode['all_descriptions'][obj]).split()

                    # previous text in conversation
                    previous_text += get_previous_utterances(episode, i - 1, partner)
                    current_input += previous_text

                    speech_input = current_input + ['<|task_speech|>'] + episode['speech'][i].split()
                    total_tokens = len(speech_input)
                    self.seq.append(torch.tensor(self.add_words(speech_input)))
                    self.lengths.append(total_tokens)
                    ones = torch.ones(total_tokens)
                    zeros = torch.zeros(self.max_seq_len - total_tokens)
                    self.masks.append(torch.cat((ones, zeros)))

                    second_persona_action = episode['action'][i]
                    second_persona_emote = episode['emote'][i]
                    if second_persona_action is not None:
                        action_input = current_input + ['<|task_action|>'] + second_persona_action.split()
                        # self.tokenizer.add_tokens(second_persona_action.split())
                        total_tokens = len(action_input)
                        self.seq.append(torch.tensor(self.add_words(action_input)))
                        self.lengths.append(total_tokens)
                        ones = torch.ones(total_tokens)
                        zeros = torch.zeros(self.max_seq_len - total_tokens)
                        self.masks.append(torch.cat((ones, zeros)))

                    if second_persona_emote is not None:
                        emote_input = current_input + ['<|task_emote|>'] + second_persona_emote.split()
                        # self.tokenizer.add_tokens(second_persona_emote.split())
                        total_tokens = len(emote_input)
                        self.seq.append(torch.tensor(self.add_words(emote_input)))
                        self.lengths.append(total_tokens)
                        ones = torch.ones(total_tokens)
                        zeros = torch.zeros(self.max_seq_len - total_tokens)
                        self.masks.append(torch.cat((ones, zeros)))

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        item = {
            "seq": self.seq[idx],
            "lengths": self.lengths[idx],
            "mask": self.masks[idx]
        }
        return item
