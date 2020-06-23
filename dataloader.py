import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import *


'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('../data/train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('../data/dev_new.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('../data/test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('../data/train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('../data/dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter2index, _ = create_dictionaries(letter_list)
    all_transcipts_ints = []
    for i in range(transcript.shape[0]):
        transcript_str = list(transcript[i].astype(str))
        transcript_str = " ".join(transcript_str)
        transcript_int = [letter2index[k] for k in transcript_str]
        transcript_int.insert(0,letter2index['<sos>'])
        transcript_int.append(letter2index['<eos>'])
        all_transcipts_ints.append(np.asarray(transcript_int))

    return  np.asarray(all_transcipts_ints)

# @TODO:Idx to letter fucn
'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = {val : idx for idx, val in enumerate(letter_list)}
    index2letter = {v: k for k, v in letter2index.items()}
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        
        self.X = [torch.FloatTensor(x) for x in self.speech] # Train data
        self.X_lens = torch.LongTensor([len(seq) for seq in self.X])
        self.X = pad_sequence(self.X)

        self.Y = None
        self.Y_lens = None
        if (text is not None):
            self.text = text
            self.Y = [torch.LongTensor(y) for y in self.text]
            self.Y_lens = torch.LongTensor([len(seq_lbl) for seq_lbl in self.Y])
            self.Y = pad_sequence(self.Y, batch_first=True)

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return self.X[:, index, :], self.X_lens[index], self.Y[index, :], self.Y_lens[index]
        else:
            return self.X[:, index, :], self.X_lens[index]


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    pass 


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    pass 