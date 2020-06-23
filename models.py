import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np
from torch.nn.utils.rnn import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
TF_SCH = 0 
def concat_pblstm(x):
    L, B, D = x.shape
    if (L%2 != 0):
        x = x[:-1,:,:]
    x = torch.transpose(x, 1, 2)
    x = x.T
    x = torch.transpose(x, 1, 2)
    x = x.reshape((-1, L//2, 2*D))
    x = torch.transpose(x, 0, 1)
    return x


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted  
        '''
        # Compute (batch_size, max_len) attention logits. "bmm" stands for "batch matrix multiplication".
        # Input shape of bmm:  (batch_szie, max_len, hidden_size), (batch_size, hidden_size, 1) 
        # Output shape of bmm: (batch_size, max_len, 1)
        key = torch.transpose(key, 0, 1)
        value = torch.transpose(value, 0, 1)
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        
        # Create an (batch_size, max_len) boolean mask for all padding positions
        # Make use of broadcasting: (1, max_len), (batch_size, 1) -> (batch_size, max_len)
        mask = torch.arange(key.size(1)).unsqueeze(0).to(DEVICE) >= lens.unsqueeze(1)
        
        # Set attention logits at padding positions to negative infinity.
        energy.masked_fill_(mask, -1e9)
        
        # Take softmax over the "source length" dimension.
        attention = nn.functional.softmax(energy, dim=1)
        
        # Compute attention-weighted sum of context vectors
        # Input shape of bmm: (batch_size, 1, max_len), (batch_size, max_len, hidden_size) 
        # Output shape of bmm: (batch_size, 1, hidden_size)
        out = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        
        # attention vectors are returned for visualization
        return out, attention


class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        out, out_lens = pad_packed_sequence(x) # Unpack the data # (Time, batch, dims)
        out = concat_pblstm(out)
        out_lens = out_lens//2
        packed_out = pack_padded_sequence(out, out_lens, enforce_sorted=False, batch_first=False)
        packed_out = self.blstm(packed_out)[0]
        return packed_out


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.linear_1 = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, bidirectional=True) # (32, 32)

        ### Add code to define the blocks of pBLSTMs! ###
        self.pblstm1 = pBLSTM(1024, 256)
        self.pblstm2 = pBLSTM(1024, 256)
        self.pblstm3 = pBLSTM(1024, hidden_dim)
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        x = self.linear_1(x)
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        outputs = self.pblstm1(outputs)
        outputs = self.pblstm2(outputs)
        outputs = self.pblstm3(outputs)
        ### Use the outputs and pass it through the pBLSTM blocks! ###
        linear_input, _ = utils.rnn.pad_packed_sequence(outputs) # (seq, batch, dims)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, isTrain=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''

        global TF_SCH
        batch_size = key.shape[1]

        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = 33*torch.ones(batch_size,1).to(DEVICE)
        context = None
        tf_rate = 0.9**TF_SCH
        TF_SCH = TF_SCH + 1
        tf_rate = max(tf_rate, 0.6)
        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do you do not get index out of range errors. 

            if (isTrain):
                prediction = nn.functional.gumbel_softmax(prediction)
                # torch.distributions.gumbel.Gumbel(loc, scale)
                if (np.random.uniform() > tf_rate):
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    char_embed = embeddings[:,i,:]
                
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            # Teacher forcing is maybe choosing between the above two char_embed values
            
            if (self.isAttended and i>0):
                inp = torch.cat([char_embed, context], dim=1)
            else:
                inp = torch.cat([char_embed, values[i,:,:]], dim=1)
                
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]
            if (self.isAttended):
                context, _ = self.attention.forward(output, key, values, lens)
                prediction = self.character_prob(torch.cat([output, context], dim=1))
            else:
                prediction = self.character_prob(torch.cat([output, values[i,:,:]], dim=1))

            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=256, key_size=256, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim, isAttended=isAttended)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        key, value = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, speech_len, text_input, isTrain=True)
        else:
            predictions = self.decoder(key, value, speech_len, text=None, isTrain=False)
        return predictions
