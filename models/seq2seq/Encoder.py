"""
S2S Encoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        # num_embeddings (int) – size of the dictionary of embeddings

        # embedding_dim (int) – the size of each embedding vector
        self.embedding_layer = nn.Embedding(input_size, emb_size)

        self.recurrent_layer = None
        if model_type == "RNN":
            # input_size – The number of expected features in the input x
            # hidden_size – The number of features in the hidden state h
            # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
            self.recurrent_layer = nn.RNN(emb_size, encoder_hidden_size, batch_first = True)
        if model_type == "LSTM":
            self.recurrent_layer = nn.LSTM(emb_size, encoder_hidden_size, batch_first = True)
            

        self.L1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.ReLU = nn.ReLU()
        self.L2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder; k,

                hidden (tensor): the weights coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################

        emb = self.embedding_layer(input)
        dropout = self.dropout_layer(emb)
        output, hidden = None, None
        c_0 = None
        if self.model_type == "LSTM":
            output, (hidden, c_0) = self.recurrent_layer(dropout)
            # c_0 = self.tanh(c_0)
        else:    
            output, hidden = self.recurrent_layer(dropout)
        hidden = self.L1(hidden)
        hidden = self.ReLU(hidden)
        hidden = self.L2(hidden)
        hidden = self.tanh(hidden)

        if self.model_type == "LSTM":
            return output, (hidden, c_0)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # Do not apply any linear layers/Relu for the cell state when model_type is #
        # LSTM before returning it.                                                 #

        return output, hidden
