"""
S2S Decoder model.  (c) 2021 Georgia Tech

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


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        
        self.embedding_layer = nn.Embedding(output_size, emb_size)
        self.recurrent_layer = None
        if model_type == "RNN":
            self.recurrent_layer = nn.RNN(emb_size, decoder_hidden_size, batch_first = True)
        if model_type == "LSTM":
            self.recurrent_layer = nn.LSTM(emb_size, decoder_hidden_size, batch_first = True)
            

        self.L1 = nn.Linear(decoder_hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        self.dropout_layer = nn.Dropout(p=dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """
        attention = None

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        #############################################################################
        #  (N 5, 1, Z 2)  @ (N 5, Z 2, T 2) = (N, 1, T)

        K = nn.functional.normalize(encoder_outputs, p=2.0, dim=2)
        # print(K.size())
        q = nn.functional.normalize(hidden, p=2.0, dim=2) #  1,N, hidden_dim
        # print(q.size())
        
        K = torch.transpose(K, 1,2)
        q = torch.transpose(q, 0,1)
        # print(K.size()) #  5, 2 ,2
        # print(q.size()) #  5,1,2
        c = q @ K # 1,5,2 * 5,2,2
        # print(c.size())

        attention = nn.functional.softmax(c, dim =2)
        # print(attention)




        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention

    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the weights coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
            where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       if attention is true, compute the attention probabilities and use   #
        #       it to do a weighted average on the hidden and cell states to        #
        #       determine what will be consumed by the recurrent layer              #
        #                                                                           #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################

        emb = self.embedding_layer(input)
        dropout = self.dropout_layer(emb)
        if self.model_type == "LSTM":
            hidden, c_0 = hidden
            
        if attention:
            # STEP 5: CACULATING CONTEXT VECTOR by Multiplying Attention weights with encoder outputs
            atte = self.compute_attention(hidden, encoder_outputs)
            # N 1 T     N T H  N1H  (encoder_outputs)--> 1,N,H
            weighted_k = atte @ encoder_outputs

            hidden = torch.transpose(weighted_k, 0,1)
            if self.model_type == "LSTM":

                atte_cell = self.compute_attention(c_0, encoder_outputs)
                # N 1 T     N T H  N1H  (encoder_outputs)--> 1,N,H
                weighted_k = atte_cell @ encoder_outputs
                
                c_0 = torch.transpose(weighted_k, 0,1)


        
        
        if self.model_type == "LSTM":

            output, (hidden, c_0) = self.recurrent_layer(dropout, (hidden, c_0))
            output = self.L1(output)
            output = self.logsoftmax(output)
            return output[:,0,:], (hidden, c_0)
        if self.model_type == "RNN":
            output, hidden = self.recurrent_layer(dropout, hidden)
        

            output = self.L1(output)
            output = self.logsoftmax(output)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
            return output[:,0,:], hidden
