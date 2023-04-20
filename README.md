
# NLP with RNN, LSTM, Seq2Seq, and Transformer models

## RNN and LSTM Unit

In this project, I will use PyTorch Linear layers and activations to implement a vanilla RNN unit. Then use PyTorch nn.Parameter and activations to implement an LSTM unit.



## Seq2Seq Implementation

In models/seq2seq you will see the files that implement the Seq2Seq model. These include the initialization and forward pass in __init__ and forward function, and the implementation of the Encoder and Decoder.


- **Seq2Seq with attention**

I will be implementing a simple form of attention to evaluate how it impacts the performance of the model. In particular, I will be implementing cosine similarity as the attention mechanism in decoder.py per this diagram.

- **Training and Hyperparameter Tuning**

Train seq2seq on the dataset with the default hyperparameters. Then perform hyperparameter tuning and include the improved results in a report explaining what I have tried.




## Transformers

I will be implementing a one-layer Transformer encoder which, similar to an RNN, can encode a sequence of inputs and produce a final output of possibility of tokens in target
language.

- **Embeddings**

I will format the input embeddings similarly to how they are constructed in [BERT(source of figure)](https://arxiv.org/pdf/1810.04805.pdf). Unlikea RNN, a Transformer does not include any positional information about the order in which the words in the sentence occur. Because of this, we need to append a positional encoding token at each position.

- **Multi-head Self-Attention**

We want to have multiple self-attention operations, computed in parallel. Each of these is called a head. We concatenate the heads and multiply them with the matrix attention_head_projection to produce the output of this layer. After every multi-head self-attention and feedforward layer, there is a residual connection + layer normalization.



