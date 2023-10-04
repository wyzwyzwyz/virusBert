'''
Author: your name
Date: 2020-12-07 13:47:31
LastEditTime: 2021-10-18 04:27:05
LastEditors: Yang jin
Description: In User Settings Edit
FilePath: /BERT-pytorch/bert_pytorch/model/bert.py
'''
import torch.nn as nn
import torch

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .config import BERTConfig


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config:BERTConfig):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(BERT,self).__init__()
        self.hidden = config.hidden
        self.n_layers = config.n_layers
        self.attn_heads = config.attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = config.hidden * 4

        # embedding for BERT, sum of positional, and token embeddings
        self.embedding = BERTEmbedding(vocab_size=config.vocab_size, embed_size=config.hidden,dropout = config.dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config.hidden, config.attn_heads, config.hidden * 4, config.dropout) for _ in range(config.n_layers)])


    def forward(self, x,segment_info=None,output_all_encoded_layers= False):
        # attention masking for padded token

        # torch.ByteTensor([batch_size, 1, seq_len, seq_len])
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        all_encoder_layer = []

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x,segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            if output_all_encoded_layers:
                all_encoder_layer.append(x)
            x = transformer.forward(x, mask)
        
        if not output_all_encoded_layers:
            all_encoder_layer = x
            
        return all_encoder_layer

class BERTMM(nn.Module):
    """
    BERT Language Model
    Only Masked Language Model
    """
    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
    
    def forward(self, x,output_all_encoded_layers= False):

        x = self.bert(x,None,output_all_encoded_layers)
      
        return self.mask_lm(x)

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
    
    def forward(self, x,segment_label,output_all_encoded_layers= False):
        
        x = self.bert(x,segment_label,output_all_encoded_layers)
      
        return self.next_sentence(x),self.mask_lm(x)

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTPooler(nn.Module):
    def __init__(self,config):
        super(BERTPooler,self).__init__()
        self.dense =nn.Linear(config.hidden,config.hidden)
        self.activation = nn.Tanh()
        self.hidden = config.hidden
    
    def mean_pooling(self,hidden_states):
        pooled_out = torch.mean(hidden_states,dim=1)
        return pooled_out
    
    def max_pooling(self,hidden_states):
        pooled_out = torch.max(hidden_states,dim=1)[0]
        return pooled_out
    
    def cls_pooling(self,hidden_states):
        first_token_tensor = hidden_states[:,0]
        return first_token_tensor

    def forward(self,hidden_states,pooling_strategy):
        """
        We "pool" the model by simply taking the hidden state corresponding to the first token
        """
        if pooling_strategy == 'MEAN':
            pooled_output = self.mean_pooling(hidden_states)
        elif pooling_strategy == 'MAX':
            pooled_output = self.max_pooling(hidden_states)
        else:
            pooled_output = self.cls_pooling(hidden_states)

        pooled_output = self.dense(pooled_output) # shape
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BERTEncoder(nn.Module):
    def __init__(self,bert:BERT,config:BERTConfig,is_feature_based = True):
        super(BERTEncoder,self).__init__()
        self.encoder= bert
        self.is_feature_based =is_feature_based
        self.pooler = BERTPooler(config)
        
       
    def forward(self,input_ids,segment_info,output_all_encoded_layers=False,pooling_strategy="MAX"):

        encoded_layers = self.encoder(input_ids,segment_info,output_all_encoded_layers) 
        if output_all_encoded_layers:
            sequence_output = encoded_layers[-1] # encoded_layers is all hidden layers
        else :
            sequence_output = encoded_layers # encoded_layers is the last hidden layer

        pooled_output = self.pooler(sequence_output,pooling_strategy)
    
        # return encoded_layers,pooled_output # As to the limited memeroy ,encoded_layers can not be returned.
        return pooled_output,sequence_output
