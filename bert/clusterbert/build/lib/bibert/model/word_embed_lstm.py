'''
Descripttion: 
version: 
Author: Yang jin
Date: 2021-09-15 13:07:14
LastEditors: Yang jin
LastEditTime: 2021-10-18 08:35:41
'''

from torch import nn
import torch
import math

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        pass
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
        
# class SelfAttention(nn.Module):
#     def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.2,attention_probs_dropout_prob=0.2):
#         super(SelfAttention, self).__init__()
#         if hidden_size % num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (hidden_size, num_attention_heads))
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = hidden_size

#         self.query = nn.Linear(input_size, self.all_head_size)
#         self.key = nn.Linear(input_size, self.all_head_size)
#         self.value = nn.Linear(input_size, self.all_head_size)

#         self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

#         # 做完self-attention 做一个前馈全连接 LayerNorm 输出
#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
#         self.out_dropout = nn.Dropout(hidden_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, input_tensor):
#         mixed_query_layer = self.query(input_tensor)
#         mixed_key_layer = self.key(input_tensor)
#         mixed_value_layer = self.value(input_tensor)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # [batch_size heads seq_len seq_len] scores
#         # [batch_size 1 1 seq_len]

#         # attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         # Fixme
#         attention_probs = self.attn_dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         hidden_states = self.dense(context_layer)
#         hidden_states = self.out_dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states



# class BERTLSTM(nn.Module):
#     """Supervised fine-tuning with labeled sequences.
#     """

#     def __init__(self, bert_encoder:BERTWordEncoder, clConfig: ClassifierConfig, num_labels: int):
#         super(BERTLSTM, self).__init__()
#         clConfig, num_labels = clConfig, num_labels
#         self.bert_encoder = bert_encoder
#         self.dropout = nn.Dropout(clConfig.dropout)

#         self.classifier = nn.ModuleList()

#         # in_dim = clConfig.hidden

#         self.lstm = nn.LSTM(
#             clConfig.hidden, clConfig.layers_hidden[0], clConfig.cdepth, dropout=clConfig.dropout, bidirectional=True)

#         self.att_hidden_size = clConfig.layers_hidden[0]
#         self.att_heads = 8 # hidden_size 的整除数
#         self.units = 2000

#         self.attention = SelfAttention(self.att_heads,clConfig.layers_hidden[0]*2,self.att_hidden_size*2)

#         self.att_out_size = self.att_hidden_size*clConfig.seq_len*2
        
#         self.linear1 = nn.Linear(self.att_out_size,self.units)

#         self.linear2 = nn.Linear(self.units,self.units)

#         self.relu = nn.ReLU()

#         self.linear3 = nn.Linear(self.units, num_labels)

#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, input_ids, token_type_ids=None, output_all_encoded_layers=False):
#         seq_output = self.bert_encoder(
#             input_ids, token_type_ids, output_all_encoded_layers)  # shape(batchsize * seq_len * hidden)
        
#         # output = seq_output.unsqueeze(1) #shape (batchsize,1,seq_len,hidden)
#         output = self.lstm(seq_output)[0]
        
#         # print (output.shape)
#         output = self.attention(output)
#         # print (output.shape)

#         output = torch.flatten(output,start_dim=1)
#         # print(output.shape)

#         output = self.relu(self.linear1(output))

#         output = self.relu(self.linear2(output))

#         # print(output.shape)
#         output = self.linear3(output)
#         # print(output.shape)

#         output_labels = self.softmax(output)
#         # print(output.shape)


#         return  output_labels
