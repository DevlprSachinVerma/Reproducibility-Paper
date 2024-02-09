from collections import OrderedDict
import logging
import sys
import math



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

#self attention
class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        
    def forward(self, Q, K, V):
        # Q: float32:[batch_size, n_queries, d_k]
        # K: float32:[batch_size, n_keys, d_k]
        # V: float32:[batch_size, n_keys, d_v]
        dk = K.shape[-1]
        dv = V.shape[-1]
        KT = torch.transpose(K, -1, -2)
        weight_logits = torch.bmm(Q, KT) / math.sqrt(dk)
        # weight_logits: float32[batch_size, n_queries, n_keys]
        weights = F.softmax(weight_logits, dim=-1)
        # weight: float32[batch_size, n_queries, n_keys]
        return torch.bmm(weights, V)
        # return float32[batch_size, n_queries, dv]
        

class MultiHeadedSelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadedSelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        print('{} {}'.format(d_model, n_heads))
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.d_v = self.d_k
        self.attention_layer = AttentionLayer()
        self.W_Qs = nn.ModuleList([
                nn.Linear(d_model, self.d_k, bias=False)
                for _ in range(n_heads)
        ])
        self.W_Ks = nn.ModuleList([
                nn.Linear(d_model, self.d_k, bias=False)
                for _ in range(n_heads)
        ])
        self.W_Vs = nn.ModuleList([
                nn.Linear(d_model, self.d_v, bias=False)
                for _ in range(n_heads)
        ])
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        # x:float32[batch_size, sequence_length, self.d_model]
        head_outputs = []
        for W_Q, W_K, W_V in zip(self.W_Qs, self.W_Ks, self.W_Vs):
            Q = W_Q(x)
            # Q float32:[batch_size, sequence_length, self.d_k]
            K = W_K(x)
            # Q float32:[batch_size, sequence_length, self.d_k]
            V = W_V(x)
            # Q float32:[batch_size, sequence_length, self.d_v]
            head_output = self.attention_layer(Q, K, V)
            # float32:[batch_size, sequence_length, self.d_v]
            head_outputs.append(head_output)
        concatenated = torch.cat(head_outputs, dim=-1)
        # concatenated float32:[batch_size, sequence_length, self.d_model]
        out = self.W_O(concatenated)
        # out float32:[batch_size, sequence_length, self.d_model]
        return out

class Feedforward(nn.Module):
    def __init__(self, d_model):
        super(Feedforward, self).__init__()
        self.d_model = d_model
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: float32[batch_size, sequence_length, d_model]
        return self.W2(torch.relu(self.W1(x)))

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_layers = nn.ModuleList([
            MultiHeadedSelfAttentionLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.ffs = nn.ModuleList([
            Feedforward(d_model)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        # x: float32[batch_size, sequence_length, self.d_model]
        for attention_layer, ff in zip(self.attention_layers, self.ffs):
            attention_out = attention_layer(x)
            # attention_out: float32[batch_size, sequence_length, self.d_model]
            x = F.layer_norm(x + attention_out, x.shape[2:])
            ff_out = ff(x)
            # ff_out: float32[batch_size, sequence_length, self.d_model]
            x = F.layer_norm(x + ff_out, x.shape[2:])
        return x
    
    
#main
def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb


def neg_log_likelihood_loss(outputs, batch_label, batch_size, seq_len):
    outputs = outputs.view(batch_size * seq_len, -1)
    score = F.log_softmax(outputs, 1)

    loss = nn.NLLLoss(ignore_index=0, size_average=False)(
        score, batch_label.view(batch_size * seq_len)
    )
    loss = loss / batch_size
    _, tag_seq = torch.max(score, 1)
    tag_seq = tag_seq.view(batch_size, seq_len)

    # print(score[0], tag_seq[0])

    return loss, tag_seq


def mse_loss(outputs, batch_label, batch_size, seq_len, word_seq_length):
    # score = torch.nn.functional.softmax(outputs, 1)
    score = torch.sigmoid(outputs)

    mask = torch.zeros_like(score)
    for i, v in enumerate(word_seq_length):
        mask[i, 0:v] = 1

    score = score * mask

    loss = nn.MSELoss(reduction="sum")(
        score.view(batch_size, seq_len), batch_label.view(batch_size, seq_len)
    )

    loss = loss / batch_size

    return loss, score.view(batch_size, seq_len)


class Network(nn.Module):
    def __init__(
        self,
        embedding_type,
        vocab_size,
        embedding_dim,
        dropout,
        hidden_dim,
        embeddings=None,
        attention=True,
    ):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}")
        prelayers = OrderedDict()
        postlayers = OrderedDict()

        if embedding_type in ("w2v", "glove"):
            if embeddings is not None:
                prelayers["embedding_layer"] = nn.Embedding.from_pretrained(embeddings)
            else:
                prelayers["embedding_layer"] = nn.Embedding(vocab_size, embedding_dim)
            prelayers["embedding_dropout_layer"] = nn.Dropout(dropout)
            embedding_dim = 300
        elif embedding_type == "bert":
            embedding_dim = 768

    

        self.lstm = BiLSTM(embedding_dim, hidden_dim // 2, num_layers=1)
        postlayers["lstm_dropout_layer"] = nn.Dropout(dropout)

        if attention:
            # increased compl with 1024D, and 16,16: for no att and att experiments
            # before: for the initial att and pretraining: heads 4 and layers 4, 128D
            # then was 128 D with heads 4 layer 1 = results for all IUI
            # postlayers["position_encodings"] = PositionalEncoding(hidden_dim)
            postlayers["attention_layer"] = Transformer(
                d_model=hidden_dim, n_heads=4, n_layers=1
            )

        postlayers["ff_layer"] = nn.Linear(hidden_dim, hidden_dim // 2)
        postlayers["ff_activation"] = nn.ReLU()
        postlayers["output_layer"] = nn.Linear(hidden_dim // 2, 1)

        self.logger.info(f"prelayers: {prelayers.keys()}")
        self.logger.info(f"postlayers: {postlayers.keys()}")

        self.pre = nn.Sequential(prelayers)
        self.post = nn.Sequential(postlayers)

    def forward(self, x, word_seq_length):
        x = self.pre(x)
        x = self.lstm(x, word_seq_length)
        #MS pritning fix model params
        #for p in self.parameters():
        #    print(p.data)
        #    break

        return self.post(x.transpose(1, 0))


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden, num_layers):
        super().__init__()
        self.net = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, word_seq_length):
        packed_words = pack_padded_sequence(x, word_seq_length, True, False)
        lstm_out, hidden = self.net(packed_words)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        return lstm_out