import json
import math
import os

import random
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embeddings):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        embeddings,
        dropout_p,
        max_length,
    ):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding.from_pretrained(embeddings) #for paragen
        #self.embedding = nn.Embedding(len(embeddings), 300) #for NMT with tamil, trying wiht senitment too
        self.attn = nn.Linear(self.input_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(
            self.input_size + self.hidden_size, self.hidden_size
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, fixations):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )

        attn_weights = attn_weights * torch.nn.ConstantPad1d((0, attn_weights.shape[-1] - fixations.shape[-2]), 0)(fixations.squeeze().unsqueeze(0))

        # attn_weights = torch.softmax(attn_weights * torch.nn.ConstantPad1d((0, attn_weights.shape[-1] - fixations.shape[-2]), 0)(fixations.squeeze().unsqueeze(0)), dim=1)
        
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # output = F.log_softmax(self.out(output[0]), dim=1)
        output = self.out(output[0])
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights
