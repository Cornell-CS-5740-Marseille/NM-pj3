import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, method='no'):
        super(model, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        if self.method == 'no':
            self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.decoder = nn.LSTM(2*hidden_dim, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)

        # additive attention
        self.v = torch.rand(1, hidden_dim)
        self.w1 = torch.rand(hidden_dim, hidden_dim)
        self.w2 = torch.rand(hidden_dim, hidden_dim)

        # multiplicative attention
        self.w_mul = torch.rand(hidden_dim, hidden_dim)

    def attention_additive(self, hidden, outputs_encoder):
        aj = torch.zeros(outputs_encoder.data.shape[0])
        for i in range(outputs_encoder.data.shape[0]):
            out_temp = outputs_encoder[i]
            temp1 = hidden.squeeze(1).mm(self.w1)
            temp2 = out_temp.mm(self.w2)
            temp3 = temp1 + temp2
            temp3 = torch.nn.functional.relu(temp3)
            aj[i] = temp3.mm(self.v.transpose(0, 1))

        ret = torch.nn.functional.softmax(aj)
        return ret

    def attention_multiplicative(self, hidden, outputs_encoder):
        aj = torch.zeros(outputs_encoder.data.shape[0])
        for i in range(outputs_encoder.data.shape[0]):
            out_temp = outputs_encoder[i]
            temp1 = hidden.squeeze(1).mm(self.w_mul)
            aj[i] = temp1.mm(out_temp.transpose(0, 1))

        ret = torch.nn.functional.softmax(aj)
        return ret

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def saveModels(self):
        torch.save(self.state_dict(), 'my_trained_model_weights')

    def forward(self, input_seq, gold_seq=None):
        input_vectors = self.embeds(torch.tensor(input_seq))
        input_vectors = input_vectors.unsqueeze(1)
        outputs, hidden = self.encoder(input_vectors)
        save_outputs = outputs
        # Technique used to train RNNs: 
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        teacher_force = False

        # This condition tells us whether we are in training or inference phase 
        if gold_seq is not None and teacher_force:
            gold_vectors = torch.cat([torch.zeros(1, self.hidden_dim), self.embeds(torch.tensor(gold_seq[:-1]))], 0)
            gold_vectors = gold_vectors.unsqueeze(1)
            gold_vectors = torch.nn.functional.relu(gold_vectors)
            outputs, hidden = self.decoder(gold_vectors, hidden)

            predictions = self.out(outputs)
            predictions = predictions.squeeze()
            vals, idxs = torch.max(predictions, 1)
            return predictions, list(np.array(idxs))
        else:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)
                if self.method == 'add':
                    attn = self.attention_additive(hidden[0], save_outputs)
                    context = attn.unsqueeze(0).mm(save_outputs.squeeze())
                    prev = torch.cat((prev, context.unsqueeze(1)), 2)
                elif self.method == 'mul':
                    attn = self.attention_multiplicative(hidden[0], save_outputs)
                    context = attn.unsqueeze(0).mm(save_outputs.squeeze())
                    prev = torch.cat((prev, context.unsqueeze(1)), 2)

                outputs, hidden = self.decoder(prev, hidden)

                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor([idx]))
                prev = prev.unsqueeze(1)
            return torch.stack(predictions), predicted_seq