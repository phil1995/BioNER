import math

from torch import Tensor
from torch.nn import Linear, LSTM, Module, init
import torch.nn.functional as F


class BiLSTM(Module):

    # de.datexis.ner.tagger.MentionTagger initializing BLSTM network 6075:512:512:256:3
    def __init__(self,
                 input_vector_size: int,
                 feedforward_layer_size: int,
                 lstm_layer_size: int,
                 out_features: int = 3):
        super().__init__()

        print(
            f"Initialize BiLSTM Network: Input Vector Size:{input_vector_size} Feedforward Layer Size: {feedforward_layer_size} LSTM Layer Size: {lstm_layer_size} Out Feature Size: {out_features}")
        self.ff1 = Linear(in_features=input_vector_size, out_features=feedforward_layer_size)
        self.ff2 = Linear(in_features=feedforward_layer_size, out_features=feedforward_layer_size)
        self.biLSTM = LSTM(input_size=feedforward_layer_size,
                           bidirectional=True, hidden_size=lstm_layer_size, batch_first=True)
        self.encoderLSTM = LSTM(input_size=lstm_layer_size * 2, hidden_size=lstm_layer_size, batch_first=True)
        self.hidden2tag = Linear(in_features=lstm_layer_size, out_features=out_features)
        self.init_weights()

    def init_weights(self):
        BiLSTM.relu(self.ff1)
        BiLSTM.relu(self.ff2)
        self.xavier_normal(self.biLSTM)
        self.xavier_normal(self.encoderLSTM)
        init.xavier_normal_(self.hidden2tag.weight)

    @staticmethod
    def xavier_normal(rnn):
        for name, param in rnn.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)

    @staticmethod
    def relu(tensor: Linear) -> Tensor:
        init.normal_(tensor.weight, std=math.sqrt(2.0 / tensor.in_features))

    def forward(self, x):
        sentence_length = len(x)
        # TODO: maybe add here the fasttext encoding?
        x = self.ff1(x)
        x = F.relu(x)
        x = self.ff2(x)
        x = F.relu(x)
        bi_lstm_out, (h, c) = self.biLSTM(x)
        lstm_out, (h, c) = self.encoderLSTM(bi_lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=1)
        permuted_tag_scores = tag_scores.permute(0, 2, 1)
        return permuted_tag_scores
