import math

from torch import Tensor
from torch.nn import Linear, LSTM, Module, init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DATEXISModel(Module):

    # de.datexis.ner.tagger.MentionTagger initializing BLSTM network 6075:512:512:256:3
    def __init__(self,
                 input_vector_size: int,
                 feedforward_layer_size: int = 150,
                 lstm_layer_size: int = 20,
                 out_features: int = 3):
        super().__init__()

        print(
            f"Initialize DATEXIS-NER Network: Input Vector Size:{input_vector_size} "
            f"Feedforward Layer Size:{feedforward_layer_size} "
            f"LSTM Layer Size:{lstm_layer_size} "
            f"Out Feature Size:{out_features}")
        self.ff1 = Linear(in_features=input_vector_size, out_features=feedforward_layer_size)
        self.biLSTM = LSTM(input_size=feedforward_layer_size,
                           bidirectional=True, hidden_size=lstm_layer_size, batch_first=True)
        self.encoderLSTM = LSTM(input_size=lstm_layer_size * 2, hidden_size=lstm_layer_size, batch_first=True)
        self.hidden2tag = Linear(in_features=lstm_layer_size, out_features=out_features)
        self.init_weights()

    def init_weights(self):
        DATEXISModel.relu(self.ff1)
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

    def forward(self, x, lengths):
        x = self.ff1(x)
        x = F.relu(x)
        x = F.relu(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bi_lstm_out, (h, c) = self.biLSTM(x)
        lstm_out, (h, c) = self.encoderLSTM(bi_lstm_out)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.hidden2tag(lstm_out)
        # Permute the tag space as CrossEntropyLoss expects an output shape of: [batch_size, nb_classes, seq_length]
        # see: https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291
        permuted_tag_space = tag_space.permute(0, 2, 1)
        return permuted_tag_space
