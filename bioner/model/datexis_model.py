import math

from torch import Tensor
from torch.nn import Linear, LSTM, Module, init, ModuleList, Dropout, BatchNorm1d
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DATEXISModel(Module):

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
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bi_lstm_out, (h, c) = self.biLSTM(x)
        lstm_out, (h, c) = self.encoderLSTM(bi_lstm_out)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.hidden2tag(lstm_out)
        # Permute the tag space as CrossEntropyLoss expects an output shape of: [batch_size, nb_classes, seq_length]
        # see: https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291
        permuted_tag_space = tag_space.permute(0, 2, 1)
        return permuted_tag_space


class StackedBiLSTMModel(Module):
    def __init__(self,
                 input_vector_size: int,
                 additional_bilstm_layer: int = 1,
                 feedforward_layer_size: int = 150,
                 lstm_layer_size: int = 20,
                 out_features: int = 3,
                 dropout_probability: float = 0,
                 batch_normalization_enabled: bool = False):
        super().__init__()

        print(
            f"Initialize StackedBiLSTMModel Network: Input Vector Size:{input_vector_size} "
            f"Feedforward Layer Size:{feedforward_layer_size} "
            f"LSTM Layer Size:{lstm_layer_size} "
            f"Out Feature Size:{out_features} "
            f"# Stacked BiLSTMs:{additional_bilstm_layer} "
            f"Dropout prob.:{dropout_probability}")
        if batch_normalization_enabled:
            print("Batch normalization enabled")
        assert additional_bilstm_layer >= 0
        assert dropout_probability >= 0 <= 1.0
        self.batch_normalization_enabled = batch_normalization_enabled
        self.dropout = Dropout(p=dropout_probability)
        self.ff1 = Linear(in_features=input_vector_size, out_features=feedforward_layer_size)
        self.biLSTM = LSTM(input_size=feedforward_layer_size,
                           bidirectional=True, hidden_size=lstm_layer_size, batch_first=True)
        self.additional_biLSTM_layers = ModuleList([LSTM(input_size=lstm_layer_size * 2,
                                                         bidirectional=True,
                                                         hidden_size=lstm_layer_size,
                                                         batch_first=True)
                                                    for i in range(additional_bilstm_layer)])
        self.encoderLSTM = LSTM(input_size=lstm_layer_size * 2, hidden_size=lstm_layer_size, batch_first=True)
        self.hidden2tag = Linear(in_features=lstm_layer_size, out_features=out_features)
        if batch_normalization_enabled:
            self.ffBatchNorm = BatchNorm1d(feedforward_layer_size)
            self.encoderLSTMBatchNorm = BatchNorm1d(lstm_layer_size)
            self.biLSTMBatchNorms = ModuleList([BatchNorm1d(lstm_layer_size * 2) for i in range(additional_bilstm_layer + 1)])
        self.init_weights()

    def init_weights(self):
        DATEXISModel.relu(self.ff1)
        self.xavier_normal(self.biLSTM)
        for additional_biLSTM_layer in self.additional_biLSTM_layers:
            self.xavier_normal(additional_biLSTM_layer)
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
        if self.batch_normalization_enabled:
            x = x.permute(0, 2, 1)
            x = self.ffBatchNorm(x)
            x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bi_lstm_out, (h, c) = self.biLSTM(x)
        if self.batch_normalization_enabled:
            bi_lstm_out = StackedBiLSTMModel.apply_batch_norm_pack_padded_sequence(sequence=bi_lstm_out,
                                                                                   batch_norm=self.biLSTMBatchNorms[0],
                                                                                   lengths=lengths.cpu())
        bi_lstm_out = self.dropout_pack_padded_sequence(sequence=bi_lstm_out,
                                                        lengths=lengths)
        for i, additional_biLSTM_layer in enumerate(self.additional_biLSTM_layers):
            bi_lstm_out, (h, c) = additional_biLSTM_layer(bi_lstm_out)
            if self.batch_normalization_enabled:
                bi_lstm_out = StackedBiLSTMModel.apply_batch_norm_pack_padded_sequence(sequence=bi_lstm_out,
                                                                                       batch_norm=self.biLSTMBatchNorms[
                                                                                           i+1],
                                                                                       lengths=lengths.cpu())
            bi_lstm_out = self.dropout_pack_padded_sequence(sequence=bi_lstm_out,
                                                            lengths=lengths)
        lstm_out, (h, c) = self.encoderLSTM(bi_lstm_out)
        if self.batch_normalization_enabled:
            lstm_out = StackedBiLSTMModel.apply_batch_norm_pack_padded_sequence(sequence=lstm_out,
                                                                                batch_norm=self.encoderLSTMBatchNorm,
                                                                                lengths=lengths.cpu())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.hidden2tag(lstm_out)
        # Permute the tag space as CrossEntropyLoss expects an output shape of: [batch_size, nb_classes, seq_length]
        # see: https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291
        permuted_tag_space = tag_space.permute(0, 2, 1)
        return permuted_tag_space

    def dropout_pack_padded_sequence(self, sequence, lengths):
        """
        Applies dropout to a pack_padded_sequence.
        As it's not possible to directly use a pack_padded_sequence as input for a dropout layer, we need to transform
        it first into a pad_packed_sequence and afterwards re-transform it again into a pack_padded_sequence.
        :param sequence: the pack_padded_sequence
        :param lengths: the lengths passed to the pack_padded_sequence
        :return:
        """
        x, _ = pad_packed_sequence(sequence, batch_first=True)
        x = self.dropout(x)
        return pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

    @staticmethod
    def apply_batch_norm_pack_padded_sequence(sequence, batch_norm, lengths):
        x, _ = pad_packed_sequence(sequence, batch_first=True)
        x = x.permute(0, 2, 1)
        x = batch_norm(x)
        x = x.permute(0, 2, 1)
        return pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)


class BioNER(Module):
    def __init__(self,
                 input_vector_size: int,
                 feedforward_layer_size: int = 2048,
                 lstm_layer_size: int = 1024,
                 out_features: int = 3,
                 dropout_probability: float = 0.9,
                 skip_connection_enabled: bool = False):
        super().__init__()
        self.skip_connection_enabled = skip_connection_enabled
        if skip_connection_enabled:
            print("Skip Connection enabled")
        self.dropout = Dropout(p=dropout_probability)
        self.ff1 = Linear(in_features=input_vector_size, out_features=feedforward_layer_size)
        self.biLSTM = LSTM(input_size=feedforward_layer_size,
                            bidirectional=True, hidden_size=lstm_layer_size, batch_first=True)
        self.additional_biLSTM_layers = ModuleList([LSTM(input_size=lstm_layer_size * 2,
                                                         bidirectional=True,
                                                         hidden_size=lstm_layer_size,
                                                         batch_first=True)
                                                    for i in range(2)])
        self.encoderLSTM = LSTM(input_size=lstm_layer_size * 2, hidden_size=lstm_layer_size, batch_first=True)
        self.hidden2tag = Linear(in_features=lstm_layer_size, out_features=out_features)
        self.init_weights()

    def init_weights(self):
        DATEXISModel.relu(self.ff1)
        self.xavier_normal(self.biLSTM)
        for additional_biLSTM_layer in self.additional_biLSTM_layers:
            self.xavier_normal(additional_biLSTM_layer)
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
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bi_lstm_out, (h, c) = self.biLSTM(x)
        identity = bi_lstm_out
        bi_lstm_out = self.dropout_pack_padded_sequence(sequence=bi_lstm_out,
                                                        lengths=lengths)
        for i, additional_biLSTM_layer in enumerate(self.additional_biLSTM_layers):
            if self.skip_connection_enabled and i == 1:
                bi_lstm_out += identity
            bi_lstm_out, (h, c) = additional_biLSTM_layer(bi_lstm_out)
            bi_lstm_out = self.dropout_pack_padded_sequence(sequence=bi_lstm_out,
                                                            lengths=lengths)
        lstm_out, (h, c) = self.encoderLSTM(bi_lstm_out)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.hidden2tag(lstm_out)
        # Permute the tag space as CrossEntropyLoss expects an output shape of: [batch_size, nb_classes, seq_length]
        # see: https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291
        permuted_tag_space = tag_space.permute(0, 2, 1)
        return permuted_tag_space

    def dropout_pack_padded_sequence(self, sequence, lengths):
        """
        Applies dropout to a pack_padded_sequence.
        As it's not possible to directly use a pack_padded_sequence as input for a dropout layer, we need to transform
        it first into a pad_packed_sequence and afterwards re-transform it again into a pack_padded_sequence.
        :param sequence: the pack_padded_sequence
        :param lengths: the lengths passed to the pack_padded_sequence
        :return:
        """
        x, _ = pad_packed_sequence(sequence, batch_first=True)
        x = self.dropout(x)
        return pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
