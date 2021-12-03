import math
from torch.nn import Linear, LSTM, Module, init, ModuleList, Dropout
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BioNER(Module):
    """
    The BioNER model as proposed in BioNER: Named Entity Recognition in the Biomedical Domain

    It has a feed-forward layer of size 2048, a stack of three BiLSTM layers each of size 1024 and a LSTM decoder layer
    also of the size 1024. Before each (Bi-)LSTM layer is an dropout layer with the dropout probability of 80%.
    BioNER predicts the most likely label for each token via softmax.
    """
    def __init__(self,
                 input_vector_size: int,
                 feedforward_layer_size: int = 2048,
                 lstm_layer_size: int = 1024,
                 out_features: int = 3,
                 dropout_probability: float = 0.8):
        super().__init__()
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
        """
        Initialize the weights in the same way as Arnold et al. initialized the weights for the (modified) version of
        DATEXIS-NER in: https://github.com/sebastianarnold/TeXoo/blob/514860d96decdf3ff6613dfcf0d27d9845ddcf60/texoo
        -entity-recognition/src/main/java/de/datexis/ner/tagger/MentionTagger.java#L86-L136 :return:
        """

        BioNER.relu(self.ff1)
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
        bi_lstm_out = self.dropout_pack_padded_sequence(sequence=bi_lstm_out,
                                                        lengths=lengths)
        for i, additional_biLSTM_layer in enumerate(self.additional_biLSTM_layers):
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
