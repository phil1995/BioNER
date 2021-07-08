import torch
from fasttext.FastText import _FastText
from torch import optim, nn
from torch.autograd.grad_mode import F
from torch.utils import data

import MedMentionsDataset

from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _download_extract_validate,
    _create_dataset_directory,
    _create_data_from_iob,
)
import itertools
import fasttext

from model.BiLSTM import BiLSTM
from model.DatasetTransformer import DatasetTransformer
from model.MedMentionsDataLoader import MedMentionsDataLoader


def load_dataset(encoder) -> MedMentionsDataset:
    # encoder = fasttext.load_model(
    #    "/Users/phil/Documents/Universität/6.Semester/Bachelorarbeit/pubmed.fasttext.3-3ngrams.neg5.1e-5_subs.bin")
    structured_dataset = MedMentionsDataset.MedMentionsStructuredDataset(
        "/Users/phil/Documents/Universität/6.Semester/Bachelorarbeit/MedMentions/full/data/test_CoNLL.txt",
        encoder=encoder)
    return MedMentionsDataset.MedMentionsDataset(structured_dataset)


def create_model(encoder: _FastText, feedforward_layer_size: int, lstm_layer_size: int) -> BiLSTM:
    """

    :param encoder:
    :param feedforward_layer_size: (DATEXIS: 512)
    :param lstm_layer_size: (DATEXIS: 256)
    :return:
    """
    embeddings_size = encoder.get_dimension()
    return BiLSTM(input_vector_size=embeddings_size, feedforward_layer_size=512, lstm_layer_size=256)


def train_model(model, dataset, learning_rate, batch_size, epochs, loss=nn.CrossEntropyLoss()):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train linear model using Adam on mini-batches.
    for epoch in range(epochs):
        # DataLoader generates random batches from a given dataset.
        data_loader = MedMentionsDataLoader(dataset=dataset, shuffle=False, num_workers=1,
                                            batch_size=1, collate_fn=collate_batch)
        # We want to report the training loss after each epoch
        epoch_loss = 0.0

        # for batch in data_loader:
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            # After each iteration of the training step, reset the local gradients stored in the network to zero.
            model.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += batch_loss.item()

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss}')


def collate_batch(batch):
    embeddings_list, label_list = [], []
    for (_embedding, _label) in batch:
        embeddings_list.append(_embedding)
        label_list.append(_label)
    embeddings_list = torch.tensor(embeddings_list, dtype=torch.float)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return embeddings_list.to(device), label_list.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    encoder = fasttext.load_model(
        "/Users/phil/Documents/Universität/6.Semester/Bachelorarbeit/pubmed.fasttext.3-3ngrams.neg5.1e-5_subs.bin")
    dataset = load_dataset(encoder=encoder)
    model = create_model(encoder=encoder, feedforward_layer_size=512, lstm_layer_size=256)
    train_model(model, dataset, learning_rate=0.001, batch_size=1, epochs=2)
