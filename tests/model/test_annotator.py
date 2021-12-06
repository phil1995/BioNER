import math
import os.path
import random

import numpy as np
import torch
from torch import optim, tensor
from torch.nn import Module, Linear
import torch.nn.functional as F

from bioner.model.annotator import Annotator, TrainingParameters
from bioner.model.bio2tag import BIO2Tag
from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.encoded_token import EncodedToken
from bioner.model.encoder.encoder import Encoder


def test_train(tmpdir):
    _test_train(tmpdir, faster_training_evaluation=False)


def test_train_with_faster_training_evaluation(tmpdir):
    _test_train(tmpdir, faster_training_evaluation=True)


def _test_train(tmpdir: str, faster_training_evaluation: bool):
    seed = 1234
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    mock_encoder = EncoderMock()
    training_dataset_file_path = os.path.join(tmpdir, "training.txt")
    validation_dataset_file_path = os.path.join(tmpdir, "validation.txt")
    model_save_path = os.path.join(tmpdir, "model_directory")
    tensorboard_log_directory = os.path.join(tmpdir, "tensorboard")
    training_log_file_path = os.path.join(tmpdir, "training.log")
    create_datasets(training_dataset_file_path, validation_dataset_file_path)
    mock_model = ModuleMock()
    training_parameters = TrainingParameters(encoder=mock_encoder,
                                             training_dataset_path=training_dataset_file_path,
                                             validation_dataset_path=validation_dataset_file_path,
                                             model=mock_model,
                                             max_epochs=12,
                                             num_workers=0,
                                             model_save_path=model_save_path,
                                             optimizer=optim.Adam(mock_model.parameters(), lr=0.0005),
                                             tensorboard_log_directory_path=tensorboard_log_directory,
                                             training_log_file_path=training_log_file_path,
                                             batch_size=2,
                                             faster_training_evaluation=faster_training_evaluation
                                             )
    state = Annotator.train(training_parameters)
    assert math.isclose(0.8, state.metrics["F1"], rel_tol=1e-5)
    assert math.isclose(0.8, state.metrics["Precision"], rel_tol=1e-5)
    assert math.isclose(0.8, state.metrics["Recall"], rel_tol=1e-5)
    assert os.path.isfile(training_log_file_path)
    assert len(os.listdir(tensorboard_log_directory)) != 0
    assert len(os.listdir(model_save_path)) == 1
    assert os.path.isfile(os.path.join(model_save_path, "best_model_1_val_f1=0.8000.pt"))
    assert len(mock_encoder.encode_dataset_passed_values) == 2

    first_training_sentence_passed_values = (tensor([[[0., 1., 1., 1.],
                                                      [1., 1., 1., 1.],
                                                      [2., 1., 1., 1.]],

                                                     [[0., 1., 1., 1.],
                                                      [2., 1., 1., 1.],
                                                      [2., 1., 1., 1.]]]),
                                             tensor([3, 3]))
    second_training_sentence_passed_values = (tensor([[[0., 1., 1., 1.],
                                                       [2., 1., 1., 1.],
                                                       [2., 1., 1., 1.]],

                                                      [[2., 1., 1., 1.],
                                                       [2., 1., 1., 1.],
                                                       [2., 1., 1., 1.]]]),
                                              tensor([3, 3]))
    first_validation_sentence = (tensor([[[0., 1., 1., 1.],
                                          [1., 1., 1., 1.],
                                          [0., 1., 1., 1.]],

                                         [[0., 1., 1., 1.],
                                          [2., 1., 1., 1.],
                                          [2., 1., 1., 1.]]]),
                                 tensor([3, 3]))
    second_validation_sentence = (tensor([[[0., 1., 1., 1.],
                                           [2., 1., 1., 1.],
                                           [0., 1., 1., 1.]],

                                          [[2., 1., 1., 1.],
                                           [2., 1., 1., 1.],
                                           [2., 1., 1., 1.]]]),
                                  tensor([3, 3]))
    expected_values_passed_to_forward = []
    for epoch in range(1, 12):
        expected_values_passed_to_forward.append(first_training_sentence_passed_values)
        expected_values_passed_to_forward.append(second_training_sentence_passed_values)
        if not faster_training_evaluation or epoch == 10:
            expected_values_passed_to_forward.append(first_training_sentence_passed_values)
            expected_values_passed_to_forward.append(second_training_sentence_passed_values)

        expected_values_passed_to_forward.append(second_validation_sentence)
        expected_values_passed_to_forward.append(first_validation_sentence)
    assert len(mock_model.forward_passed_values) == len(expected_values_passed_to_forward)
    for i, expected_passed_value in enumerate(expected_values_passed_to_forward):
        expected_x, expected_length = expected_passed_value
        actual_x, actual_length = mock_model.forward_passed_values[i]
        assert torch.equal(expected_x, actual_x)
        assert torch.equal(expected_length, actual_length)


def test_annotate_dataset(tmpdir):
    test_dataset_file_path = os.path.join(tmpdir, "test.txt")
    test_dataset_content = create_train_document_content()
    create_dataset(test_dataset_content, test_dataset_file_path)
    dataset = Annotator.load_dataset(path=test_dataset_file_path, encoder=EncoderMock())
    model_mock = AnnotateDatasetModuleMock()
    annotated_dataset = Annotator.annotate_dataset(dataset, model_mock)
    expected_tags = [[BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.OUTSIDE],
                     [BIO2Tag.OUTSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN],
                     [BIO2Tag.OUTSIDE, BIO2Tag.BEGIN, BIO2Tag.INSIDE],
                     [BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.BEGIN]]
    expected_text = [["Lorem", "ipsum", "dolor"],
                     ["Eirmod", "tempor", "."],
                     ["ut", "labore", "et"],
                     ["dolore", "magna", "aliquyam"]]
    actual_tags = []
    actual_text = []
    for document in annotated_dataset.documents:
        for sentence in document.sentences:
            sentence_tags = []
            sentence_text = []
            for token in sentence.tokens:
                sentence_tags.append(token.tag)
                sentence_text.append(token.text)
            actual_tags.append(sentence_tags)
            actual_text.append(sentence_text)
    assert actual_tags == expected_tags
    assert actual_text == expected_text

    actual_flattened_tags = []
    actual_flattened_text = []
    for sentence in annotated_dataset.sentences:
        sentence_tags = []
        sentence_text = []
        for token in sentence:
            sentence_tags.append(token.tag)
            sentence_text.append(token.text)
        actual_flattened_tags.append(sentence_tags)
        actual_flattened_text.append(sentence_text)
    assert actual_flattened_tags == expected_tags
    assert actual_flattened_text == expected_text


def create_datasets(training_dataset_file_path: str, validation_dataset_file_path: str):
    training_dataset_content = create_train_document_content()
    create_dataset(training_dataset_content, training_dataset_file_path)

    validation_dataset_content = create_validation_document_content()
    create_dataset(validation_dataset_content, validation_dataset_file_path)


def create_dataset(content, path):
    with open(path, "w") as text_file:
        text_file.write(content)


def create_train_document_content() -> str:
    return """-DOCSTART-	0	0	O

Lorem	0	5	B
ipsum	6	10	I
dolor	11	16	O

Eirmod	0	5	B
tempor	6	8	O
.	9	10	O

-DOCSTART-	0	0	O

ut	0	5	B
labore	6	8	O
et	9	10	O

dolore	0	5	O
magna	6	8	O
aliquyam	9	10	O

"""


def create_validation_document_content() -> str:
    return """-DOCSTART-	0	0	O

Foo	0	3	B
Bar	4	7	I
Baz	8	11	B

Eirmod	0	5	B
tempor	6	8	O
.	9	10	O

-DOCSTART-	0	0	O

ut	0	5	B
labore	6	8	O
et	9	10	B

dolore	0	5	O
magna	6	8	O
aliquyam	9	10	O

"""


class EncoderMock(Encoder):

    def __init__(self):
        self.encode_dataset_passed_values = []

    def encode(self, dataset: CoNLLDataset):
        self.encode_dataset_passed_values.append(dataset)
        for document in dataset.documents:
            for sentence in document.sentences:
                encoded_tokens = [EncodedToken(encoding=np.array([BIO2Tag.get_index(token.tag), 1, 1, 1]),
                                               text=token.text,
                                               start=token.start,
                                               end=token.end,
                                               tag=token.tag) for token in
                                  sentence.tokens]
                sentence.tokens = encoded_tokens

    def get_embeddings_vector_size(self) -> int:
        return 4


class ModuleMock(Module):

    def __init__(self):
        self.forward_passed_values = []
        super().__init__()
        self.ff = Linear(in_features=4, out_features=3)

    def forward(self, x, lengths):
        self.forward_passed_values.append((x, lengths))
        x = self.ff(x)
        x = F.relu(x)
        # Permute the tag space as CrossEntropyLoss expects an output shape of: [batch_size, nb_classes, seq_length]
        # see: https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291
        return x.permute(0, 2, 1)


class AnnotateDatasetModuleMock(Module):

    def __init__(self):
        self.forward_passed_values = []
        super().__init__()

    def forward(self, x, lengths):
        self.forward_passed_values.append((x, lengths))
        return tensor(
            [
                [[0.1, 1.1, 0.3],
                 [0, 1.2, 0.3],
                 [0, 0, 1.2]],  # Sentence 1 (B, I, O)
                [[0.1, 1.1, 1.3],
                 [1.0, 0.1, 0.3],
                 [1.1, 0.6, 1.2]],  # Sentence 2 (O, B, B)
                [
                    [0.6, 1.1, 1.3],
                    [1.0, 0.1, 1.5],
                    [1.1, 0.6, 1.2]],  # Sentence 3 (O, B, I)
                [
                    [0.9, 1.5, 1.3],
                    [1.0, 1.1, 0.3],
                    [1.1, 1.6, 1.2]],  # Sentence 4 (O, O, B)
            ]
        )
