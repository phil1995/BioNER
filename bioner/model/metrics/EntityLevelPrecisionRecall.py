from typing import Sequence, Callable, Union

import torch
from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.precision import _BasePrecisionRecall

from bioner.model.BIO2Tag import BIO2Tag

__all__ = ["EntityLevelPrecision", "EntityLevelRecall"]


class EntityLevelPrecision(_BasePrecisionRecall):
    def __init__(
            self,
            output_transform: Callable = lambda x: x,
            average: bool = False,
            device: Union[str, torch.device] = torch.device("cpu"),
            ignore_index: int = -100
    ):
        super(EntityLevelPrecision, self).__init__(
            output_transform=output_transform, average=average, is_multilabel=False, device=device
        )
        self.ignore_index = ignore_index

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        num_classes = y_pred.size(1)
        if y.max() + 1 > num_classes:
            raise ValueError(
                f"y_pred contains less classes than y. Number of predicted classes is {num_classes}"
                f" and element in y has invalid class = {y.max().item() + 1}."
            )

        gold_standard_labels = _create_BIO2_labels_from_batch_indices(y, ignore_index=self.ignore_index)
        gold_standard_annotations = convert_labeled_tokens_to_annotations(labeled_tokens=gold_standard_labels)

        predicted_batch_indices = torch.argmax(y_pred, dim=1)
        predicted_labels = _create_BIO2_labels_from_batch_indices(predicted_batch_indices,
                                                                  ignore_index=self.ignore_index)

        filtered_predicted_labels = filtered_labels(gold_standard_labels=gold_standard_labels,
                                                    predicted_labels=predicted_labels)
        predicted_annotations = convert_labeled_tokens_to_annotations(labeled_tokens=filtered_predicted_labels)
        true_positives = count_true_positives(gold_standard_annotations=gold_standard_annotations,
                                              predicted_annotations=predicted_annotations)

        # Count the number of predicted entities
        all_positives = len(predicted_annotations)

        self._true_positives += torch.tensor(true_positives, dtype=torch.float64, device=self._device)
        self._positives += torch.tensor(all_positives, dtype=torch.float64, device=self._device)
        self._updated = True


class EntityLevelRecall(_BasePrecisionRecall):
    def __init__(
            self,
            output_transform: Callable = lambda x: x,
            average: bool = False,
            device: Union[str, torch.device] = torch.device("cpu"),
            ignore_index: int = -100
    ):
        super(EntityLevelRecall, self).__init__(
            output_transform=output_transform, average=average, is_multilabel=False, device=device
        )
        self.ignore_index = ignore_index

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        num_classes = y_pred.size(1)
        if y.max() + 1 > num_classes:
            raise ValueError(
                f"y_pred contains less classes than y. Number of predicted classes is {num_classes}"
                f" and element in y has invalid class = {y.max().item() + 1}."
            )

        gold_standard_labels = _create_BIO2_labels_from_batch_indices(y, ignore_index=self.ignore_index)
        gold_standard_annotations = convert_labeled_tokens_to_annotations(labeled_tokens=gold_standard_labels)

        predicted_batch_indices = torch.argmax(y_pred, dim=1)
        predicted_labels = _create_BIO2_labels_from_batch_indices(predicted_batch_indices,
                                                                  ignore_index=self.ignore_index)

        filtered_predicted_labels = filtered_labels(gold_standard_labels=gold_standard_labels,
                                                    predicted_labels=predicted_labels)
        predicted_annotations = convert_labeled_tokens_to_annotations(labeled_tokens=filtered_predicted_labels)
        true_positives = count_true_positives(gold_standard_annotations=gold_standard_annotations,
                                              predicted_annotations=predicted_annotations)

        # Count the number of gold standard entities
        all_positives = len(gold_standard_annotations)

        self._true_positives += torch.tensor(true_positives, dtype=torch.float64, device=self._device)
        self._positives += torch.tensor(all_positives, dtype=torch.float64, device=self._device)
        self._updated = True


class Annotation:
    def __init__(self, sentence_id: int, start_token_id: int, end_token_id: int):
        self.sentence_id = sentence_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

    def __eq__(self, other):
        if isinstance(other, Annotation):
            return self.sentence_id == other.sentence_id and self.start_token_id == other.start_token_id \
                   and self.end_token_id == other.end_token_id
        return False


def _create_BIO2_labels_from_batch_indices(indices_batch: [[int]], ignore_index: int) -> [[BIO2Tag]]:
    batch_labels = []
    for indices in indices_batch:
        labels = _create_BIO2_labels_from_indices(indices, ignore_index=ignore_index)
        batch_labels.append(labels)
    return batch_labels


def _create_BIO2_labels_from_indices(indices: [int], ignore_index: int) -> [BIO2Tag]:
    labels = []
    for tag_index in indices:
        if tag_index != ignore_index:
            labels.append(BIO2Tag.index_to_type(tag_index))
    return labels


def convert_labeled_tokens_to_annotations(labeled_tokens: [[BIO2Tag]]) -> [Annotation]:
    annotations = []
    for sentence_index, sentence in enumerate(labeled_tokens):
        tokens = []
        for token_index, tag in enumerate(sentence):
            if len(tokens) == 0:
                if tag == BIO2Tag.BEGIN:
                    tokens.append(token_index)
                elif tag == BIO2Tag.INSIDE:  # I after O, treat as B (see DATEXIS Repo: MentionAnnotation.java TODO: add url)
                    tokens.append(token_index)
                elif tag == BIO2Tag.OUTSIDE:
                    # O after O -> do nothing
                    pass
            else:
                if tag == BIO2Tag.BEGIN:
                    annotations.append(
                        Annotation(sentence_id=sentence_index, start_token_id=tokens[0], end_token_id=tokens[-1]))
                    tokens.clear()
                    tokens.append(token_index)
                elif tag == BIO2Tag.INSIDE:
                    tokens.append(token_index)
                elif tag == BIO2Tag.OUTSIDE:
                    annotations.append(
                        Annotation(sentence_id=sentence_index, start_token_id=tokens[0], end_token_id=tokens[-1]))
                    tokens.clear()
        # Last annotation
        if len(tokens) != 0:
            annotations.append(
                Annotation(sentence_id=sentence_index, start_token_id=tokens[0], end_token_id=tokens[-1]))
            tokens.clear()
    return annotations


def count_true_positives(gold_standard_annotations: [Annotation], predicted_annotations: [Annotation]) -> int:
    result = 0
    for predicted_annotation in predicted_annotations:
        if any(gold_standard_annotation == predicted_annotation for gold_standard_annotation in
               gold_standard_annotations):
            result += 1
    return result


def filtered_labels(gold_standard_labels, predicted_labels):
    """
    Filter out the padding from the predicted_labels:
    We can remove the padding with ignore_index from the gold_standard_labels but not directly from the predicted
    labels as the predicted index can differ from the orig. "padding index" / ignore_index.
    Therefore, we exploit that exactly one label is predicted per token and thus truncate the padding
    by using the length of the already filtered gold_standard_label.
    """
    filtered_predicted_labels = []
    for index, correct_labels in enumerate(gold_standard_labels):
        filtered_predicted_labels.append(predicted_labels[index][0: len(correct_labels)])
    return filtered_predicted_labels