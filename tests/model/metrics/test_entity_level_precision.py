import pytest
import torch

from model.BIO2Tag import BIO2Tag
from model.metrics.EntityLevelPrecisionRecall import EntityLevelPrecision
from entity_level_test_utils import transform_tag_to_prob, transform_tag_to_index


def test_multiclass_input():
    pr = EntityLevelPrecision()
    assert pr._updated is False

    def _test(y_pred, y, batch_size, expected_precision):
        pr.reset()
        assert pr._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))
        else:
            pr.update((y_pred, y))

        assert pr._type == "multiclass"
        assert pr._updated is True
        assert isinstance(pr.compute(), torch.Tensor)
        assert expected_precision == pytest.approx(pr.compute())

    def get_test_cases():

        test_cases = [
            # TP= 1; TP+FP = 2
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.OUTSIDE],
                1,
                0.5
            ),
            # Predicted Inside without leading Begin
            # Treat the Inside tag without leading Begin as an Begin TODO: is this really correct?
            # TP=3; TP+FP = 3
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.INSIDE, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                1,
                3/3
            ),
            # Predicted all outside
            # TP= 0; TP+FP = 0
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE],
                1,
                0
            ),
            # Multiple batches
            # TP= 1; TP+FP = 2
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.OUTSIDE],
                2,
                0.5
            ),
        ]

        return test_cases

    test_cases = get_test_cases()
    for y, y_pred, batch_size, expected_precision in test_cases:
        # unsqueeze to add additional dimension
        # permute y_pred to get shape (batch_size, num_categories, sentence length)
        y_pred = torch.tensor(list(map(transform_tag_to_prob, y_pred))).unsqueeze(0).permute(0, 2, 1)
        y = torch.tensor(list(map(transform_tag_to_index, y))).unsqueeze(0)
        _test(y_pred, y, batch_size, expected_precision)
