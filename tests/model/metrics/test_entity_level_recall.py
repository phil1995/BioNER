import pytest
import torch

from model.BIO2Tag import BIO2Tag
from model.metrics.EntityLevelPrecisionRecall import EntityLevelRecall
from entity_level_test_utils import transform_tag_to_prob, transform_tag_to_index


def test_multiclass_input():
    recall = EntityLevelRecall()
    assert recall._updated is False

    def _test(y_pred, y, batch_size, expected_precision):
        recall.reset()
        assert recall._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                recall.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))
        else:
            recall.update((y_pred, y))

        assert recall._type == "multiclass"
        assert recall._updated is True
        assert isinstance(recall.compute(), torch.Tensor)
        assert expected_precision == pytest.approx(recall.compute())


    def get_test_cases():

        test_cases = [
            # TP= 1; TP+FN = 3
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.OUTSIDE],
                1,
                1/3
            ),
            # Predicted Inside without leading Begin
            # Treat the Inside tag without leading Begin as an Begin TODO: is this really correct?
            # TP=2; TP+FN = 3
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.INSIDE, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                1,
                3/3
            ),
            # Predicted all outside
            # TP= 0; TP+FN = 3
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE, BIO2Tag.OUTSIDE],
                1,
                0
            ),
            # Multiple batches
            # TP= 1; TP+FN = 3
            (
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.BEGIN, BIO2Tag.OUTSIDE],
                [BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.BEGIN, BIO2Tag.INSIDE, BIO2Tag.OUTSIDE],
                2,
                1/3
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