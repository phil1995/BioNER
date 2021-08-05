from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import WhitespaceTokenizer

from bioner.model.BIO2Tag import BIO2Tag
from bioner.model.MedMentionsDataset import MedMentionsDataset
from bioner.model.metrics.EntityLevelPrecisionRecall import convert_labeled_tokens_to_annotations, count_true_positives


class SciBERTNER:
    """
    SciBERT evaluation inspired by MedLinker: https://github.com/danlou/MedLinker
    """
    def __init__(self, contextual_ner_path):
        self.contextual_ner = Predictor.from_path(contextual_ner_path, cuda_device=0)

        # switch-off tokenizer (expect pretokenized, space-separated strings)
        self.contextual_ner._tokenizer = WhitespaceTokenizer()

        # load labels (to use logits, wip)
        self.contextual_ner_labels = []
        with open(contextual_ner_path + 'vocabulary/labels.txt', 'r') as labels_f:
            for line in labels_f:
                self.contextual_ner_labels.append(line.strip())

    def eval(self, dataset: MedMentionsDataset):
        total_true_positives = 0
        total_true_negatives = 0  # TN = 0 as every token gets annotated
        total_false_positives = 0
        total_false_negatives = 0
        for document in dataset.documents:
            for sentence in document:
                sentence_str = " ".join([token.text for token in sentence.tokens])
                predictions = self.contextual_ner_labels.predict(sentence_str)

                tokens = predictions['words']
                predicted_tags = predictions['tags']
                predicted_bio2_tags = create_bio2_tags_from_bioul_tags(predicted_tags)
                predicted_annotations = convert_labeled_tokens_to_annotations(labeled_tokens=[predicted_bio2_tags])

                gold_standard_token_text = [token.text for token in sentence.tokens]
                assert tokens == gold_standard_token_text
                gold_standard_bio2_tags = [token.tag for token in sentence.tokens]
                gold_standard_annotations = convert_labeled_tokens_to_annotations(
                    labeled_tokens=[gold_standard_bio2_tags])

                true_positives = count_true_positives(gold_standard_annotations=gold_standard_annotations,
                                                      predicted_annotations=predicted_annotations)

                false_negatives = len(gold_standard_annotations) - true_positives
                false_positives = len(predicted_annotations) - true_positives

                total_true_positives += true_positives
                total_false_positives += false_positives
                total_false_negatives += false_negatives
        precision = total_true_positives / (total_true_positives + total_false_positives)
        recall = total_true_positives / (total_true_positives + total_false_negatives)
        f1 = 2.0 * precision * recall / (precision + recall)
        print(f"TP:{total_true_positives} FP:{total_false_positives} "
              f"TN:{total_true_negatives} FN:{total_false_negatives} "
              f"Precision:{precision} Recall:{recall} F1:{f1}")


def create_bio2_tags_from_bioul_tags(tags: [str]) -> [BIO2Tag]:
    return map(create_bio2_tag_from_bioul_tag, tags)


def create_bio2_tag_from_bioul_tag(tag: str) -> BIO2Tag:
    first_character = tag[0]
    if first_character in ['B', 'I', '0']:
        return BIO2Tag(first_character)
    if first_character == 'L':
        return BIO2Tag.INSIDE
    if first_character == 'U':
        return BIO2Tag.BEGIN
    else:
        raise ValueError('Tag does not conform to the BIOUL scheme')
