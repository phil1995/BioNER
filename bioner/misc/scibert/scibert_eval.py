import argparse
from copy import deepcopy

import numpy as np
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from bioner.model.bio2tag import BIO2Tag
from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.token import Token
from bioner.model.metrics.entity_level_precision_recall import convert_labeled_tokens_to_annotations, count_true_positives

from sklearn.metrics import precision_recall_fscore_support


class SciBERTNER:
    """
    SciBERT evaluation inspired by MedLinker: https://github.com/danlou/MedLinker
    """

    def __init__(self, contextual_ner_path):
        self.contextual_ner = Predictor.from_path(contextual_ner_path, cuda_device=0)

        # switch-off tokenizer (expect pretokenized, space-separated strings)
        self.contextual_ner._tokenizer = JustSpacesWordSplitter()

        # load labels (to use logits, wip)
        self.contextual_ner_labels = []
        with open(contextual_ner_path + 'vocabulary/labels.txt', 'r') as labels_f:
            for line in labels_f:
                self.contextual_ner_labels.append(line.strip())

    def evaluate(self, dataset: CoNLLDataset):
        total_true_positives = 0
        total_true_negatives = 0  # TN = 0 as every token gets annotated
        total_false_positives = 0
        total_false_negatives = 0

        total_gold_standard_bio2_tags = []
        total_predicted_bio2_tags = []
        for document in dataset.documents:
            for sentence in document:
                sentence_str = " ".join([token.text for token in sentence.tokens])
                predictions = self.contextual_ner.predict(sentence_str)

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

                # Token Level Metrics
                assert len(predicted_bio2_tags) == len(gold_standard_bio2_tags)
                total_predicted_bio2_tags.extend(predicted_bio2_tags)
                total_gold_standard_bio2_tags.extend(gold_standard_bio2_tags)
                assert len(total_predicted_bio2_tags) == len(total_gold_standard_bio2_tags)

        precision = total_true_positives / (total_true_positives + total_false_positives)
        recall = total_true_positives / (total_true_positives + total_false_negatives)
        f1 = 2.0 * precision * recall / (precision + recall)
        print(f"TP:{total_true_positives} FP:{total_false_positives} "
              f"TN:{total_true_negatives} FN:{total_false_negatives} "
              f"Precision:{precision} Recall:{recall} F1:{f1}")

        token_lvl_precision, token_lvl_recall, token_lvl_f1, _ = precision_recall_fscore_support(
            np.array([BIO2Tag.get_index(tag) for tag in total_gold_standard_bio2_tags]),
            np.array([BIO2Tag.get_index(tag) for tag in total_predicted_bio2_tags]),
            average='micro')
        print("Token Level Metrics:")
        print(f"Precision:{token_lvl_precision} Recall:{token_lvl_recall} F1:{token_lvl_f1}")
        token_lvl_precision, token_lvl_recall, token_lvl_f1, _ = precision_recall_fscore_support(
            np.array([BIO2Tag.get_index(tag) for tag in total_gold_standard_bio2_tags]),
            np.array([BIO2Tag.get_index(tag) for tag in total_predicted_bio2_tags]),
            average='macro')
        print(f"Precision:{token_lvl_precision} Recall:{token_lvl_recall} F1:{token_lvl_f1}")

    def predict_labels(self, sentence: [Token]) -> [BIO2Tag]:
        sentence_str = " ".join([token.text for token in sentence.tokens])
        predictions = self.contextual_ner.predict(sentence_str)
        tokens = predictions['words']
        assert tokens == [token.text for token in sentence.tokens]

        predicted_tags = predictions['tags']
        predicted_bio2_tags = create_bio2_tags_from_bioul_tags(predicted_tags)
        return predicted_bio2_tags

    def annotate(self, dataset: CoNLLDataset) -> CoNLLDataset:
        annotated_dataset = deepcopy(dataset)

        for doc_index, document in enumerate(dataset.documents):
            for sentence_index, sentence in enumerate(document.sentences):
                predicted_labels = self.predict_labels(sentence=sentence)
                for token_index, label in enumerate(predicted_labels):
                    document_to_annotate = annotated_dataset.documents[doc_index]
                    sentence_to_annotate = document_to_annotate.sentences[sentence_index]
                    token = sentence_to_annotate.tokens[token_index]
                    token.tag = label
        return annotated_dataset

    def annotate_to_file(self, dataset_file_path: str, output_file_path: str):
        dataset = CoNLLDataset(data_file_path=dataset_file_path)
        annotated_dataset = self.annotate(dataset=dataset)
        CoNLLDataset.write_dataset_to_file(dataset=annotated_dataset, file_path=output_file_path)


def create_bio2_tags_from_bioul_tags(tags: [str]) -> [BIO2Tag]:
    return list(map(create_bio2_tag_from_bioul_tag, tags))


def create_bio2_tag_from_bioul_tag(tag: str) -> BIO2Tag:
    first_character = tag[0]
    if first_character in ['B', 'I', 'O']:
        return BIO2Tag(first_character)
    if first_character == 'L':
        return BIO2Tag.INSIDE
    if first_character == 'U':
        return BIO2Tag.BEGIN
    else:
        raise ValueError('Tag does not conform to the BIOUL scheme')


def write_dataset_to_conll_file(dataset: CoNLLDataset, annotated_dataset: CoNLLDataset, file_path: str):
    with open(file_path, 'w', encoding='utf8') as output_file:
        for document_index, document in enumerate(dataset.documents):
            annotated_document = annotated_dataset.documents[document_index]
            for sentence_index, sentence in enumerate(document.sentences):
                annotated_sentence = annotated_document.sentences[sentence_index]
                output_file.write("\n")
                for token_index, token in enumerate(sentence.tokens):
                    annotated_token = annotated_sentence.tokens[token_index]
                    token_str = f"{token.text} {token.tag.value} {annotated_token.tag.value}"
                    output_file.write(token_str + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SciBERT')
    parser.add_argument("--evaluationDatasetPath", required=True)
    parser.add_argument("--contextualNERPath", required=True)
    parser.add_argument("--outputFilePath")
    args = parser.parse_args()
    evaluator = SciBERTNER(contextual_ner_path=args.contextualNERPath)
    evaluation_dataset = CoNLLDataset(data_file_path=args.evaluationDatasetPath)
    if args.outputFilePath:
        evaluator.annotate_to_file(dataset_file_path=args.evaluationDatasetPath,
                                   output_file_path=args.outputFilePath)
    else:
        evaluator.evaluate(dataset=evaluation_dataset)
