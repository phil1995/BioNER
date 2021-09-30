from __future__ import annotations
import argparse
import csv
import random
from collections import defaultdict
from typing import Optional

from bioner.model.BIO2Tag import BIO2Tag
from bioner.model.CoNLLDataset import CoNLLDataset
from bioner.model.metrics.EntityLevelPrecisionRecall import convert_labeled_tokens_to_annotations, count_true_positives


class ManualErrorAnalysis:
    def __init__(self, gold_standard_dataset: CoNLLDataset, dataset: CoNLLDataset, name: Optional[str] = ""):
        self.error_indices = set()
        self.perfect_indices = set()
        gold_standard_dataset.flatten_dataset()
        self.gold_standard_dataset = gold_standard_dataset
        dataset.flatten_dataset()
        self.dataset = dataset
        self.name = name

    def analyze_sentence(self, sentence_index):
        predicted_labels = [token.tag for token in self.dataset.sentences[sentence_index]]
        gold_standard_labels = [token.tag for token in self.gold_standard_dataset.sentences[sentence_index]]
        predicted_annotations = convert_labeled_tokens_to_annotations([predicted_labels])
        gold_standard_annotations = convert_labeled_tokens_to_annotations([gold_standard_labels])
        true_positives = count_true_positives(gold_standard_annotations=gold_standard_annotations,
                                              predicted_annotations=predicted_annotations)

        if true_positives == len(gold_standard_annotations) == len(predicted_annotations):
            self.perfect_indices.add(sentence_index)
        else:
            self.error_indices.add(sentence_index)

    def compare_to(self, other_dataset: CoNLLDataset):
        other_analysis = ManualErrorAnalysis(gold_standard_dataset=self.gold_standard_dataset, dataset=other_dataset)
        for perfect_index in self.perfect_indices:
            other_analysis.analyze_sentence(perfect_index)
        for error_index in self.error_indices:
            other_analysis.analyze_sentence(error_index)
        return other_analysis

    def get_all_sentence_indices(self):
        return self.perfect_indices.union(self.error_indices)

    def write_sentence_tags_to_file(self, csv_writer: csv.writer, sentence_index: int):
        sentence = self.dataset.sentences[sentence_index]
        tags = [token.tag for token in sentence]
        contains_error = 1 if sentence_index in self.error_indices else 0
        ManualErrorAnalysis.write_tag_line_to_file(csv_writer, name_column=self.name, error_column=contains_error,
                                                   tags=tags)

    @staticmethod
    def export_to_csv(error_analysis_objects: [ManualErrorAnalysis], output_file_path: str):
        sentence_indices = list(error_analysis_objects[0].get_all_sentence_indices())
        sentence_indices.sort()
        gold_standard_dataset = error_analysis_objects[0].gold_standard_dataset
        with open(output_file_path, 'w') as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', lineterminator='\n')
            for sentence_index in sentence_indices:
                sentence = gold_standard_dataset.sentences[sentence_index]
                # write token str
                line = ['Text', '-1'] + [token.text for token in sentence]
                csv_writer.writerow(line)

                # write gold standard tags
                ManualErrorAnalysis.write_tag_line_to_file(csv_writer, name_column='Gold Standard',
                                                           error_column=-1,
                                                           tags=[token.tag for token in sentence])

                for error_analysis in error_analysis_objects:
                    error_analysis.write_sentence_tags_to_file(csv_writer,
                                                               sentence_index=sentence_index)
                # Add empty line between each sentence
                output_file.write("\n")

    @staticmethod
    def write_tag_line_to_file(csv_writer: csv.writer, name_column: str, error_column: int, tags: [BIO2Tag]):
        line = [name_column, error_column] + tags
        csv_writer.writerow(line)


class ErrorStatisticResult:
    def __init__(self):
        self.errors = defaultdict(lambda: 0)
        self.total_annotations = defaultdict(lambda: 0)


class OverlappingStatisticResult:
    def __init__(self, true_positives: int, false_positives: int, false_negatives: int, name: str):
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.name = name


class ErrorAnalysis:
    def __init__(self, gold_standard_dataset: CoNLLDataset, dataset: CoNLLDataset, name: Optional[str] = ""):
        gold_standard_dataset.flatten_dataset()
        self.gold_standard_dataset = gold_standard_dataset
        dataset.flatten_dataset()
        self.dataset = dataset
        self.name = name
        self.false_positive_annotations = []
        self.false_negative_annotations = []
        self.true_positive_annotations = []

    def analyze_annotations(self):
        predicted_labels = [[token.tag for token in sentence] for sentence in self.dataset.sentences]
        gold_standard_labels = [[token.tag for token in sentence] for sentence in self.gold_standard_dataset.sentences]

        predicted_annotations = convert_labeled_tokens_to_annotations(predicted_labels)
        gold_standard_annotations = convert_labeled_tokens_to_annotations(gold_standard_labels)

        self.false_positive_annotations = list(set(predicted_annotations) - set(gold_standard_annotations))
        self.false_negative_annotations = list(set(gold_standard_annotations) - set(predicted_annotations))
        self.true_positive_annotations = list(set(predicted_annotations).intersection(set(gold_standard_annotations)))


def calc_overlapping_statistics(analysis_1: ErrorAnalysis, analysis_2: ErrorAnalysis):
    overlapping_false_positives = list(
        set(analysis_1.false_positive_annotations).intersection(set(analysis_2.false_positive_annotations)))
    overlapping_false_negatives = list(
        set(analysis_1.false_negative_annotations).intersection(set(analysis_2.false_negative_annotations)))
    overlapping_true_positives = list(
        set(analysis_1.true_positive_annotations).intersection(set(analysis_2.true_positive_annotations)))
    return OverlappingStatisticResult(true_positives=len(overlapping_true_positives),
                                      false_positives=len(overlapping_false_positives),
                                      false_negatives=len(overlapping_false_negatives),
                                      name=f"{analysis_1.name} - {analysis_2.name}")


class ErrorStatistics:
    def __init__(self, gold_standard_dataset: CoNLLDataset):
        gold_standard_dataset.flatten_dataset()
        self.gold_standard_dataset = gold_standard_dataset

    def calc_error_stats_for_lengths(self, dataset: CoNLLDataset) -> ErrorStatisticResult:
        dataset.flatten_dataset()
        result = ErrorStatisticResult()
        for sentence_index, sentence in enumerate(self.gold_standard_dataset.sentences):
            predicted_labels = [token.tag for token in dataset.sentences[sentence_index]]
            gold_standard_labels = [token.tag for token in sentence]
            predicted_annotations = convert_labeled_tokens_to_annotations([predicted_labels])
            gold_standard_annotations = convert_labeled_tokens_to_annotations([gold_standard_labels])

            for annotation in gold_standard_annotations:
                result.total_annotations[len(annotation)] += 1
                if annotation not in predicted_annotations:
                    result.errors[len(annotation)] += 1
        return result


def select_errors(gold_standard_dataset: CoNLLDataset, dataset: CoNLLDataset, n: int = 100, seed: int = 1632737901):
    analysis = ManualErrorAnalysis(gold_standard_dataset=gold_standard_dataset,
                                   dataset=dataset,
                                   name="BioNER")
    indices = list(range(len(dataset.sentences)))
    random.seed(seed)
    while len(analysis.error_indices) < n and len(analysis.get_all_sentence_indices()) < len(dataset.sentences):
        sentence_index = random.choice(indices)
        analysis.analyze_sentence(sentence_index=sentence_index)
    return analysis


def compare_analysis_to_other_annotator_predictions(error_analysis: ManualErrorAnalysis,
                                                    annotated_dataset: CoNLLDataset):
    other_annotator_analysis = error_analysis.compare_to(other_dataset=annotated_dataset)
    same_errors = list(error_analysis.error_indices & other_annotator_analysis.error_indices)
    print(len(same_errors))


def deep_error_comparison(sentence_index: int, first_annotated_dataset: CoNLLDataset,
                          second_annotated_dataset: CoNLLDataset, gold_standard_dataset: CoNLLDataset):
    predicted_labels_dataset_1 = [token.tag for token in first_annotated_dataset.sentences[sentence_index]]
    predicted_labels_dataset_2 = [token.tag for token in second_annotated_dataset.sentences[sentence_index]]
    gold_standard_labels = [token.tag for token in gold_standard_dataset.sentences[sentence_index]]
    predicted_annotations_1 = convert_labeled_tokens_to_annotations([predicted_labels_dataset_1])
    predicted_annotations_2 = convert_labeled_tokens_to_annotations([predicted_labels_dataset_2])
    gold_standard_annotations = convert_labeled_tokens_to_annotations([gold_standard_labels])


def percentage_of_model(overlapping_result: OverlappingStatisticResult, analysis_1: ErrorAnalysis,
                        analysis_2: ErrorAnalysis):
    print(f"Overlapping Results for: {overlapping_result.name}\n"
          f"TP: {overlapping_result.true_positives} "
          f"({overlapping_result.true_positives / len(analysis_1.true_positive_annotations)} of {analysis_1.name} and "
          f"{overlapping_result.true_positives / len(analysis_2.true_positive_annotations)} of {analysis_2.name})\n"
          f"FP: {overlapping_result.false_positives} "
          f"({overlapping_result.false_positives / len(analysis_1.false_positive_annotations)} of {analysis_1.name} and "
          f"{overlapping_result.false_positives / len(analysis_2.false_positive_annotations)} of {analysis_2.name})\n"
          f"FN: {overlapping_result.false_negatives} "
          f"({overlapping_result.false_negatives / len(analysis_1.false_negative_annotations)} of {analysis_1.name} and "
          f"{overlapping_result.false_negatives / len(analysis_2.false_negative_annotations)} of {analysis_2.name})")


def main():
    parser = argparse.ArgumentParser(description='Error Analysis')
    parser.add_argument('--gold_dataset',
                        type=str,
                        help='Path to the gold standard dataset file',
                        required=True)
    parser.add_argument('--bioNER',
                        type=str,
                        help='Path to the dataset annotated with BioNER',
                        required=True)
    parser.add_argument('--sciBERT',
                        type=str,
                        help='Path to the dataset annotated with SciBERT',
                        required=False)
    parser.add_argument('--outputFile',
                        type=str,
                        help='Path to the error analysis output file',
                        required=False)
    args = parser.parse_args()
    gold_standard_dataset = CoNLLDataset(data_file_path=args.gold_dataset)

    bioner_annotated_dataset = CoNLLDataset(data_file_path=args.bioNER)

    analysis = select_errors(gold_standard_dataset=gold_standard_dataset,
                             dataset=bioner_annotated_dataset)

    automatic_bioner_error_analysis = ErrorAnalysis(gold_standard_dataset=gold_standard_dataset,
                                                    dataset=bioner_annotated_dataset,
                                                    name="BioNER")
    automatic_bioner_error_analysis.analyze_annotations()

    all_analyses = [analysis]
    if args.sciBERT is not None:
        scibert_annotated_dataset = CoNLLDataset(data_file_path=args.sciBERT)
        scibert_analysis = analysis.compare_to(other_dataset=scibert_annotated_dataset)
        scibert_analysis.name = "SciBERT"
        all_analyses.append(scibert_analysis)

        automatic_scibert_error_analysis = ErrorAnalysis(gold_standard_dataset=gold_standard_dataset,
                                                         dataset=scibert_annotated_dataset,
                                                         name="SciBERT")
        automatic_scibert_error_analysis.analyze_annotations()
        overlapping_result = calc_overlapping_statistics(automatic_bioner_error_analysis,
                                                         automatic_scibert_error_analysis)
        percentage_of_model(overlapping_result=overlapping_result,
                            analysis_1=automatic_bioner_error_analysis,
                            analysis_2=automatic_scibert_error_analysis)

    if args.outputFile is not None:
        ManualErrorAnalysis.export_to_csv(error_analysis_objects=all_analyses,
                                          output_file_path=args.outputFile)


if __name__ == '__main__':
    main()
