from __future__ import annotations
import argparse
import csv
import random
from typing import Optional, TextIO

import fasttext
import torch
from fasttext.FastText import _FastText

from bioner.misc.biobert import postprocess_predictions
from bioner.misc.scibert.scibert_eval import SciBERTNER
from bioner.model.Annotator import Annotator
from bioner.model.BIO2Tag import BIO2Tag
from bioner.model.CoNLLDataset import CoNLLDataset
from bioner.model.metrics.EntityLevelPrecisionRecall import convert_labeled_tokens_to_annotations, count_true_positives
from bioner.model.model_loader import DATEXISNERStackedBiLSTMLayerConfiguration, ModelLoader


class ErrorAnalysis:
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

        if true_positives == len(gold_standard_annotations):
            self.perfect_indices.add(sentence_index)
        else:
            self.error_indices.add(sentence_index)

    def compare_to(self, other_dataset: CoNLLDataset):
        other_analysis = ErrorAnalysis(gold_standard_dataset=self.gold_standard_dataset, dataset=other_dataset)
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
        ErrorAnalysis.write_tag_line_to_file(csv_writer, name_column=self.name, error_column=contains_error, tags=tags)

    @staticmethod
    def export_to_csv(error_analysis_objects: [ErrorAnalysis], output_file_path: str):
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
                ErrorAnalysis.write_tag_line_to_file(csv_writer, name_column='Gold Standard',
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


def annotate_dataset_with_scibert(dataset_file_path, contextual_ner_path: str):
    evaluator = SciBERTNER(contextual_ner_path=contextual_ner_path)
    dataset = CoNLLDataset(data_file_path=dataset_file_path, encoder=None)
    return evaluator.annotate(dataset=dataset)


def annotate_dataset_with_bioner(dataset_file_path, encoder: _FastText, model_path: str):
    dataset = CoNLLDataset(data_file_path=dataset_file_path,
                           encoder=encoder)
    model_configuration = current_best_bioner_model_configuration()
    model = ModelLoader.create_custom_stacked_datexis_ner_model(model_configuration)
    model.load_state_dict(torch.load(model_path))
    return Annotator.annotate_dataset(dataset=dataset,
                                      model=model)


def current_best_bioner_model_configuration():
    return DATEXISNERStackedBiLSTMLayerConfiguration(input_vector_size=300,
                                                     feedforward_layer_size=2048,
                                                     lstm_layer_size=2048,
                                                     amount_of_stacked_bilstm_layer=1,
                                                     dropout_probability=0.0)


def select_errors(gold_standard_dataset: CoNLLDataset, dataset: CoNLLDataset, n: int = 100, seed: int = 1632737901):
    analysis = ErrorAnalysis(gold_standard_dataset=gold_standard_dataset,
                             dataset=dataset,
                             name="BioNER")
    indices = list(range(len(dataset.sentences)))
    random.seed(seed)
    while len(analysis.error_indices) < n and len(analysis.get_all_sentence_indices()) < len(dataset.sentences):
        sentence_index = random.choice(indices)
        analysis.analyze_sentence(sentence_index=sentence_index)
    return analysis


def compare_analysis_to_other_annotator_predictions(error_analysis: ErrorAnalysis, annotated_dataset: CoNLLDataset):
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


def main():
    parser = argparse.ArgumentParser(description='Error Analysis')
    parser.add_argument('--embeddings',
                        type=str,
                        help='Path to the embeddings file',
                        required=True)
    parser.add_argument('--dataset',
                        type=str,
                        help='Path to the dataset file',
                        required=True)
    parser.add_argument('--outputFile',
                        type=str,
                        help='Path to the error analysis output file',
                        required=True)
    parser.add_argument('--bioNER',
                        type=str,
                        help='Path to the BioNER model file',
                        required=True)
    parser.add_argument('--sciBERT',
                        type=str,
                        help='Path to the SciBERT model file',
                        required=False)
    args = parser.parse_args()
    encoder = fasttext.load_model(
        path=args.embeddings)
    dataset_path = args.dataset
    bioner_annotated_dataset = annotate_dataset_with_bioner(
        dataset_file_path=dataset_path,
        model_path=args.bioNER,
        encoder=encoder)
    gold_standard_dataset = CoNLLDataset(
        data_file_path=dataset_path,
        encoder=None)
    analysis = select_errors(gold_standard_dataset=gold_standard_dataset,
                             dataset=bioner_annotated_dataset)
    all_analyses = [analysis]
    if args.sciBERT is not None:
        scibert_annotated_dataset = annotate_dataset_with_scibert(dataset_file_path=dataset_path,
                                                                  contextual_ner_path=args.sciBERT)
        scibert_analysis = analysis.compare_to(other_dataset=scibert_annotated_dataset)
        all_analyses.append(scibert_analysis)

    ErrorAnalysis.export_to_csv(error_analysis_objects=all_analyses,
                                output_file_path=args.outputFile)


if __name__ == '__main__':
    main()
