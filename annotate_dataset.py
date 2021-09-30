import argparse

import fasttext
import torch
from fasttext.FastText import _FastText

from bioner.model.Annotator import Annotator
from bioner.model.CoNLLDataset import CoNLLDataset
from bioner.model.model_loader import DATEXISNERStackedBiLSTMLayerConfiguration, ModelLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def annotate_dataset_with_bioner(dataset_file_path, encoder: _FastText, model_path: str):
    dataset = CoNLLDataset(data_file_path=dataset_file_path,
                           encoder=encoder)
    model_configuration = current_best_bioner_model_configuration()
    model = ModelLoader.create_custom_stacked_datexis_ner_model(model_configuration)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return Annotator.annotate_dataset(dataset=dataset,
                                      model=model)


def current_best_bioner_model_configuration():
    return DATEXISNERStackedBiLSTMLayerConfiguration(input_vector_size=300,
                                                     feedforward_layer_size=2048,
                                                     lstm_layer_size=2048,
                                                     amount_of_stacked_bilstm_layer=1,
                                                     dropout_probability=0.0)


def main():
    parser = argparse.ArgumentParser(description='Annotate CoNLL dataset')
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
                        help='Path to the output file for storing the annotated CoNLL dataset',
                        required=True)
    parser.add_argument('--model',
                        type=str,
                        help='Path to the BioNER model file',
                        required=True)
    args = parser.parse_args()
    encoder = fasttext.load_model(
        path=args.embeddings)
    bioner_annotated_dataset = annotate_dataset_with_bioner(
        dataset_file_path=args.dataset,
        model_path=args.model,
        encoder=encoder)
    CoNLLDataset.write_dataset_to_file(dataset=bioner_annotated_dataset, file_path=args.outputFile)


if __name__ == '__main__':
    main()
