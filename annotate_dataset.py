import argparse
import os

import torch

from bioner.model.annotator import Annotator
from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.bioner_model import BioNER
from bioner.model.encoder.fasttext_encoder import FasttextEncoder, FastTextEmbedding
from dataset_to_conll_file import write_dataset_to_conll_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def annotate_dataset_with_bioner(dataset_file_path, encoder: FasttextEncoder, model_path: str):
    dataset = Annotator.load_dataset(path=dataset_file_path, encoder=encoder)
    model = BioNER(encoder.get_embeddings_vector_size())
    model.load_state_dict(torch.load(model_path, map_location=device))
    return Annotator.annotate_dataset(dataset=dataset,
                                      model=model)


def main():
    parser = argparse.ArgumentParser(description='Annotate CoNLL dataset')
    parser.add_argument('--embeddings',
                        type=str,
                        help='Path to the embeddings file',
                        required=False)
    parser.add_argument('--embeddingsRoot',
                        type=str,
                        help='Path where the embeddings can be downloaded to',
                        required=False)
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
    parser.add_argument('--enableExportCoNLL',
                        action='store_true',
                        help='Enable to store the predictions side by side'
                             ' with the gold standard labels for the original conll eval script')
    args = parser.parse_args()

    if args.embeddings is None and args.embeddingsRoot is None:
        parser.error("You need to set either --embeddings or --embeddingsRoot")

    embeddings_file_path = None
    if args.embeddings is None:
        fasttext_embedding = FastTextEmbedding(embeddings_root=args.embeddingsRoot, ngram_range="3-4")
        embeddings_file_path = fasttext_embedding.filepath
    else:
        embeddings_file_path = args.embeddings

    encoder = FasttextEncoder(embeddings_file_path=embeddings_file_path)
    bioner_annotated_dataset = annotate_dataset_with_bioner(dataset_file_path=args.dataset,
                                                            model_path=args.model,
                                                            encoder=encoder)
    CoNLLDataset.write_dataset_to_file(dataset=bioner_annotated_dataset, file_path=args.outputFile)

    if args.enableExportCoNLL:
        conll_output_path, _ = os.path.splitext(args.outputFile)
        conll_output_path = conll_output_path + ".conll"
        gold_standard_dataset = CoNLLDataset(data_file_path=args.dataset)
        write_dataset_to_conll_file(dataset=gold_standard_dataset,
                                    annotated_dataset=bioner_annotated_dataset,
                                    file_path=conll_output_path)


if __name__ == '__main__':
    main()
