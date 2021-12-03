import argparse
import random

import torch
from torch import optim

from bioner.model.annotator import TrainingParameters, Annotator
from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.encoder.datexis_encoder import DATEXISEncoder
from bioner.model.model_loader import DATEXISNERLayerConfiguration, ModelLoader


def main():
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Train DATEXIS-NER')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('--training',
                                type=str,
                                help='Path to the training dataset file',
                                required=True)
    required_named.add_argument('--validation',
                                type=str,
                                help='Path to the validation dataset file',
                                required=True)
    required_named.add_argument('--test',
                                type=str,
                                help='Path to the test dataset file',
                                required=False)
    required_named.add_argument('--batchSize',
                                type=int,
                                help='Batch size',
                                required=True)
    required_named.add_argument('--learningRate',
                                type=float,
                                help='Learning rate',
                                required=True)
    required_named.add_argument('--modelOutputFolder',
                                type=str,
                                help='The folder where the best model should be saved',
                                required=True)
    required_named.add_argument('--maxEpochs',
                                type=int,
                                help='Maximum training epochs',
                                required=True)
    required_named.add_argument('--numWorkers',
                                type=int,
                                default=0,
                                help='Number of workers (defaults to 0)')
    required_named.add_argument('--tensorboardLogDirectory',
                                type=str,
                                help='The directory where to log the tensorboard data',
                                required=False)
    required_named.add_argument('--trainingsLogFile',
                                type=str,
                                help='The file path where to log the PyTorch Ignite training and validation',
                                required=False)
    required_named.add_argument('--enableFasterTraining',
                                action='store_true',
                                help='Enable faster training by compute metrics only every 10th epoch')
    args = parser.parse_args()

    # Reproducibility
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1632737901)
    random.seed(1632737901)

    encoder = DATEXISEncoder()
    training_dataset = CoNLLDataset(data_file_path=args.training)
    encoder.learn_trigram_encoding(training_dataset)
    layer_configuration = DATEXISNERLayerConfiguration(input_vector_size=encoder.get_embeddings_vector_size())
    model = ModelLoader.load_model(name="DATEXIS-NER",
                                   layer_configuration=layer_configuration)
    parameters = TrainingParameters(encoder=encoder,
                                    batch_size=args.batchSize,
                                    training_dataset_path=args.training,
                                    validation_dataset_path=args.validation,
                                    test_dataset_path=args.test,
                                    model_save_path=args.modelOutputFolder,
                                    max_epochs=args.maxEpochs,
                                    num_workers=args.numWorkers,
                                    tensorboard_log_directory_path=args.tensorboardLogDirectory,
                                    training_log_file_path=args.trainingsLogFile,
                                    optimizer=optim.Adam(model.parameters(), lr=args.learningRate),
                                    model=model,
                                    faster_training_evaluation=args.enableFasterTraining)
    Annotator.train(parameters)


if __name__ == "__main__":
    main()
