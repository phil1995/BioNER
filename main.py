import argparse
import torch
from model.Annotator import Annotator, TrainingParameters

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Train Annotator')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('--embeddings',
                                type=str,
                                help='Path to the embeddings file',
                                required=True)
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
    args = parser.parse_args()

    parameters = TrainingParameters(encoder_embeddings_path=args.embeddings,
                                    batch_size=args.batchSize,
                                    learning_rate=args.learningRate,
                                    training_dataset_path=args.training,
                                    validation_dataset_path=args.validation,
                                    test_dataset_path=args.test,
                                    model_save_path=args.modelOutputFolder,
                                    max_epochs=args.maxEpochs,
                                    num_workers=args.numWorkers,
                                    tensorboard_log_directory_path=args.tensorboardLogDirectory,
                                    training_log_file_path=args.trainingsLogFile)
    Annotator.train(parameters)


