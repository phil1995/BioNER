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
    args = parser.parse_args()

    parameters = TrainingParameters(encoder_embeddings_path=args.embeddings,
                                    batch_size=args.batchSize,
                                    training_dataset_path=args.training,
                                    validation_dataset_path=args.validation,
                                    test_dataset_path=args.test,
                                    model_save_path=args.modelOutputFolder,
                                    max_epochs=args.maxEpochs,
                                    num_workers=args.numWorkers)
    Annotator.train(parameters)


