import argparse

from model.Annotator import Annotator, TrainingParameters, TestParameters
from os.path import join
if __name__ == '__main__':
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
                                required=True)
    required_named.add_argument('--batchSize',
                                type=int,
                                help='Batch size',
                                required=True)
    required_named.add_argument('--modelOutputFolder',
                                type=str,
                                help='The folder where the best model should be saved',
                                required=True)
    args = parser.parse_args()

    parameters = TrainingParameters(encoder_embeddings_path=args.embeddings,
                                    batch_size=args.batchSize,
                                    training_dataset_path=args.training,
                                    validation_dataset_path=args.validation,
                                    test_dataset_path=args.test,
                                    model_save_path=args.modelOutputFolder)
    Annotator.train(parameters)