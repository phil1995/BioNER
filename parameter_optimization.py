import argparse
import random
from os import path, mkdir

import torch
from torch import optim

from bioner.model.Annotator import TrainingParameters, Annotator
from bioner.model.encoder.FasttextEncoder import FasttextEncoder
from bioner.model.model_loader import DATEXISNERStackedBiLSTMLayerConfiguration, ModelLoader

lstm_sizes = [20, 256, 512, 1024, 2048, 4096]
ff_sizes = [150, 256, 512, 1024, 2048, 4096]
learning_rates = [0.005, 0.0005]


def get_model_save_path(output_folder, batch_size, learning_rate, ff_size, lstm_size, additional_bilstm_layers,
                        dropout):
    return path.join(output_folder,
                     f"batch_size={batch_size}_lr={learning_rate}_ff_size={ff_size}_lstm_size={lstm_size}_additional_bilstm_layers={additional_bilstm_layers}_dropout={dropout}")


def get_tensorboard_log_directory_path(model_save_path):
    return path.join(model_save_path, "tensorboard_logs")


def create_directories(output_folder, batch_size, learning_rate, ff_size, lstm_size, additional_bilstm_layers, dropout):
    model_save_path = get_model_save_path(output_folder=output_folder,
                                          batch_size=batch_size,
                                          learning_rate=learning_rate,
                                          ff_size=ff_size,
                                          lstm_size=lstm_size,
                                          additional_bilstm_layers=additional_bilstm_layers,
                                          dropout=dropout)
    mkdir(model_save_path)
    tensorboard_log_directory = get_tensorboard_log_directory_path(model_save_path=model_save_path)
    mkdir(tensorboard_log_directory)


def main():
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
    required_named.add_argument('--modelOutputFolder',
                                type=str,
                                help='The folder where the best model should be saved',
                                required=True)
    required_named.add_argument('--additionalBiLSTMLayers',
                                type=int,
                                help='The amount of additional BiLSTM layers (stacked)',
                                required=True)
    required_named.add_argument('--dropoutProbability',
                                type=float,
                                help='The dropout probability, should be between 0.0 and 1.0',
                                required=True)
    args = parser.parse_args()

    # Reproducibility
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1632737901)
    random.seed(1632737901)

    additional_bilstm_layers = args.additionalBiLSTMLayers
    dropout = args.dropoutProbability
    output_folder = args.modelOutputFolder
    max_epochs = 300
    batch_size = 64

    for learning_rate in learning_rates:
        for lstm_size in lstm_sizes:
            for ff_size in ff_sizes:
                create_directories(output_folder=output_folder,
                                   batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   ff_size=ff_size,
                                   lstm_size=lstm_size,
                                   additional_bilstm_layers=additional_bilstm_layers,
                                   dropout=dropout)

    encoder = FasttextEncoder(embeddings_file_path=args.embeddings)

    for learning_rate in learning_rates:
        for lstm_size in lstm_sizes:
            last_f1_score = 0.0
            for ff_size in ff_sizes:
                layer_configuration = DATEXISNERStackedBiLSTMLayerConfiguration(
                    input_vector_size=encoder.get_embeddings_vector_size(),
                    feedforward_layer_size=ff_size,
                    lstm_layer_size=lstm_size,
                    amount_of_stacked_bilstm_layer=additional_bilstm_layers,
                    dropout_probability=args.dropoutProbability)
                model = ModelLoader.load_model(name="CustomConfig_Stacked-DATEXIS-NER",
                                               layer_configuration=layer_configuration)
                model_save_path = get_model_save_path(output_folder=output_folder,
                                                      batch_size=batch_size,
                                                      learning_rate=learning_rate,
                                                      ff_size=ff_size,
                                                      lstm_size=lstm_size,
                                                      additional_bilstm_layers=additional_bilstm_layers,
                                                      dropout=dropout)

                tensorboard_log_directory = get_tensorboard_log_directory_path(model_save_path=model_save_path)

                parameters = TrainingParameters(encoder=encoder,
                                                batch_size=batch_size,
                                                training_dataset_path=args.training,
                                                validation_dataset_path=args.validation,
                                                model_save_path=model_save_path,
                                                max_epochs=max_epochs,
                                                tensorboard_log_directory_path=tensorboard_log_directory,
                                                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                                                model=model,
                                                faster_training_evaluation=True)
                state = Annotator.train(parameters)
                if state.metrics['F1'] < last_f1_score:
                    break
                last_f1_score = state.metrics['F1']


if __name__ == "__main__":
    main()
