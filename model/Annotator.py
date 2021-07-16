from os.path import join
from typing import Optional

import fasttext
import torch
from fasttext.FastText import _FastText
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Precision, Recall, ConfusionMatrix
from ignite.utils import setup_logger
from torch import optim, nn

from model.BiLSTM import BiLSTM
from model.MedMentionsDataLoader import MedMentionsDataLoader
from model.MedMentionsDataset import MedMentionsDataset
from model.metrics.EntityLevelPrecisionRecall import EntityLevelPrecision, EntityLevelRecall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    embeddings_list, label_list = [], []
    for (_embedding, _label) in batch:
        embeddings_list.append(_embedding)
        label_list.append(_label)
    embeddings_list = torch.tensor(embeddings_list, dtype=torch.float)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return embeddings_list.to(device), label_list.to(device)


class TrainingParameters:
    def __init__(self, encoder_embeddings_path: str, training_dataset_path: str, validation_dataset_path: str,
                 batch_size: int, model_save_path: str, max_epochs: int, num_workers: int = 0,
                 test_dataset_path: Optional[str] = None):
        self.encoder_embeddings_path = encoder_embeddings_path
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.max_epochs = max_epochs
        self.num_workers = num_workers


class TestParameters:
    def __init__(self, encoder_embeddings_path: str, model_checkpoint_file_path: str, test_dataset_path: str):
        self.encoder_embeddings_path = encoder_embeddings_path
        self.model_save_path = model_checkpoint_file_path
        self.test_dataset_path = test_dataset_path


class Annotator:

    @staticmethod
    def train(parameters: TrainingParameters):
        print(f"Start training with batch size:{parameters.batch_size} max.Epochs:{parameters.max_epochs} "
              f"on {parameters.training_dataset_path}")
        encoder = fasttext.load_model(parameters.encoder_embeddings_path)

        training_dataset = Annotator.load_dataset(path=parameters.training_dataset_path, encoder=encoder)
        training_data_loader = MedMentionsDataLoader(dataset=training_dataset, shuffle=True,
                                                     num_workers=parameters.num_workers,
                                                     batch_size=parameters.batch_size, collate_fn=collate_batch)

        model = Annotator.create_model(input_vector_size=encoder.get_dimension())

        trainer = Annotator.create_trainer(model=model, optimizer=optim.Adam(model.parameters(), lr=0.001),
                                           criterion=nn.CrossEntropyLoss())

        validation_dataset = Annotator.load_dataset(path=parameters.validation_dataset_path, encoder=encoder)
        validation_data_loader = MedMentionsDataLoader(dataset=validation_dataset, shuffle=True,
                                                       num_workers=parameters.num_workers,
                                                       batch_size=parameters.batch_size, collate_fn=collate_batch)
        evaluator = Annotator.create_evaluator(model)
        # Run model's validation at the end of each epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, Annotator.validation, trainer, evaluator,
                                  validation_data_loader)

        handler = EarlyStopping(patience=10, score_function=Annotator.score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        evaluator.add_event_handler(Events.COMPLETED, handler)

        # Attach the checkpoint_handler to an evaluator to save best model during the training according to computed
        # validation metric
        to_save = {'model': model}
        checkpoint_handler = Checkpoint(
            to_save, DiskSaver(parameters.model_save_path, create_dir=True, atomic=True),
            n_saved=1, filename_prefix='best',
            score_function=Annotator.score_function, score_name="val_f1",
            global_step_transform=global_step_from_engine(trainer)
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        # Add Logger
        trainer.logger = setup_logger("trainer")
        evaluator.logger = setup_logger("evaluator")

        # Start the training
        trainer.run(training_data_loader, max_epochs=parameters.max_epochs)

        print("Training done")

        # Test
        if parameters.test_dataset_path is not None:
            Annotator.test(encoder=encoder, test_dataset_path=parameters.test_dataset_path,
                           best_model_state_path=join(parameters.model_save_path, checkpoint_handler.last_checkpoint),
                           num_workers=parameters.num_workers)

    @staticmethod
    def test(encoder: _FastText, test_dataset_path: str, best_model_state_path: str, num_workers: int = 0):
        print(f"Start test on: {test_dataset_path}")
        model = Annotator.create_model(input_vector_size=encoder.get_dimension())
        model.load_state_dict(torch.load(best_model_state_path))
        test_dataset = Annotator.load_dataset(path=test_dataset_path, encoder=encoder)
        test_data_loader = MedMentionsDataLoader(dataset=test_dataset, shuffle=False, num_workers=num_workers,
                                                 batch_size=1, collate_fn=collate_batch)
        evaluator = Annotator.create_evaluator(model)
        state = evaluator.run(test_data_loader)
        print(state.metrics)

    @staticmethod
    def create_evaluator(model):
        precision = EntityLevelPrecision()
        recall = EntityLevelRecall()
        f1 = (precision * recall * 2 / (precision + recall + 1e-20)).mean()
        confusion_matrix = ConfusionMatrix(num_classes=3)
        evaluator = create_supervised_evaluator(model, metrics={"Precision": precision,
                                                                "Recall": recall,
                                                                "F1": f1,
                                                                "confusion_matrix": confusion_matrix})

        return evaluator

    @staticmethod
    def create_f1_score():
        precision = Precision(average=False)
        recall = Recall(average=False)
        f1 = (precision * recall * 2 / (precision + recall + 1e-20)).mean()
        return f1

    @staticmethod
    def create_trainer(model, optimizer, criterion):
        def train_step(engine, batch):
            model.train()
            inputs, labels = batch
            # After each iteration of the training step, reset the local gradients stored in the network to zero.
            model.zero_grad()

            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            return {'loss': batch_loss.item(),
                    'y_pred': outputs,
                    'y': labels}

        trainer = Engine(train_step)
        return trainer

    @staticmethod
    def validation(trainer, evaluator, validation_data_loader):
        state = evaluator.run(validation_data_loader)
        # print computed metrics
        print(
            f"Validation - Epoch:{trainer.state.epoch} | Precision:{state.metrics['Precision']},"
            f" Recall:{state.metrics['Recall']}, F1:{state.metrics['F1']}")

    @staticmethod
    def score_function(engine):
        return engine.state.metrics['F1']

    @staticmethod
    def load_dataset(path, encoder) -> MedMentionsDataset:
        structured_dataset = MedMentionsDataset(path, encoder=encoder)
        return structured_dataset

    @staticmethod
    def create_model(input_vector_size: int, feedforward_layer_size: int = 512, lstm_layer_size: int = 256) -> BiLSTM:
        """

        :param input_vector_size: the size of the embeddings
        :param feedforward_layer_size: (DATEXIS: 512)
        :param lstm_layer_size: (DATEXIS: 256)
        :return:
        """
        model = BiLSTM(input_vector_size=input_vector_size, feedforward_layer_size=feedforward_layer_size,
                       lstm_layer_size=lstm_layer_size)
        model.to(device)
        return model
