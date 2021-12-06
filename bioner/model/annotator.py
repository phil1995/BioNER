import logging
from copy import deepcopy
from typing import Optional, Sequence, Dict

import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Loss, Metric
from ignite.utils import setup_logger
from torch import optim, nn

from bioner.model.conll_dataloader import CoNLLDataLoader
from bioner.model.conll_dataset import CoNLLDataset
from bioner.model.encoder.encoder import Encoder
from bioner.model.metrics.entity_level_precision_recall import EntityLevelPrecision, EntityLevelRecall, \
    _create_BIO2_labels_from_batch_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use ignore_index so that the padded outputs do not contribute to the input gradient when using CrossEntropyLoss
# see https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
ignore_label_index = -100


def collate_batch(batch):
    embeddings_list, label_list = [], []
    original_lengths = [len(embeddings) for embeddings, _ in batch]

    max_sequence_length = max(len(x) for _, x in batch)
    for (_embedding, _label) in batch:
        padded_embedding = np.zeros_like(_embedding[0])
        padded_label = np.ones_like(_label[0]) * ignore_label_index
        if len(_embedding) != len(_label):
            raise ValueError("Embeddings length differ from label length expected:", len(_embedding),
                             "but got:", len(_label))

        for i in range(len(_embedding), max_sequence_length):
            _embedding.append(padded_embedding)
            _label.append(padded_label)
        embeddings_list.append(_embedding)
        label_list.append(_label)
    embeddings_list = torch.tensor(embeddings_list, dtype=torch.float)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return embeddings_list.to(device), label_list.to(device), torch.tensor(original_lengths, dtype=torch.long).to(
        device)


class TrainingParameters:
    def __init__(self, encoder: Encoder, training_dataset_path: str, validation_dataset_path: str,
                 model: nn.Module, model_save_path: str, optimizer: optim,
                 batch_size: int, max_epochs: int, num_workers: int = 0,
                 tensorboard_log_directory_path: Optional[str] = None,
                 training_log_file_path: Optional[str] = None,
                 faster_training_evaluation: Optional[bool] = False):
        self.encoder = encoder
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.model = model
        self.model_save_path = model_save_path
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.tensorboard_log_directory_path = tensorboard_log_directory_path
        self.training_log_file_path = training_log_file_path
        self.faster_training_evaluation = faster_training_evaluation


class Annotator:

    @staticmethod
    def train(parameters: TrainingParameters):

        encoder = parameters.encoder
        model = parameters.model
        model = model.to(device)
        training_dataset = Annotator.load_dataset(path=parameters.training_dataset_path, encoder=encoder)
        training_data_loader = CoNLLDataLoader(dataset=training_dataset, shuffle=True,
                                               num_workers=parameters.num_workers,
                                               batch_size=parameters.batch_size, collate_fn=collate_batch)

        criterion = nn.CrossEntropyLoss(ignore_index=ignore_label_index)

        trainer = Annotator.create_trainer(model=model, optimizer=parameters.optimizer, criterion=criterion)

        validation_dataset = Annotator.load_dataset(path=parameters.validation_dataset_path, encoder=encoder)
        validation_data_loader = CoNLLDataLoader(dataset=validation_dataset, shuffle=True,
                                                 num_workers=parameters.num_workers,
                                                 batch_size=parameters.batch_size, collate_fn=collate_batch)
        precision = EntityLevelPrecision()
        recall = EntityLevelRecall()

        def deterministic_cross_entropy_loss(y_pred, y):
            """
            CrossEntropyLoss does not have a deterministic implementation for CUDA.
            For more details:  https://github.com/pytorch/pytorch/issues/46024

            Therefore, we calculate the CrossEntropyLoss on the CPU and move the tensor back to the default device
            afterwards.
            """
            y_pred = y_pred.to('cpu')
            y = y.to('cpu')
            loss = criterion(y_pred, y)
            loss = loss.to(device)
            return loss

        metrics = {"Precision": precision, "Recall": recall,
                   "F1": (precision * recall * 2 / (precision + recall + 1e-20)).mean(),
                   "loss": Loss(deterministic_cross_entropy_loss)}

        train_evaluator = Annotator.create_evaluator(model=model, metrics=metrics)
        validation_evaluator = Annotator.create_evaluator(model=model, metrics=metrics)

        training_evaluation_interval = 1
        if parameters.faster_training_evaluation:
            training_evaluation_interval = 10
            logging.warning(f"Faster Training Evaluation enabled - compute metrics "
                            f"only every {training_evaluation_interval} epochs")

        @trainer.on(Events.EPOCH_COMPLETED(every=training_evaluation_interval))
        def compute_training_metrics(engine):
            # Evaluate after each training (or every 10th if faster_training_evaluation is enabled) epoch
            # on training dataset
            Annotator.log_results(epoch=engine.state.epoch,
                                  state=train_evaluator.run(training_data_loader),
                                  logger=train_evaluator.logger)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            # Evaluate after each training epoch on validation dataset
            Annotator.log_results(epoch=engine.state.epoch,
                                  state=validation_evaluator.run(validation_data_loader),
                                  logger=validation_evaluator.logger)

        # Early stopping
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        handler = EarlyStopping(patience=10, score_function=Annotator.score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, handler)

        # Attach the checkpoint_handler to an evaluator to save best model during the training according to computed
        # validation metric
        to_save = {'model': model}
        checkpoint_handler = Checkpoint(
            to_save, DiskSaver(parameters.model_save_path, create_dir=True, atomic=True),
            n_saved=1, filename_prefix='best',
            score_function=Annotator.score_function, score_name="val_f1",
            global_step_transform=global_step_from_engine(trainer)
        )
        validation_evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        # Add Logger
        trainer.logger = setup_logger("Trainer", filepath=parameters.training_log_file_path, reset=True)
        train_evaluator.logger = setup_logger("Train Evaluator", filepath=parameters.training_log_file_path)
        validation_evaluator.logger = setup_logger("Validation Evaluator", filepath=parameters.training_log_file_path)

        # Add Tensorboard Logger
        tb_logger = None
        if parameters.tensorboard_log_directory_path is not None:
            tb_logger = Annotator.add_tensorboard_logger(log_dir=parameters.tensorboard_log_directory_path,
                                                         trainer=trainer,
                                                         train_evaluator=train_evaluator,
                                                         validation_evaluator=validation_evaluator)
        # Add Progress Bars
        ProgressBar(persist=False, desc="Train").attach(trainer)
        ProgressBar(persist=False, desc="Train Evaluation").attach(train_evaluator)
        ProgressBar(persist=False, desc="Validation Evaluation").attach(validation_evaluator)

        # Start the training
        print(f"Start training with batch size:{parameters.batch_size} max.Epochs:{parameters.max_epochs} "
              f"on {parameters.training_dataset_path}")
        trainer.run(training_data_loader, max_epochs=parameters.max_epochs)

        # Close Tensorboard Logger (if available)
        if tb_logger is not None:
            tb_logger.close()

        return validation_evaluator.state

    @staticmethod
    def log_results(epoch: int, state: State, logger: logging.Logger):
        logger.info(f"Training-Epoch:{epoch} | Evaluation Results | Precision:{state.metrics['Precision']},"
                    f" Recall:{state.metrics['Recall']}, F1:{state.metrics['F1']}")

    @staticmethod
    def score_function(engine):
        return engine.state.metrics['F1']

    @staticmethod
    def load_dataset(path, encoder) -> CoNLLDataset:
        dataset = CoNLLDataset(path)
        print(f"Start encoding dataset: {path}")
        encoder.encode(dataset)
        print(f"Finished encoding dataset: {path}")
        return dataset

    @staticmethod
    def add_tensorboard_logger(log_dir: str, trainer: Engine, train_evaluator: Engine,
                               validation_evaluator: Engine) -> TensorboardLogger:
        tb_logger = TensorboardLogger(log_dir=log_dir)
        for tag, engine in [("training", train_evaluator), ("validation", validation_evaluator)]:
            tb_logger.attach_output_handler(
                engine=engine,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )
        return tb_logger

    @staticmethod
    def create_trainer(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module):
        def padding_train_step(trainer, batch):
            model.train()
            optimizer.zero_grad()
            x, y, original_lengths = batch  # prepare_batch(batch)

            y_pred = model(x, original_lengths)
            # Fix for non-deterministic CrossEntropyLoss on GPU
            y_pred = y_pred.to('cpu')
            y = y.to('cpu')
            loss = criterion(y_pred, y)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            return loss.item()

        return Engine(padding_train_step)

    @staticmethod
    def create_evaluator(model: nn.Module, metrics: Dict[str, Metric]):
        def padding_evaluation_step(engine: Engine, batch: Sequence[torch.Tensor]):
            model.eval()
            with torch.no_grad():
                x, y, original_lengths = batch
                y_pred = model(x, original_lengths)
                # transform the output (similar to the output_transform from the PyTorch Ignite
                # supervised_evaluation_step: lambda x, y, y_pred: (y_pred, y)
                return y_pred, y

        evaluator = Engine(padding_evaluation_step)
        Annotator.attach_metrics(metrics=metrics, engine=evaluator)
        return evaluator

    @staticmethod
    def attach_metrics(metrics: Dict[str, Metric], engine: Engine):
        for name, metric in metrics.items():
            metric.attach(engine, name)

    @staticmethod
    def annotate_dataset(dataset: CoNLLDataset, model: nn.Module):
        all_predicted_labels = Annotator.predict_labels(dataset=dataset, model=model)
        annotated_dataset = deepcopy(dataset)
        i = 0
        for doc_index, document in enumerate(dataset.documents):
            for sentence_index, sentence in enumerate(document.sentences):
                predicted_labels = all_predicted_labels[i]
                for token_index, label in enumerate(predicted_labels):
                    document_to_annotate = annotated_dataset.documents[doc_index]
                    sentence_to_annotate = document_to_annotate.sentences[sentence_index]
                    token = sentence_to_annotate.tokens[token_index]
                    token.tag = label
                i += 1
        return annotated_dataset

    @staticmethod
    def predict_labels(dataset: CoNLLDataset, model: nn.Module):
        all_predicted_labels = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            data_loader = CoNLLDataLoader(dataset=dataset, shuffle=False, num_workers=0,
                                          batch_size=128, collate_fn=collate_batch)
            for batch in data_loader:
                x, y, original_lengths = batch
                y_pred = model(x, original_lengths)
                predicted_batch_indices = torch.argmax(y_pred, dim=1)
                predicted_batch_indices = [predicted_indices[:original_lengths[index]] for index, predicted_indices in
                                           enumerate(predicted_batch_indices)]
                predicted_labels = _create_BIO2_labels_from_batch_indices(predicted_batch_indices,
                                                                          ignore_index=ignore_label_index)

                for predicted_sentence_labels in predicted_labels:
                    all_predicted_labels.append(predicted_sentence_labels)
        return all_predicted_labels
