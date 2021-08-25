from argparse import Namespace

from torch import nn

from bioner.model.datexis_model import DATEXISModel


class LayerConfiguration:
    def __init__(self, input_vector_size: int):
        self.input_vector_size = input_vector_size


class DATEXISNERLayerConfiguration(LayerConfiguration):
    def __init__(self, input_vector_size: int,
                 feedforward_layer_size: int = 150,
                 lstm_layer_size: int = 20,
                 out_features: int = 3):
        super().__init__(input_vector_size=input_vector_size)
        self.feedforward_layer_size = feedforward_layer_size
        self.lstm_layer_size = lstm_layer_size
        self.out_features = out_features


class LayerConfigurationCreator:
    @staticmethod
    def create_layer_configuration(input_vector_size: int, args: Namespace):
        if args.ff1 is not None and args.lstm1 is not None:
            return DATEXISNERLayerConfiguration(input_vector_size=input_vector_size,
                                                feedforward_layer_size=args.ff1,
                                                lstm_layer_size=args.lstm1)
        if args.ff1 is not None:
            return DATEXISNERLayerConfiguration(input_vector_size=input_vector_size,
                                                feedforward_layer_size=args.ff1)
        if args.lstm1 is not None:
            return DATEXISNERLayerConfiguration(input_vector_size=input_vector_size,
                                                lstm_layer_size=args.lstm1)
        return LayerConfiguration(input_vector_size=input_vector_size)


class ModelLoader:
    @staticmethod
    def load_model(name: str, layer_configuration: LayerConfiguration) -> nn.Module:
        if name == "DATEXIS-NER":
            return ModelLoader.create_original_datexis_ner_model(
                input_vector_size=layer_configuration.input_vector_size)
        if name == "CustomConfig_DATEXIS-NER":
            return ModelLoader.create_custom_datexis_ner_model(layer_configuration=layer_configuration)

    @staticmethod
    def create_original_datexis_ner_model(input_vector_size: int) -> DATEXISModel:
        """
        Creates the original DATEXIS-NER model from the paper:
        Robust Named Entity Recognition in Idiosyncratic Domains (https://arxiv.org/abs/1608.06757)
        :param input_vector_size: the size of the embeddings
        """
        return DATEXISModel(input_vector_size=input_vector_size)

    @staticmethod
    def create_custom_datexis_ner_model(layer_configuration: DATEXISNERLayerConfiguration) -> DATEXISModel:
        """
        Creates the original DATEXIS-NER model from the paper:
        Robust Named Entity Recognition in Idiosyncratic Domains (https://arxiv.org/abs/1608.06757)
        but with a custom layer configuration
        :param layer_configuration: the custom layer configuration for the model
        """
        return DATEXISModel(input_vector_size=layer_configuration.input_vector_size,
                            feedforward_layer_size=layer_configuration.feedforward_layer_size,
                            lstm_layer_size=layer_configuration.lstm_layer_size,
                            out_features=layer_configuration.out_features)
