import abc
import logging
from typing import List, Tuple

import numpy as np
import yaml
from keras.utils import np_utils
from sklearn import model_selection

from alphai_es_datasource.wf.definitions import WF_COLUMNS, WF_EXTRA_COLUMNS, WF_STRING_COLUMNS
from alphai_es_datasource.utils import ModelData, Metadata, Chunk


class AbstractTransformer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def transform(self, data: List[Chunk]) -> ModelData:
        """
        Transforms a list of chunks into a piece of data that the model can use.
        """
        raise NotImplementedError


class SimpleTransformer(AbstractTransformer):
    DROP_COLS = ['wind_turbine', 'label', 'IsFaulty']

    def __init__(self, reshape_x=True, transpose_x=False, normalise_x=True):
        self.reshape_x = reshape_x
        self.transpose_x = transpose_x
        self.normalise_x = normalise_x

    def transform(self, data: List[Chunk], metadata: Metadata = None) -> ModelData:
        X_ = []
        y_ = []

        # TODO
        mean = 0
        std = 0

        for i in data:
            X_.append(i.data.drop(columns=self.DROP_COLS).values)
            y_.append(i.label.value)

        X_ = np.array(X_)
        y_ = np.array(y_)

        logging.debug(f"Shape at at 110: {X_.shape}, {y_.shape}")
        X__ = X_

        if self.reshape_x:
            new_shape = X_.shape[1] * X_.shape[2]
            X__ = X__.reshape((-1, new_shape))

        if self.normalise_x:
            new_shape = X_.shape[1] * X_.shape[2]
            X__ = X__.reshape((-1, new_shape))
            if not metadata:
                mean = np.mean(X__, axis=0)
                std = np.std(X__, axis=0)
            else:
                mean = metadata.mean
                std = metadata.std
            X__ = (X__ - mean) / std
            X__ = X__.reshape((-1, X_.shape[1], X_.shape[2]))

        if self.transpose_x:
            X__ = np.transpose(X__, (0, 2, 1))

        y__ = y_

        return ModelData(data=X__, labels=y__, metadata=Metadata(mean=mean, std=std))


class ModelInfo:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            self.model_info = yaml.load(f)

    def all_variables(self):
        model_vars = set(WF_COLUMNS) - set(WF_EXTRA_COLUMNS) - set(WF_STRING_COLUMNS)
        return tuple(sorted(list(model_vars)))


class Splitter:

    @staticmethod
    def train_test_split(data: ModelData, test_size=0.2, random_state=1337) -> Tuple[ModelData, ModelData]:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data.data, data.labels, test_size=test_size,
                                                                            random_state=random_state)
        return (ModelData(x_train, np_utils.to_categorical(y_train), data.metadata),
                ModelData(x_test, np_utils.to_categorical(y_test), data.metadata))
