from collections import namedtuple
from enum import Enum
from typing import List

FaultInterval = namedtuple('FaultInterval', 'start end duration')
ServiceInterval = namedtuple('ServiceInterval', 'start end duration')

Chunk = namedtuple('Chunk', 'data label')
PlainChunk = namedtuple('PlainChunk', 'data starts_at ends_at next_fault_at')
NaiveChunk = namedtuple('NaiveChunk', 'data starts_at ends_at')

Prediction = namedtuple("Prediction", "probabilities classes")
EvaluationStats = namedtuple("EvaluationStats", "precision recall f1_score support")
ModelData = namedtuple('ModelData', 'data labels metadata')
Metadata = namedtuple('Metadata', 'mean std')
ModelPrediction = namedtuple('ModelPrediction', 'component timestamp abnormality prediction')


class Labels(Enum):
    NORMAL = 0
    ABNORMAL = 1
    BOTH = 3
    UNKNOWN = 4  # use this for incoming data


class RawData:
    def __init__(self, normal: List[Chunk], abnormal: List[Chunk]):
        self.normal = normal
        self.abnormal = abnormal

    @property
    def both(self) -> List[Chunk]:
        return self.normal + self.abnormal


class ModelPredictionList(list):
    def __init__(self, component: str, data: List, prediction: Prediction):
        self._data = data
        self._component = component
        self._prediction = prediction
        self._timestamps = self.calculate_prediction_points(self._data)
        self.prediction_with_timestamps = zip(self._timestamps, self._prediction.probabilities,
                                              self._prediction.classes)
        self._inner_list = []
        for timestamp, probability, label_value in self.prediction_with_timestamps:
            self._inner_list.append(
                ModelPrediction(self._component, timestamp, probability, Labels(label_value)),
            )

        super(ModelPredictionList, self).__init__(self._inner_list)

    def calculate_prediction_points(self, chunks: List[Chunk]):
        timestamps = []
        for chunk in chunks:
            start = chunk.data.iloc[0].name
            end = chunk.data.iloc[-1].name
            duration = end - start
            half_duration = duration / 2
            half_point = start + half_duration
            timestamps.append(half_point)

        return timestamps
