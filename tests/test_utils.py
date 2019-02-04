import os
from datetime import timedelta

import pandas as pd

from alphai_es_datasource.utils.utils import create_chunks_from_dataframe

CURRENT_DIR = os.path.dirname(__file__)

RESOURCE_DIR = os.path.join(CURRENT_DIR, 'resources')


def _create_dataframe():
    sample_dataframe = pd.read_csv(os.path.join(RESOURCE_DIR, 'data.csv'))
    sample_dataframe['timestamp'] = sample_dataframe['timestamp'].apply(lambda x: pd.Timestamp(x))
    return sample_dataframe.set_index('timestamp')


def test_create_chunks_from_dataframe_completed():
    sample_dataframe = _create_dataframe()
    chunk_size = 5
    stride = 1
    resolution = 600

    chunk_iterator = create_chunks_from_dataframe(sample_dataframe, chunk_size, stride, resolution)

    chunk_list = list(chunk_iterator)

    assert len(chunk_list) == 196
    for chunk in chunk_list:
        assert len(chunk.index) == chunk_size

    for left, right in zip(chunk_list[0:-1], chunk_list[1:]):

        deltas = right.index[0] - left.index[0]
        assert deltas == pd.Timedelta(timedelta(seconds=resolution*stride))


def test_create_chunks_from_dataframe_with_gaps():

    sample_dataframe = _create_dataframe()

    half_hour_gap = sample_dataframe.loc['2017-07-30 04:40:00':'2017-07-30 05:10:00'].index

    sample_dataframe = sample_dataframe.drop(index=half_hour_gap)

    chunk_size = 5
    stride = 1
    resolution = 600

    chunk_iterator = create_chunks_from_dataframe(sample_dataframe, chunk_size, stride, resolution)

    chunk_list = list(chunk_iterator)

    assert len(chunk_list) == 188

    sample_dataframe = _create_dataframe()

    two_hour_gap = sample_dataframe.loc['2017-07-30 22:10:00':'2017-07-31 00:10:00'].index
    sample_dataframe = sample_dataframe.drop(index=two_hour_gap)

    print('#'*10)
    chunk_iterator = create_chunks_from_dataframe(sample_dataframe, chunk_size, stride, resolution)

    chunk_list = list(chunk_iterator)

    assert len(chunk_list) == 179

    sample_dataframe = _create_dataframe()
    sample_dataframe = sample_dataframe.drop(index=[pd.Timestamp('2017-07-30 22:10:00')])

    assert sample_dataframe.shape == (199, 7)

    chunk_iterator = create_chunks_from_dataframe(sample_dataframe, chunk_size, stride, resolution)
    chunk_list = list(chunk_iterator)

    assert len(chunk_list) == 191


def test_create_chunks_from_dataframe_with_gaps_different_strides():

    sample_dataframe = _create_dataframe()

    half_hour_gap = sample_dataframe.loc['2017-07-30 04:40:00':'2017-07-30 05:10:00'].index

    sample_dataframe = sample_dataframe.drop(index=half_hour_gap)

    chunk_size = 6
    stride = 2
    resolution = 600

    chunk_iterator = create_chunks_from_dataframe(sample_dataframe, chunk_size, stride, resolution)

    chunk_list = list(chunk_iterator)

    assert len(chunk_list) == 94
