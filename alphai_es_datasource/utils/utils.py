from datetime import datetime, timedelta
from typing import Sized, List

import pandas as pd



def hits_to_dataframe(hits):
    try:
        hits = [hit.to_dict() for hit in hits]
    except Exception as e:
        pass

    dataframe = pd.DataFrame(hits)

    if len(dataframe):
        dataframe['timestamp'] = dataframe['timestamp'].apply(
            lambda x: datetime.utcfromtimestamp(x / 1000)
        )

        dataframe = dataframe.set_index('timestamp').sort_index()

    return dataframe


def percentage_split(sequence: Sized, percentages: List[float]):
    assert sum(percentages) == 1.0, f"Invalid percentage list: {percentages}"
    previous = 0
    sequence_size = len(sequence)
    cumulative_percentage = 0
    for percentage in percentages:
        cumulative_percentage += percentage
        next_ = int(cumulative_percentage * sequence_size)
        yield sequence[previous:next_]
        previous = next_


def create_chunks_from_dataframe(dataframe, chunk_size, stride, df_resolution):
    """
    :param pd.DataFrame dataframe: data
    :param int chunk_size:
    :param int stride:
    :param int df_resolution: dataframe resolution in seconds

    :return generator:
    """

    dataframe = dataframe.dropna()

    start = dataframe.index[0]
    end = dataframe.index[-1]

    chunk_delta = timedelta(seconds=df_resolution * (chunk_size - 1))
    stride_delta = timedelta(seconds=df_resolution * stride)

    while start < end:

        the_chunk = dataframe.loc[start: start + chunk_delta]
        if len(the_chunk.index) == chunk_size:
            yield the_chunk

        start = start + stride_delta




