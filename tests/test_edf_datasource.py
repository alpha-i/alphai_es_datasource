import datetime

import pandas as pd
import pytest

from alphai_es_datasource.wf import WFDataSource
from alphai_es_datasource.utils import Labels, PlainChunk


@pytest.mark.skip
def test_wf_can_get_data():
    wf_data = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=datetime.datetime(2015, 3, 1),
        end_date=datetime.datetime(2016, 2, 29),
        chunk_size=72,
        stride=72,
        abnormal_window_duration=24,
        turbines=(1,),
        fields=('ActiveEnergyExport', 'WpsStatus'),
    )

    train_data, validation_data, test_data = wf_data.get_split_data(Labels.BOTH)

    for idx, chunk in enumerate(train_data):
        # check that no chunk contains a gap
        gaps = chunk.data.index.to_series().diff().unique()
        assert len(gaps) == 2, f"chunk {idx} contains a gap!"

    assert len(train_data) > len(test_data) > len(validation_data)

    # Data contains a fault between 09:30 and 12:20 of 2015-08-07
    faulty_intervals = wf_data._get_faulty_intervals()

    assert len(faulty_intervals) == 1
    assert faulty_intervals[1][0].start == pd.Timestamp('2015-08-07 08:30:00')
    assert faulty_intervals[1][0].end == pd.Timestamp('2015-08-07 11:20:00')

    # last interval ends
    assert faulty_intervals[1][-1].start == pd.Timestamp('2016-02-03 10:20:00')

    normal_data = wf_data.get_normal_data()

    # check that a faulty interval is not in the normal data
    assert pd.Timestamp('2015-08-07 08:30:00') not in normal_data[1].index
    assert pd.Timestamp('2015-08-07 07:30:00') not in normal_data[1].index
    # normal data starts a day before the faulty (cause window is 1 day)
    assert pd.Timestamp('2015-08-06 07:20:00') in normal_data[1].index

    abnormal_data = wf_data.get_abnormal_data()

    assert abnormal_data[1].iloc[0].name == pd.Timestamp('2015-08-06 08:30:00')
    assert abnormal_data[1].iloc[-1].name == pd.Timestamp('2016-02-03 10:10:00')


def test_wf_ds_can_return_chunks():
    wf_data = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=datetime.datetime(2015, 3, 1),
        end_date=datetime.datetime(2016, 2, 29),
        chunk_size=10,
        stride=72,
        abnormal_window_duration=24,
        turbines=(1,),
        fields=('ActiveEnergyExport', 'WpsStatus'),
    )

    chunks = wf_data.get_list_of_chunks()

    # it must contain data from a single turbine
    assert len(chunks) == 1

    # it must contain more than one chunk of data
    assert len(chunks[1]) > 0

    chunk = chunks[1][0]

    # the chunks order must always be the same
    assert chunk.starts_at == pd.Timestamp('2015-06-23 04:00:00')
    assert chunk.ends_at == pd.Timestamp('2015-06-23 05:30:00')
    assert chunk.next_fault_at == pd.Timestamp('2015-08-07 08:30:00')

    # data will have four columns (ActiveEnergyExport, IsFaulty, WpsStatus, wind_turbine)
    # and 10 timepoints (as per chunk size)
    assert chunk.data.shape == (10, 4)

    # check that the chunks are ordered chronologically
    assert chunks[1][0].data.iloc[0].name < chunks[1][0].data.iloc[-1].name

    # check that the chunks contain no gaps
    for idx, chunk in enumerate(chunks[1]):
        gaps = chunk.data.index.to_series().diff().unique()
        assert len(gaps) == 2, f"chunk {idx} contains a gap!"


def test_wf_data_bug():
    passwf_data = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=datetime.datetime(2015, 8, 8),
        end_date=datetime.datetime(2016, 2, 28),
        chunk_size=10,
        stride=72,
        abnormal_window_duration=240,
        turbines=(2,),
        fields=('ActiveEnergyExport', 'WpsStatus'),
    )
    list_of_chunks = passwf_data.get_list_of_chunks()
    assert list_of_chunks


def test_wf_data_bug_again():
    passwf_data = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist',
        start_date=datetime.datetime(2016, 1, 1),
        end_date=datetime.datetime(2016, 2, 28),
        chunk_size=144,
        stride=143,
        abnormal_window_duration=240,
        turbines=(1,2,3),
        fields=('ActiveEnergyExport', 'WpsStatus'),
    )
    list_of_chunks = passwf_data.get_list_of_chunks()
    for turbine, chunks in list_of_chunks.items():
        print(f"Listing turbine {turbine}")
        for idx, chunk in enumerate(chunks):
            if not chunk.next_fault_at:
                continue
            distance_to_fault = chunk.next_fault_at - chunk.starts_at
            print(f"At index {idx} for {turbine}, distance to fault is {distance_to_fault}")
    assert list_of_chunks


def test_get_naive_chunks():
    passwf_data = WFDataSource(
        host='51.144.39.71:9200',
        index_name='wf_scada_hist_no_interpolation',
        start_date=datetime.datetime(2016, 1, 1),
        end_date=datetime.datetime(2016, 2, 28),
        chunk_size=144,
        stride=143,
        abnormal_window_duration=240,
        turbines=(1,),
        fields=('ActiveEnergyExport', 'WpsStatus'),
    )

    chunks = passwf_data.get_naive_chunks()

    assert len(chunks[1])
