import datetime
import itertools
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

from alphai_es_datasource.utils import (
    FaultInterval, Chunk, PlainChunk, NaiveChunk, Labels, hits_to_dataframe
)


class ElasticSearchConnector:
    def __init__(self, host: str, index_name: str, *args, **kwargs):
        self.search = Search(using=Elasticsearch(hosts=host), index=index_name)

    def get_window(self, turbines: tuple, start_date: datetime.datetime, end_date: datetime.datetime, fields=('*',)):
        self.search = self.search.filter("range", timestamp={
            "gte": start_date,
            "lt": end_date})
        self.search = self.search.filter(
            "terms", wind_turbine=turbines,
        )

        fields = ('timestamp',) + fields
        self.search = self.search.source(include=fields)
        self.search = self.search.sort('timestamp')

        hits = list(self.search.scan())
        return hits_to_dataframe(hits)

    def get_plain_data(self, turbines: tuple, start_date: datetime.datetime, end_date: datetime.datetime,
                       fields=('*',)):
        self.search = self.search.filter("range", timestamp={
            "gte": start_date,
            "lt": end_date})
        self.search = self.search.filter(
            "terms", wind_turbine=turbines,
        )

        fields = ('timestamp',) + fields
        self.search = self.search.source(include=fields)
        self.search = self.search.sort('timestamp')

        hits = list(self.search.scan())
        return hits_to_dataframe(hits)


class WFDataSource:
    # time window of abnormal data before a fault occurs.
    # should actually be two WEEKS, but let's keep it small
    # for investigation purposes
    FAULTY_COLUMN = 'IsFaulty'
    TURBINE_NUMBER_FIELD = 'wind_turbine'
    RESOLUTION = pd.Timedelta(minutes=10)

    def __init__(self, host: str, index_name: str,
                 start_date: datetime.datetime, end_date: datetime.datetime,
                 abnormal_window_duration: int, chunk_size: int, stride: int,
                 turbines: tuple, fields=('*',), *args, **kwargs):
        """
        :param host: the elasticsearch host (in `ip:port` format)
        :param index_name: the data index
        :param start_date: start of the query window
        :param end_date: end of the query window
        :param chunk_size: how many timestamps will be included in each chunk
        :param stride: the stride allows us to get overlapping chunks of data
        :param turbines: select the turbine number (or numbers)
        :param fields: the fields to select, defaults to '*' (all)
        """

        self.es_connector = ElasticSearchConnector(host=host, index_name=index_name)

        self.chunk_size = chunk_size
        self.stride = stride
        self.RANDOM_SEED = 5
        self.abnormal_window_duration = datetime.timedelta(hours=abnormal_window_duration)

        self.turbines = turbines
        self.start_date = start_date
        self.end_date = end_date
        self.fields = fields

        self._data = self.es_connector.get_window(
            turbines=turbines,
            start_date=start_date,
            end_date=end_date,
            fields=fields + (self.FAULTY_COLUMN, self.TURBINE_NUMBER_FIELD)
        )

        self._grouped = {number: df for number, df in self._data.groupby(self.TURBINE_NUMBER_FIELD)}

    def get_plain_data(self):
        """
        Utility method to return a plain view of the window
        with no chunks and no filtering of good and bad chunks
        """
        return self.es_connector.get_plain_data(
            turbines=self.turbines,
            start_date=self.start_date,
            end_date=self.end_date,
            fields=self.fields + (self.FAULTY_COLUMN, self.TURBINE_NUMBER_FIELD)
        )

    def _get_faulty_intervals(self):
        """
        In order to distinguish between normal and abnormal data we have to get out the "faulty" data
        (i.e. when a turbine was shutdown outside of a planned maintenance), and consider the `abnormal_window`
        interval before the fault as being abnormal (because it led to a malfunction).

        :return: list of faults intervals, with start and end timestamps
        """

        def get_interval(dataframe):
            faulty_data = dataframe[getattr(dataframe, self.FAULTY_COLUMN) == 1].copy()

            faulty_timestamps = faulty_data.index.to_series()
            faulty_diffs = faulty_data.index.to_series().diff()

            faulty_intervals = [
                FaultInterval(start=faulty_diffs.index[0], end=faulty_timestamps[1:][0], duration=self.RESOLUTION)
            ]

            for idx, diff in enumerate(faulty_diffs[1:]):
                if diff != self.RESOLUTION:
                    # we found a new start
                    faulty_intervals.append(
                        FaultInterval(start=faulty_timestamps[1:][idx], end=faulty_timestamps[1:][idx], duration=self.RESOLUTION)
                    )
                else:
                    # we keep going until we get to the end of the interval
                    partial_fault = faulty_intervals[-1]
                    updated_fault = FaultInterval(start=partial_fault.start, end=faulty_timestamps[1:][idx], duration=faulty_timestamps[1:][idx]-partial_fault.start)
                    faulty_intervals[-1] = updated_fault


            return faulty_intervals

        interval_gatherer = defaultdict(list)
        for turbine_number, group in self._grouped.items():
            intervals = get_interval(group)
            interval_gatherer[turbine_number] = intervals

        return interval_gatherer

    def _exclude_faulty_data(self):
        """
        Filter out faulty data from the full window of data
        We also need to exclude the data collected during an annual service
        """
        faulty_intervals = self._get_faulty_intervals()

        # given the intervals, progressively
        # amend the original data from intervals of fault
        good_data = defaultdict(list)

        for turbine, intervals in faulty_intervals.items():
            whole_turbine_data = self._grouped[turbine].copy()
            for interval in intervals:
                good_data_for_turbine = whole_turbine_data.loc[
                    (whole_turbine_data.index < interval.start) |
                    (whole_turbine_data.index > interval.end)
                    ].copy()
                # we progressively take out faulty data from the set
                whole_turbine_data = good_data_for_turbine
            good_data[turbine] = whole_turbine_data

        return good_data

    def _deduplicate_data(self, data_dict):
        returned = {}
        for turbine_number, data in data_dict.items():
            dataf = pd.concat(data)
            # we're going to get duplicates here
            # because additional faults can occur in the interval window
            dataf = dataf[~dataf.index.duplicated(keep='first')]
            returned[turbine_number] = dataf
        return returned

    def get_abnormal_data(self):
        """
        Only get the abnormal data from the full data window
        """
        fault_intervals = self._get_faulty_intervals()
        non_faulty_data = self._exclude_faulty_data()

        abnormal_data = defaultdict(list)
        for turbine, intervals in fault_intervals.items():
            turbine_data = non_faulty_data[turbine].copy()
            for interval in intervals:
                # for each interval, fetch the `abnormal_window_size` period before the start
                abnormal_start = interval.start - self.abnormal_window_duration
                abnormal_interval = turbine_data.loc[
                    (turbine_data.index >= abnormal_start) &
                    (turbine_data.index < interval.start)
                    ].copy()
                abnormal_interval['label'] = Labels.ABNORMAL.value
                abnormal_data[turbine].append(
                    abnormal_interval
                )

        if not abnormal_data:
            return pd.DataFrame()

        return self._deduplicate_data(abnormal_data)

    def get_normal_data(self):
        """
        Filter out abnormal data from the non-faulty data
        """
        non_faulty_data = self._exclude_faulty_data()
        abnormal_data = self.get_abnormal_data()

        normal_data = defaultdict(list)
        for turbine_number, data in non_faulty_data.items():
            abnormal_indexes = abnormal_data[turbine_number].index
            normal_interval = data.loc[data.index.difference(abnormal_indexes)].copy()
            normal_interval['label'] = Labels.NORMAL.value
            normal_data[turbine_number].append(
                normal_interval
            )

        return self._deduplicate_data(normal_data)

    def get_test_data(self) -> List[Chunk]:
        """
        Same logic of get_all_data, but it shouldn't take into account any labelling
        Just chunk the data from the start, according to chunk sizes and stride.
        """
        all_data = self._grouped
        chunks_by_turbine = self._chunks_of_type(all_data, Labels.UNKNOWN)
        return list(itertools.chain.from_iterable(chunks_by_turbine.values()))

    def _chunks_of_type(self, data, label) -> Dict[int, List[Chunk]]:
        # split the original data in chunks
        list_of_chunks = defaultdict(list)
        for turbine_number, turbine_data in data.items():
            # need to group by continuous segments of data
            # to avoid crossing faulty lines
            diffs = turbine_data.index.to_series().diff()
            discontinuity_starts = diffs.loc[diffs != self.RESOLUTION].index

            groups = []
            start_at = turbine_data.index[0]
            for start in discontinuity_starts:
                group = turbine_data.loc[
                    (turbine_data.index >= start_at) & (turbine_data.index < start)
                    ].copy()
                groups.append(group)
                start_at = start

            # Corner case: we don't have gaps in the whole data set
            # so the only "discontinuity" is the NaT at the start of the series

            if len(discontinuity_starts) == 1 and discontinuity_starts[0] == turbine_data.index[0]:
                turbine_data['label'] = label.value
                groups = [turbine_data]

            for group in groups:
                # split the indexes depending on chunk and stride
                length = len(group.index)
                indexes = list(range(length))
                chunk_starting_points = indexes[
                                        0::self.stride + 1]  # +1 is needed to force the boundary to be inclusive
                chunks = [list(range(starting_point, starting_point + self.chunk_size + 1))
                          for starting_point in chunk_starting_points]

                # shuffle the chunks indexes list
                random.shuffle(chunks, random=lambda: 0.1)

                for chunk in chunks:
                    this_chunk = group.iloc[chunk[0]:chunk[-1]]
                    if len(this_chunk) < self.chunk_size:
                        # we'll discard an incomplete chunk
                        continue
                    list_of_chunks[turbine_number].append(Chunk(data=this_chunk, label=label))

        return list_of_chunks

    def _get_chunks(self) -> Tuple[Dict[int, List[Chunk]], Dict[int, List[Chunk]]]:
        """
        Gives back a shuffled list of data chunks
        """

        # get the requested data type
        normal_data = self.get_normal_data()
        abnormal_data = self.get_abnormal_data()

        random.seed(self.RANDOM_SEED)


        return self._chunks_of_type(normal_data, Labels.NORMAL), self._chunks_of_type(abnormal_data, Labels.ABNORMAL)

    def get_groups(self):
        data = self._exclude_faulty_data()

        groups = defaultdict(list)
        for turbine_number, turbine_data in data.items():
            # need to group by continuous segments of data
            # to avoid crossing faulty lines
            diffs = turbine_data.index.to_series().diff()
            discontinuity_starts = diffs.loc[diffs != self.RESOLUTION].index
            start_at = turbine_data.index[1]  # the first element will always be the first timepoint (NaT diff)
            for start in discontinuity_starts:
                group = turbine_data.loc[
                    (turbine_data.index >= start_at) & (turbine_data.index < start)
                    ].copy()
                if len(group):
                    groups[turbine_number].append(group)
                start_at = start
            else:
                # we need to add the rest of the stuff
                # after the last discontinuity
                group = turbine_data.loc[turbine_data.index >= start_at].copy()
                groups[turbine_number].append(group)

        return groups

    def get_all_data(self) -> List:
        """
        It mixes together abnormal and normal data for all the turbines
        """

        normal, abnormal = self._get_chunks()
        return list(itertools.chain(*[normal[turbine] + abnormal[turbine] for turbine in self.turbines]))

    def get_list_of_chunks(self) -> Dict[int, List[PlainChunk]]:
        """
        Here we want to give back a list of chunks going backwards from the anomaly
        including information about the timestamp of the fault laying "in front of" them
        """

        groups_of_good_data = self.get_groups()
        fault_intervals = self._get_faulty_intervals()

        turbine_chunks = defaultdict(list)

        for turbine_number, turbine_groups_of_good_data in groups_of_good_data.items():
            intervals = fault_intervals[turbine_number] + [None]
            for group in turbine_groups_of_good_data:
                for interval in intervals:
                    if not interval:
                        fault_starts_at = None
                        continue
                    if group.index[0] > interval.end:
                        continue
                    else:
                        fault_starts_at = interval.start
                        break

                # here we have an interval which starts before

                backwards_group = group.iloc[::-1].copy()
                length = len(backwards_group.index)
                indexes = list(range(length))
                chunk_starting_points = indexes[0::self.stride + 1]  # +1 is needed to force the boundary to be inclusive
                chunks = [list(range(starting_point, starting_point + self.chunk_size + 1))
                          for starting_point in chunk_starting_points]
                random.shuffle(chunks, random=lambda: 0.1)

                for chunk in chunks:

                    this_chunk = backwards_group.iloc[chunk[0]:chunk[-1]]
                    if len(this_chunk) < self.chunk_size:
                        # we'll discard an incomplete chunk
                        continue
                    this_chunk_reversed = this_chunk.iloc[::-1].copy()
                    turbine_chunks[turbine_number].append(
                        PlainChunk(
                            data=this_chunk_reversed,
                            starts_at=this_chunk_reversed.index[0],
                            ends_at=this_chunk_reversed.index[-1],
                            next_fault_at=fault_starts_at
                        )
                    )

        return turbine_chunks

    def get_naive_chunks(self):
        """
        This will return a simple list of chunks, regardless of any discountinuity or any supposed abnormal / normal
        distinction. If a chunk contains a NaN anywhere (for missing data in a variable), just throw it away
        """

        whole_data_grouped = self._grouped

        aggregator = defaultdict(list)
        for turbine_number, turbine_data in whole_data_grouped.items():
            length = len(turbine_data.index)
            indexes = list(range(length))
            chunk_starting_points = indexes[0::self.stride + 1]  # +1 is needed to force the boundary to be inclusive
            chunks = [list(range(starting_point, starting_point + self.chunk_size + 1))
                      for starting_point in chunk_starting_points]

            for chunk in chunks:
                this_chunk = turbine_data.iloc[chunk[0]:chunk[-1]]
                if len(this_chunk) < self.chunk_size:
                    # we'll discard an incomplete chunk
                    continue
                # check if there's any NaN in here before appending
                # otherwise call a nice `continue`
                if this_chunk.isnull().sum().sum() > 0:
                    continue
                aggregator[turbine_number].append(
                    NaiveChunk(
                        data=this_chunk,
                        starts_at=this_chunk.index[0],
                        ends_at=this_chunk.index[-1],
                    )
                )

        return aggregator
