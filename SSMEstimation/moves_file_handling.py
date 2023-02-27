from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from typing import List

import data_writer
import file_handling
from post_processing import drop_warmup_samples
import readers

_km_to_mile = 0.6213712

# TODO [Feb 23]: somewhere we must consider the possibility of merging link
#  segments


def get_individual_vehicle_trajectories_to_moves(
        scenario_name: str,
        scenario_info: file_handling.ScenarioInfo,
        first_minute: int = 15,
        vehs_per_simulation: int = 1):
    """
    Finds the first vehicle entering the simulation link after the cut-off
    minute and save its info in MOVES format
    """
    file_handler = file_handling.FileHandler(scenario_name)
    main_link_numbers = file_handler.get_main_links()
    first_link_id = main_link_numbers[0]
    last_minute = 20
    first_second = first_minute * 60
    interval = (last_minute - first_minute) * 60 / vehs_per_simulation
    link_id_map = dict(zip(main_link_numbers, range(len(main_link_numbers))))

    link_eval_reader = readers.LinkEvaluationReader(scenario_name)
    link_eval_data = link_eval_reader.load_data_from_scenario(scenario_info)
    link_eval_data = link_eval_data[link_eval_data['link_number'].isin(
        main_link_numbers)]

    vehicle_record_reader = readers.VehicleRecordReader(scenario_name)
    data_generator = vehicle_record_reader.generate_all_data_from_scenario(
        scenario_info)
    link_data_list = []
    speed_data_list = []
    init_link_number = 0
    for (vehicle_records, _) in data_generator:
        vehicle_records['secondID'] = np.floor(vehicle_records['time']).astype(
            int)
        vehicle_records = vehicle_records[vehicle_records['link'].isin(
            main_link_numbers)]
        for i in range(vehs_per_simulation):
            cut_off_second = int(first_second + i * interval)
            candidates = vehicle_records.loc[
                (vehicle_records['time'] == cut_off_second)
                & (vehicle_records['link'] == first_link_id)]
            veh_id = candidates.loc[candidates['x'] == candidates['x'].min(),
                                    'veh_id'].iloc[0]
            vehicle_data = vehicle_records[vehicle_records['veh_id'] == veh_id]

            single_veh_link_data = (
                _create_link_evaluation_data_for_single_vehicle(
                    link_eval_data, vehicle_data))
            single_veh_link_data['link_id'] = (
                    single_veh_link_data['link_number'].map(link_id_map)
                    + init_link_number)
            link_data_list.append(single_veh_link_data)

            single_veh_speed_data = _create_moves_speeds_from_vehicle_record(
                vehicle_data, main_link_numbers)
            single_veh_speed_data['secondID'] -= cut_off_second
            single_veh_speed_data['linkID'] = (
                    single_veh_speed_data['linkID'].map(link_id_map)
                    + init_link_number)

            init_link_number = single_veh_link_data['link_id'].max() + 1
            speed_data_list.append(single_veh_speed_data)

    moves_links_processor = MovesLinksProcessor(scenario_name)
    moves_link_data = moves_links_processor.process(
        pd.concat(link_data_list))

    moves_link_source_processor = MovesLinkSourceProcessor(scenario_name)
    moves_link_source_data = moves_link_source_processor.process(
        moves_link_data)

    moves_speed_data = pd.concat(speed_data_list)


    # Save all files
    link_writer = data_writer.MOVESLinksWriter(scenario_name)
    link_writer.save_data(moves_link_data, scenario_info)
    link_source_writer = data_writer.MOVESLinkSourceWriter(scenario_name)
    link_source_writer.save_data(moves_link_source_data, scenario_info)
    drive_sched_writer = data_writer.MOVESLinkDriveWriter(scenario_name)
    drive_sched_writer.save_data(moves_speed_data, scenario_info)


# def translate_links_from_vissim_to_moves_old(
#         scenario_name: str,
#         scenario_info: file_handling.ScenarioInfo,
#         links: List[int] = None, warmup_minutes: int = 10):
#     """
#     Reads link evaluation output files from VISSIM and write link,
#     link source and drive schedule xls files for use in MOVES
#     """
#
#     # Load VISSIM data
#     link_evaluation_reader = readers.LinkEvaluationReader(scenario_name)
#     link_evaluation_data = (
#         link_evaluation_reader.load_all_data_from_scenario(scenario_info))
#     drop_warmup_samples(link_evaluation_data, warmup_minutes)
#
#     if links is not None:
#         link_evaluation_data.drop(
#             index=link_evaluation_data[
#                 ~link_evaluation_data['link_number'].isin(links)].index,
#             inplace=True
#         )
#
#     # We will pretend each simulation is a new set of links to avoid having to
#     # run MOVES over and over again.
#     n_links_per_simulation = len(link_evaluation_data['link_number'].unique())
#     link_evaluation_data['link_id'] = (
#             (link_evaluation_data['simulation_number'] - 1)
#             * n_links_per_simulation + link_evaluation_data['link_number'])
#     link_evaluation_data.sort_values('link_id', kind='stable', inplace=True,
#                                      ignore_index=True)
#
#     # Aggregated data files
#     aggregated_link_data = link_evaluation_data.groupby('link_id')[[
#         'link_id', 'segment_length', 'volume', 'average_speed']].mean()
#     moves_link_data = _fill_moves_link_data_from_link_evaluation_data(
#         scenario_name, aggregated_link_data)
#     link_writer = data_writer.MOVESLinksWriter(scenario_name)
#     link_writer.save_data(moves_link_data, scenario_info)
#
#     # Set all sources
#     moves_link_source_data = _fill_moves_link_source_data(
#         scenario_name, aggregated_link_data['link_id'])
#     link_source_writer = data_writer.MOVESLinkSourceWriter(scenario_name)
#     link_source_writer.save_data(moves_link_source_data, scenario_info)
#
#     # Get the speeds for each link per second
#     drive_sched = _create_moves_speeds_from_link_evaluation(
#         link_evaluation_data)
#     drive_sched_writer = data_writer.MOVESLinkDriveWriter(scenario_name)
#     drive_sched_writer.save_data(drive_sched, scenario_info)


def translate_link_evaluation_to_moves(
        scenario_info: file_handling.ScenarioInfo,
        warmup_minutes: int = 5):
    """
    Reads link evaluation output files from VISSIM and write link,
    link source and drive schedule xls files for use in MOVES
    """
    scenario_name = 'platoon_discretionary_lane_change'
    # Load VISSIM data
    link_evaluation_reader = readers.LinkEvaluationReader(scenario_name)
    link_evaluation_data = (
        link_evaluation_reader.load_data_from_scenario(
            scenario_info)
    )
    drop_warmup_samples(link_evaluation_data, warmup_minutes)

    file_handler = file_handling.FileHandler(scenario_name)
    links = file_handler.get_main_links()
    link_evaluation_data.drop(
        index=link_evaluation_data[
            ~link_evaluation_data['link_number'].isin(links)].index,
        inplace=True
    )

    # Each lane of each link segment is a new MOVES link.
    link_evaluation_data['linkID'] = MovesProcessor.create_unique_link_ids(
        link_evaluation_data)

    link_evaluation_data.sort_values('link_id', kind='stable', inplace=True,
                                     ignore_index=True)

    # Aggregated data files
    aggregated_link_data = link_evaluation_data.groupby('linkID')[[
        'linkID', 'segment_length', 'volume', 'average_speed']].mean()
    processors = [MovesLinksProcessor(scenario_name),
                  MovesLinkSourceProcessor(scenario_name),
                  MovesLinkDriveProcessor(scenario_name)]
    for p in processors:
        moves_data = p.process(aggregated_link_data)
        p.writer.save_data(moves_data, scenario_info)


# def _fill_moves_link_data_from_link_evaluation_data(
#         scenario_name: str,
#         link_evaluation_data: pd.DataFrame) -> pd.DataFrame:
#     # We will pretend each simulation is a different link to avoid having to
#     # run MOVES over and over again
#     moves_link_reader = readers.MovesLinkReader(scenario_name)
#     county_id = moves_link_reader.get_count_id()
#     zone_id = moves_link_reader.get_zone_id()
#     road_type_id = moves_link_reader.get_road_id()
#     off_net_id = moves_link_reader.get_off_road_id()
#     moves_link_data = pd.DataFrame()
#
#     moves_link_data['linkID'] = link_evaluation_data['link_id']
#     moves_link_data['countyID'] = county_id
#     moves_link_data['zoneID'] = zone_id
#     moves_link_data['roadTypeID'] = road_type_id
#     moves_link_data['linkLength'] = (link_evaluation_data['segment_length']
#                                      / 1000 * _km_to_mile)
#     moves_link_data['linkVolume'] = np.round(link_evaluation_data['volume'])
#     moves_link_data['linkAvgSpeed'] = (
#             link_evaluation_data['average_speed'] * _km_to_mile)
#     moves_link_data['linkDescription'] = 0
#     moves_link_data['linkAvgGrade'] = 0
#     # Add one off-network link
#     moves_link_data.loc[max(moves_link_data.index) + 1] = [
#         moves_link_data['linkID'].max() + 1, county_id, zone_id, off_net_id,
#         0, 0, 0, 0, 0]
#     return moves_link_data


def _create_link_evaluation_data_for_single_vehicle(
        link_evaluation_data: pd.DataFrame,
        vehicle_data: pd.DataFrame
):
    """
    :param link_evaluation_data: Link evaluation data of simulation
    :param vehicle_data: Vehicle record of a single vehicle
    """
    # We want data[link_id, segment_length, volume, average_speed]
    # relevant_data = link_evaluation_data[
    #         link_evaluation_data['link_number'].isin(main_links)]
    link_and_length = link_evaluation_data.groupby(
        'link_number', as_index=False)['segment_length'].first()
    average_speed = vehicle_data.groupby('link')['vx'].mean()
    new_data = link_and_length.merge(
        average_speed, left_on='link_number', right_index=True)
    new_data['volume'] = 1
    return new_data.rename(columns={'vx': 'average_speed'})


# def _fill_moves_link_data_from_vissim_vehicle(
#         scenario_name: str, link_data: pd.DataFrame,
#         vehicle_data: pd.DataFrame):
#     used_links = vehicle_data['link'].unique()
#     used_links_idx = link_data[link_data['number'].isin(
#         used_links)].index
#
#     moves_link_reader = readers.MovesLinkReader(scenario_name)
#     county_id = moves_link_reader.get_count_id()
#     zone_id = moves_link_reader.get_zone_id()
#     road_type_id = moves_link_reader.get_road_id()
#     moves_link_data = pd.DataFrame()
#     avg_speed_per_link = vehicle_data.groupby('link')['vx'].mean()
#     # moves_link_data['linkID'] = (link_data.loc[used_links_idx, 'number'].
#     #                              reset_index().index)
#     moves_link_data['linkID'] = link_data.loc[used_links_idx, 'number']
#     moves_link_data['countyID'] = county_id
#     moves_link_data['zoneID'] = zone_id
#     moves_link_data['roadTypeID'] = road_type_id
#     moves_link_data['linkLength'] = (link_data.loc[used_links_idx, 'length']
#                                      / 1000 * _km_to_mile).to_numpy()
#     moves_link_data['linkVolume'] = 1
#     temp = moves_link_data[['linkID']].merge(avg_speed_per_link,
#                                              left_on='linkID',
#                                              right_index=True)
#     moves_link_data['linkAvgSpeed'] = temp['vx'] * _km_to_mile
#     moves_link_data['linkDescription'] = 0
#     moves_link_data['linkAvgGrade'] = 0
#     return moves_link_data


def _fill_moves_link_source_data(scenario_name: str,
                                 link_ids: pd.Series) -> pd.DataFrame:
    # NOTE: for now we're only dealing with cars
    moves_link_source_reader = readers.MovesLinkSourceReader(scenario_name)
    car_id = moves_link_source_reader.get_passenger_vehicle_id()
    link_ids.name = 'linkID'
    moves_link_source_data = pd.DataFrame(link_ids)
    moves_link_source_data['sourceTypeID'] = car_id
    moves_link_source_data['sourceTypeHourFraction'] = 1
    return moves_link_source_data


# def _create_moves_speeds_from_link_evaluation(
#         link_evaluation_data: pd.DataFrame) -> pd.DataFrame:
#     # Create the Moves-matching data. We consider the speed constant during
#     # the interval
#     interval_str = link_evaluation_data['time_interval'].iloc[0].split('-')
#     interval_duration = int(interval_str[1]) - int(interval_str[0])
#     drive_sched = pd.DataFrame(np.repeat(link_evaluation_data[[
#         'link_id', 'average_speed']].to_numpy(), interval_duration, axis=0))
#     drive_sched.columns = ['linkID', 'speed']
#     drive_sched.reset_index(drop=True, inplace=True)
#     drive_sched['speed'] *= _km_to_mile
#     drive_sched['linkID'] = drive_sched['linkID'].astype(int)
#     first_link = drive_sched['linkID'].iloc[0]
#     samples_per_link = np.count_nonzero(drive_sched['linkID'] == first_link)
#     drive_sched['secondID'] = (drive_sched.index + 1 -
#                                (drive_sched['linkID'] - first_link)
#                                * samples_per_link).astype(int)
#     drive_sched['grade'] = 0
#     return drive_sched


def _create_moves_speeds_from_vehicle_record(
        vehicle_data: pd.DataFrame, link_ids: List) -> pd.DataFrame:
    drive_sched = vehicle_data.groupby(
        'secondID', as_index=False).agg({'link': 'first', 'vx': 'mean'})
    drive_sched.drop(
        index=drive_sched[~drive_sched['link'].isin(link_ids)].index,
        inplace=True
    )
    drive_sched['grade'] = 0
    drive_sched['vx'] *= _km_to_mile
    drive_sched.rename(
        columns={'link': 'linkID', 'vx': 'speed'}, inplace=True)
    return drive_sched


class MovesProcessor(ABC):

    def __init__(self, scenario_name: str, writer):
        self.scenario_name = scenario_name
        self.writer = writer(scenario_name)
        # self.file_handler = file_handling.FileHandler(scenario_name)

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def create_unique_link_ids(data: pd.DataFrame):
        """
        :param data: Link evaluation data from VISSIM
        """
        return data[['simulation_number', 'link_number',
                     'link_segment', 'lane']].apply(tuple, axis=1).rank(
            method='dense').astype(int)


class MovesLinksProcessor(MovesProcessor):
    """
    Processes aggregated data from VISSIM link evaluation to MOVES
    """
    _writer = data_writer.MOVESLinksWriter

    def __init__(self, scenario_name: str):
        MovesProcessor.__init__(self, scenario_name, self._writer)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: Link evaluation data from VISSIM
        """

        # We will pretend each simulation is a different link to avoid having to
        # run MOVES over and over again
        moves_link_reader = readers.MovesLinkReader(self.scenario_name)
        county_id = moves_link_reader.get_count_id()
        zone_id = moves_link_reader.get_zone_id()
        road_type_id = moves_link_reader.get_road_id()
        off_net_id = moves_link_reader.get_off_road_id()
        moves_link_data = pd.DataFrame()

        moves_link_data['linkID'] = data['link_id']
        moves_link_data['countyID'] = county_id
        moves_link_data['zoneID'] = zone_id
        moves_link_data['roadTypeID'] = road_type_id
        moves_link_data['linkLength'] = (data['segment_length']
                                         / 1000 * _km_to_mile)
        moves_link_data['linkVolume'] = np.round(data['volume'])
        moves_link_data['linkAvgSpeed'] = (
                data['average_speed'] * _km_to_mile)
        moves_link_data['linkDescription'] = 0
        moves_link_data['linkAvgGrade'] = 0
        # Add one off-network link
        moves_link_data.loc[max(moves_link_data.index) + 1] = [
            moves_link_data['linkID'].max() + 1, county_id, zone_id, off_net_id,
            0, 0, 0, 0, 0]
        return moves_link_data


class MovesLinkSourceProcessor(MovesProcessor):
    _writer = data_writer.MOVESLinkSourceWriter

    def __init__(self, scenario_name: str):
        MovesProcessor.__init__(self, scenario_name, self._writer)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: All link ids
        """
        # NOTE: for now we're only dealing with cars
        moves_link_source_reader = readers.MovesLinkSourceReader(
            self.scenario_name)
        car_id = moves_link_source_reader.get_passenger_vehicle_id()
        # link_ids = data['linkID']
        # link_ids.name = 'linkID'
        # moves_link_source_data = pd.DataFrame(link_ids)
        # moves_link_source_data['sourceTypeID'] = car_id
        # moves_link_source_data['sourceTypeHourFraction'] = 1
        moves_link_source_data = pd.DataFrame(data={
            'linkID': data['linkID'], 'sourceTypeID': car_id,
            'sourceTypeHourFraction': 1
        })
        return moves_link_source_data


class MovesLinkDriveProcessor(MovesProcessor):
    _writer = data_writer.MOVESLinkDriveWriter

    def __init__(self, scenario_name: str, from_vehicle_records=False):
        MovesProcessor.__init__(self, scenario_name, self._writer)
        self.from_vehicle_records = from_vehicle_records

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.from_vehicle_records:
            return MovesLinkDriveProcessor.process_from_vehicle_records(data)
        else:
            return MovesLinkDriveProcessor.process_from_link_data(data)

    @staticmethod
    def process_from_vehicle_records(data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: Vehicle records on the relevant link
        """
        drive_sched = data.groupby(
            'secondID', as_index=False).agg({'link': 'first', 'vx': 'mean'})
        # drive_sched.drop(
        #     index=drive_sched[~drive_sched['link'].isin(link_ids)].index,
        #     inplace=True
        # )
        drive_sched['grade'] = 0
        drive_sched['vx'] *= _km_to_mile
        drive_sched.rename(
            columns={'link': 'linkID', 'vx': 'speed'}, inplace=True)
        return drive_sched

    @staticmethod
    def process_from_link_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: Link evaluation data from VISSIM
        """
        # Create the Moves-matching data. We consider the speed constant during
        # the interval
        interval_str = data['time_interval'].iloc[0].split('-')
        interval_duration = int(interval_str[1]) - int(interval_str[0])
        drive_sched = pd.DataFrame(np.repeat(data[[
            'linkID', 'average_speed']].to_numpy(), interval_duration,
                                             axis=0))
        drive_sched.columns = ['linkID', 'speed']
        drive_sched.reset_index(drop=True, inplace=True)
        drive_sched['speed'] *= _km_to_mile
        drive_sched['linkID'] = drive_sched['linkID'].astype(int)
        first_link = drive_sched['linkID'].iloc[0]
        samples_per_link = np.count_nonzero(drive_sched['linkID'] == first_link)
        drive_sched['secondID'] = (drive_sched.index + 1 -
                                   (drive_sched['linkID'] - first_link)
                                   * samples_per_link).astype(int)
        drive_sched['grade'] = 0
        return drive_sched
