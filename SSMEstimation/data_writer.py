import os
from typing import List, Union

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from vehicle import Vehicle, VehicleType
import file_handling


class DataWriter:
    # file_extension = '.csv'

    def __init__(self, data_type_identifier: str, file_extension: str,
                 network_name: str, vehicle_type: List[VehicleType]):
        network_file = file_handling.get_file_name_from_network_name(
            network_name)
        network_relative_address = (
            file_handling.get_relative_address_from_network_name(network_name))
        self.file_base_name = (network_file + '_'
                               + data_type_identifier)
        self.network_data_dir = os.path.join(
            file_handling.get_networks_folder(), network_relative_address)
        self.file_extension = file_extension
        self.vehicle_type = vehicle_type
        #  [vt.name.lower() for vt in vehicle_type]

    @staticmethod
    def _save_as_csv(data: pd.DataFrame, folder_path: str,
                     file_name: str):

        full_address = os.path.join(folder_path, file_name)
        try:
            data.to_csv(full_address, index=False)
        except FileNotFoundError:
            print('Could not save at', full_address, '. Saving at: ',
                  file_handling.get_networks_folder(), 'instead.')
            data.to_csv(os.path.join(file_handling.get_networks_folder(),
                                     file_name), index=False)


class PostProcessedDataWriter(DataWriter):
    """Helps saving aggregated SSM results to files"""
    _data_type_identifier = 'SSM Results'
    _file_extension = '.csv'

    def __init__(self, network_name: str, vehicle_type: List[VehicleType]):
        DataWriter.__init__(self, self._data_type_identifier,
                            self._file_extension, network_name, vehicle_type)

    def save_as_csv(self, data: pd.DataFrame,
                    controlled_vehicles_percentage: Union[List[int], None],
                    vehicles_per_lane: Union[int, None]):
        """
        Saves the data on the proper results folder based on the simulated
        network, controlled vehicles percentage and vehicle input.
        Parameters with None value are only accepted for test runs.
        :param data: data to be saved
        :param controlled_vehicles_percentage: percentage of controlled
         vehicles in the simulation
        :param vehicles_per_lane: vehicle input per lane of the simulation
        :return: Nothing, just saves the data
        """
        file_name = self.file_base_name + self.file_extension
        folder_path = file_handling.get_data_folder(
            self.network_data_dir, self.vehicle_type,
            controlled_vehicles_percentage, vehicles_per_lane)
        # percentage_folder = file_handling.create_percent_folder_name(
        #     controlled_vehicles_percentage, self.vehicle_type)
        # vehicles_per_lane_folder = (
        #     file_handling.create_vehs_per_lane_folder_name(
        #         vehicles_per_lane))
        # folder_path = os.path.join(self.network_data_dir, percentage_folder,
        #                            vehicles_per_lane_folder)
        self._save_as_csv(data, folder_path, file_name)


class RiskyManeuverWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Risky Maneuvers'

    def __init__(self, network_name: str, vehicle_type: List[VehicleType]):
        DataWriter.__init__(self, self._data_type_identifier,
                            self._file_extension, network_name, vehicle_type)


class TrafficLightViolationWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Traffic Light Violations'
    _file_extension = '.csv'

    def __init__(self, network_name: str, vehicle_type: List[VehicleType]):
        DataWriter.__init__(self, self._data_type_identifier,
                            self._file_extension, network_name, vehicle_type)


class DiscomfortWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Discomfort'
    _file_extension = '.csv'

    def __init__(self, network_name: str, vehicle_type: List[VehicleType]):
        DataWriter.__init__(self, self._data_type_identifier,
                            self._file_extension, network_name, vehicle_type)


class LaneChangeWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Lane Changes'
    _file_extension = '.csv'

    def __init__(self, network_name: str, vehicle_type: List[VehicleType]):
        DataWriter.__init__(self, self._data_type_identifier,
                            self._file_extension, network_name, vehicle_type)

class MergedDataWriter(DataWriter):
    _data_type_identifier = 'Merged Data'
    _file_extension = '.csv'

    def __init__(self, network_name: str, vehicle_type: List[VehicleType]):
        DataWriter.__init__(self, self._data_type_identifier,
                            self._file_extension, network_name,
                            vehicle_type)

    def save_as_csv(self, data: pd.DataFrame,
                    controlled_vehicles_percentage: List[int]):
        file_name = self.file_base_name + self.file_extension
        percentage_folder = file_handling.create_percent_folder_name(
            controlled_vehicles_percentage, self.vehicle_type)
        folder_path = os.path.join(self.network_data_dir, percentage_folder)
        self._save_as_csv(data, folder_path, file_name)


class SyntheticDataWriter:
    """Creates synthetic data to help checking if the SSM computation is
    correct
    """

    file_extension = '.csv'
    data_dir = ('C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation'
                '\\synthetic_data\\')

    def write_data(self, data):
        """

        :param data: pandas dataframe
        :return: None
        """
        data.to_csv(os.path.join(self.data_dir,
                                 'synthetic_data' + self.file_extension),
                    index=False)

    @staticmethod
    def create_single_veh_data(vf, vl, is_lane_changing: bool = False,
                               follower_type=Vehicle.VISSIM_CAR_ID,
                               leader_type=Vehicle.VISSIM_CAR_ID):
        """
        Creates simple data for tests. Data includes only one vehicle at
        constant speed, with constant relative speed to its leader and
        linearly increasing gap.

        Only two vehicles, each one with constant speed. The kinematics do not
        have to be correct, i.e., we do not guarantee x = x0 + vt, since this is
        not important for the SSM tests.

        :param vf: Follower longitudinal velocity
        :param vl: Leader longitudinal velocity
        :param is_lane_changing: Indicates whether to use lane changing
         parameters
        :param follower_type: Follower vehicle type
        :param leader_type: Leader vehicle type
        :return: pandas dataframe with the synthetic data
        """

        max_gap = 50  # 10 ** 2
        max_time = 10 ** 2
        gap_interval = 0.1
        n_points = int(max_gap / gap_interval) + 1

        time = np.round(np.linspace(0, max_time, n_points), 2)
        follower_id = 1
        leader_id = follower_id + 1
        follower_df = pd.DataFrame({'time': time})
        follower_df['veh_id'] = follower_id
        follower_df['veh_type'] = follower_type * np.ones(n_points, dtype=int)
        follower_df['lane'] = np.ones(n_points, dtype=int)
        follower_df['x'] = np.zeros(n_points)
        follower_df['vx'] = vf
        follower_df['lane_change'] = 1 if is_lane_changing else 'None'
        follower_df['leader_id'] = leader_id
        follower_df['delta_x'] = np.round(np.linspace(gap_interval, max_gap,
                                                      n_points), 2)
        follower_df['delta_v'] = vf - vl
        follower_df['leader_type'] = leader_type * np.ones(n_points, dtype=int)

        return follower_df


class SignalControllerTreeEditor:
    """
    Class to edit signal controller files (.sig).

    Currently, we assume the signal controllers have a single signal group,
    which is always set to the Red-Green-Amber sequence.
    """
    _file_extension = '.sig'
    traffic_light_red = '1'
    traffic_light_green = '3'
    traffic_light_amber = '4'

    def set_times(self, signal_controller_tree: ET.ElementTree,
                  red_duration: int, green_duration: int,
                  amber_duration: int = 5, starts_at_red: bool = True):
        """"
        Sets red, green and optionally amber times. The function also sets the
        cycle time equal to the sum of the three times.

        :param signal_controller_tree: obtained by a SignalContrllerFileReader
        :param red_duration: in seconds
        :param green_duration: in seconds
        :param amber_duration: in seconds (optional)
        :param starts_at_red: if true, traffic light starts red. Otherwise,
         it starts green
        """
        # could include checks to count the number of signal controllers and
        # confirm the signal sequence

        # VISSIM uses nano seconds:
        red_duration *= 1000
        green_duration *= 1000
        amber_duration *= 1000

        program = signal_controller_tree.find('progs').find('prog')
        cycle_time = int(red_duration + green_duration + amber_duration)
        program.set('cycletime', str(cycle_time))

        if starts_at_red:
            red_start_time = '0'
            green_start_time = str(red_duration)
        else:
            green_start_time = '0'
            red_start_time = str(green_duration)
        signal_group_program = program.find('sgs').find('sg')

        for command in signal_group_program.iter('cmd'):
            if command.get('display') == self.traffic_light_red:
                command.set('begin', red_start_time)
            elif command.get('display') == self.traffic_light_green:
                command.set('begin', green_start_time)
            else:
                print('Unexpected traffic light color. Value ',
                      command.get('display'))

        signal_group_program.find('fixedstates').find('fixedstate').set(
            'duration', str(amber_duration))

    def save_file(self, signal_controller_tree: ET.ElementTree,
                  folder: str, file_name: str):
        signal_controller_tree.write(os.path.join(folder, file_name
                                                  + self._file_extension),
                                     encoding='UTF-8',
                                     xml_declaration=True)

    @staticmethod
    def set_signal_controller_id(signal_controller_tree: ET.ElementTree,
                                 new_id: int):
        signal_controller_tree.getroot().set('id', str(new_id))
