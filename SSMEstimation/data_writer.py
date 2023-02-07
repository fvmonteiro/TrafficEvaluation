import os
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from vehicle import PlatoonLaneChangeStrategy, Vehicle, VehicleType
import file_handling


class DataWriter:

    def __init__(self, data_type_identifier: str, file_extension: str,
                 scenario_name: str):
        self.file_handler = file_handling.FileHandler(scenario_name)
        self.file_base_name = (self.file_handler.get_file_name() + '_'
                               + data_type_identifier)
        self.file_extension = file_extension
        # self.vehicle_type = vehicle_type
        #  [vt.name.lower() for vt in vehicle_type]

    def _save_as_csv(self, data: pd.DataFrame, folder_path: str,
                     file_name: str):
        full_address = os.path.join(folder_path, file_name)
        try:
            data.to_csv(full_address, index=False)
        except OSError:
            self.file_handler.set_is_data_in_cloud(False)
            result_folder = self.file_handler.get_networks_folder()
            print('Could not save at', full_address, '. Saving at: ',
                  result_folder, 'instead.')
            data.to_csv(os.path.join(result_folder,
                                     file_name), index=False)

    @staticmethod
    def _save_as_xls(self, data: pd.DataFrame, folder_path: str,
                     file_name: str, sheet_name: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        full_address = os.path.join(folder_path, file_name)
        try:
            data.to_excel(full_address, sheet_name, index=False)
        except OSError:
            print('Could not save at', full_address, '. Saving at: ',
                  file_handling.get_moves_folder(), 'instead.')
            data.to_csv(os.path.join(file_handling.get_moves_folder(),
                                     file_name), index=False)


class PostProcessedDataWriter(DataWriter):
    """Helps saving results obtained after processing VISSIM results to files"""
    _file_extension = '.csv'

    def __init__(self, scenario_name: str,  # vehicle_type: List[VehicleType],
                 data_type_identifier: str):
        DataWriter.__init__(self, data_type_identifier,
                            self._file_extension, scenario_name)

    def save_as_csv(
            self, data: pd.DataFrame,
            vehicle_percentages: Dict[VehicleType, int],
            vehicle_input_per_lane: int,
            accepted_risk: int = None,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, int] = None):
        """
        Saves the data on the proper results folder based on the simulated
        network, controlled vehicles percentage and vehicle input.
        Parameters with None value are only accepted for test runs.

        :param data: data to be saved
        :param vehicle_percentages: Describes the percentages of controlled
         vehicles in the simulations.
        :param vehicle_input_per_lane: vehicle input per lane of the simulation
        :param accepted_risk: maximum lane changing risk
        :param platoon_lane_change_strategy:
        :param orig_and_dest_lane_speeds:
        :return: Nothing, just saves the data
        """

        file_name = self.file_base_name + self.file_extension
        folder_path = self.file_handler.get_vissim_data_folder(
            vehicle_percentages, vehicle_input_per_lane,
            accepted_risk=accepted_risk,
            platoon_lane_change_strategy=platoon_lane_change_strategy,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds)
        self._save_as_csv(data, folder_path, file_name)


class SSMDataWriter(PostProcessedDataWriter):
    """Helps saving aggregated SSM results to files"""
    _data_type_identifier = 'SSM Results'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class RiskyManeuverWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Risky Maneuvers'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class TrafficLightViolationWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Traffic Light Violations'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class DiscomfortWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Discomfort'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class LaneChangeWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Lane Changes'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class LaneChangeIssuesWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Lane Change Issues'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class PlatoonLaneChangeEfficiencyWriter(PostProcessedDataWriter):
    _data_type_identifier = 'Platoon Lane Change Efficiency'

    def __init__(self, scenario_name: str):
        PostProcessedDataWriter.__init__(self, scenario_name,
                                         self._data_type_identifier)


class MOVESDataWriter(DataWriter):
    _file_extension = '.xlsx'

    def __init__(self, scenario_name: str, data_type_identifier: str,
                 sheet_name: str):
        DataWriter.__init__(self, 'MOVES_' + data_type_identifier,
                            self._file_extension, scenario_name)
        self.sheet_name = sheet_name

    def save_data(
            self, data: pd.DataFrame,
            vehicle_percentages: Dict[VehicleType, int],
            vehicle_input_per_lane: int,
            accepted_risk: int = None,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, int] = None
            ):
        folder_path = self.file_handler.get_moves_data_folder(
            vehicle_percentages, vehicle_input_per_lane,
            accepted_risk, platoon_lane_change_strategy,
            orig_and_dest_lane_speeds
        )
        file_name = self.file_base_name + self.file_extension
        self._save_as_xls(data, folder_path, file_name, self.sheet_name)


class MOVESLinksWriter(MOVESDataWriter):
    _data_type_identifier = 'links'
    _sheet_name = 'link'

    def __init__(self, scenario_name: str):
        MOVESDataWriter.__init__(self, scenario_name,
                                 self._data_type_identifier, self._sheet_name)


class MOVESLinkSourceWriter(MOVESDataWriter):
    _data_type_identifier = 'linksource'
    _sheet_name = 'linkSourceTypeHour'

    def __init__(self, scenario_name: str):
        MOVESDataWriter.__init__(self, scenario_name,
                                 self._data_type_identifier, self._sheet_name)


class MOVESLinkDriveWriter(MOVESDataWriter):
    _data_type_identifier = 'linkdrive'
    _sheet_name = 'driveSchedule'

    def __init__(self, scenario_name: str):
        MOVESDataWriter.__init__(self, scenario_name,
                                 self._data_type_identifier, self._sheet_name)


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
