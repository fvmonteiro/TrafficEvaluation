from dataclasses import dataclass
import os
import shutil
from typing import Dict, List, Union

from vehicle import VehicleType


@dataclass
class _PCInfo:
    """Used to help run simulation in different computers"""
    easy_id: str
    networks_folder: str
    shared_folder: str


@dataclass
class _NetworkInfo:
    """Contains information about different VISSIM network"""
    file_name: str
    on_ramp_link: List[int]
    off_ramp_link: List[int]
    merging_link: List[int]


@dataclass
class _ScenarioInfo:
    """Contains information about different simulation scenarios"""
    network: str
    file_name: str
    results_folder: str
    on_ramp_link: List[int]
    off_ramp_link: List[int]
    merging_link: List[int]


_folders_map = {
    'DESKTOP-P2O85S9': _PCInfo('personal_pc',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'TrafficSimulation\\VISSIM_networks',
                               'G:\\My Drive\\Safety in Mixed Traffic'
                               '\\data_exchange'),
    'DESKTOP-626HHGI': _PCInfo('usc-old',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'AV_TrafficSimulation\\VISSIM_networks',
                               'C:\\Users\\fvall\\Google Drive\\'
                               'Safety in Mixed Traffic\\data_exchange'),
    'DESKTOP-B1GECOE': _PCInfo('usc',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'TrafficSimulation\\VISSIM_networks',
                               'G:\\My Drive\\Safety in Mixed Traffic'
                               '\\data_exchange'),
}

# TODO: have separate scenario and network info data classes.
_scenario_info = {
    'in_and_out_safe':
        _ScenarioInfo('in_and_out', 'highway_in_and_out_lanes', 'safe',
                      [2, 10001], [10003, 5], [3]),
    'in_and_out_risk_in_headway':
        _ScenarioInfo('in_and_out', 'highway_in_and_out_lanes',
                      'risk_in_headway',
                      [2, 10001], [10003, 5], [3]),
    'in_and_out_risk_in_gap':
        _ScenarioInfo('in_and_out', 'highway_in_and_out_lanes',
                      'risk_in_gap',
                      [2, 10001], [10003, 5], [3]),
    'in_and_merge':
        _ScenarioInfo('in_and_merge', 'highway_in_and_merge', 'results',
                      [2, 10001], [], [3]),
    'i710':
        _ScenarioInfo('i710', 'I710-MultiSec-3mi', 'results', [], [], []),
    'us101':
        _ScenarioInfo('us101', 'US_101', 'results', [4], [5], []),
    'traffic_lights':
        _ScenarioInfo('traffic_lights_study', 'traffic_lights_study',
                      'traffic_lights_study', [], [], []),
    'platoon_lane_change':
        _ScenarioInfo('platoon_lane_change', 'platoon_lane_change',
                      'platoon_lane_change',
                      [2, 10001], [1003, 5], [3])
}


def temp_name_editing():
    folder = ("C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation"
              "\\VISSIM_networks\\highway_in_and_out_lanes\\test")
    for file in os.listdir(folder):
        file_str = os.fsdecode(file)
        file_name, file_ext = file_str.split('.')
        base_name = file_name[:-3]
        num_str = int(file_name[-3:]) + 5
        new_name = base_name + str(num_str).rjust(3, '0') + '.' + file_ext
        old_file = os.path.join(folder, file_str)
        new_file = os.path.join(folder, new_name)
        os.rename(old_file, new_file)


class FileHandler:
    """Class is the interface between scenario names and all their properties"""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.scenario_info = _scenario_info[scenario_name]
        # self.network_name = _scenario_info[scenario_name].network

    def get_network_name(self):
        return self.scenario_info.network

    def get_file_name(self):
        return self.scenario_info.file_name

    def get_network_file_relative_address(self):
        # The network files are in folder with the same name as the file
        return self.scenario_info.file_name

    def get_network_file_folder(self):
        return os.path.join(get_networks_folder(),
                            self.get_network_file_relative_address())

    def get_results_relative_address(self):
        return os.path.join(self.get_network_file_relative_address(),
                            self.scenario_info.results_folder)

    def get_results_base_folder(self):
        return os.path.join(get_networks_folder(),
                            self.get_results_relative_address())

    def get_on_ramp_links(self):
        return self.scenario_info.on_ramp_link

    def get_off_ramp_links(self):
        return self.scenario_info.off_ramp_link

    def get_merging_links(self):
        return self.scenario_info.merging_link

    def get_data_folder(self,
                        vehicle_type: List[VehicleType],
                        controlled_percentage: List[int],
                        vehicles_per_lane: int,
                        accepted_risk: int) -> str:
        """
        Creates a string with the full path of the simulation results data
        folder. If all parameters are None, returns the test data folder

        :param vehicle_type: list of enums indicating the vehicle (controller)
         type
        :param controlled_percentage: Percentage of autonomous vehicles
         in the simulation. Current possible values: 0:25:100
        :param vehicles_per_lane: Vehicle input per lane on VISSIM. Possible
         values depend on the controlled_vehicles_percentage: 500:500:2500
        :param accepted_risk: maximum lane changing risk in m/s
        :return: string with the folder where the data is
        """
        results_base_folder = self.get_results_base_folder()
        if (vehicle_type is None and controlled_percentage is None
                and vehicles_per_lane is None):
            return os.path.join(results_base_folder, 'test')

        if sum(controlled_percentage) == 0:
            accepted_risk = None
            results_base_folder = os.path.join(self.get_network_file_folder())

        percent_folder = create_percent_folder_name(
            controlled_percentage, vehicle_type)
        vehicle_input_folder = create_vehs_per_lane_folder_name(
            vehicles_per_lane)
        accepted_risk_folder = create_accepted_risk_folder_name(
            accepted_risk)
        return os.path.join(results_base_folder, percent_folder,
                            vehicle_input_folder, accepted_risk_folder)

    def find_min_max_file_number(self,
                                 data_identifier: str,
                                 file_format: str,
                                 vehicle_type: List[VehicleType],
                                 percentage: List[int],
                                 vehicles_per_lane: int,
                                 accepted_risk: int = None) -> (int, int):
        """"
        Looks for the file with the highest simulation number. This is
        usually the file containing results from all simulations.

        :param data_identifier: last part of the file name
        :param file_format: file extension
        :param vehicle_type: Enum to indicate the vehicle (controller) type
        :param percentage: Percentage of autonomous vehicles
         in the simulation.
        :param vehicles_per_lane: Vehicle input per lane used in simulation
        :param accepted_risk: accepted lane change risk
        :return: highest simulation number.
        """
        max_simulation_number = -1
        min_simulation_number = 10000
        results_folder = self.get_data_folder(
            vehicle_type, percentage, vehicles_per_lane, accepted_risk)
        network_file = self.get_file_name()
        for file in os.listdir(results_folder):
            file_str = os.fsdecode(file)
            if (file_str.startswith(network_file + data_identifier)
                    and file_str.endswith(file_format)):
                file_no_extension = file_str.split('.')[0]
                try:
                    sim_number = int(file_no_extension.split('_')[-1])
                except ValueError:
                    print('File {} is not being read because its name does not '
                          'end with a number.'.format(file_no_extension))
                    continue
                if sim_number > max_simulation_number:
                    max_simulation_number = sim_number
                if sim_number < min_simulation_number:
                    min_simulation_number = sim_number

        return min_simulation_number, max_simulation_number

    def copy_results_from_multiple_scenarios(
            self,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            accepted_risks: List[int]):
        for item in percentages_per_vehicle_types:
            vehicle_types = list(item.keys())
            percentages = list(item.values())
            for vi in vehicles_per_lane:
                for ar in accepted_risks:
                    self.copy_result_files(vehicle_types,
                                           percentages, vi, ar)

    def copy_result_files(self, vehicle_types: List[VehicleType],
                          controlled_percentages: List[int],
                          vehicles_per_lane: int,
                          accepted_risk: int):
        """Copies data collections, link segments, lane change data, SSMs,
        Risky Maneuvers and Violations files to a similar folder structure in
        Google Drive. """

        source_dir = self.get_data_folder(
            vehicle_types, controlled_percentages, vehicles_per_lane,
            accepted_risk)
        target_dir = os.path.join(
            get_shared_folder(),
            source_dir.split(get_networks_folder() + "\\")[1]
        )
        # base_target_folder = os.path.join(
        #     get_shared_folder(),
        #     self.get_results_relative_address())
        # percent_folder = create_percent_folder_name(
        #     controlled_percentages, vehicle_types)
        # vehicle_input_folder = create_vehs_per_lane_folder_name(
        #     vehicles_per_lane)
        # accepted_risk_folder = create_accepted_risk_folder_name(
        #     accepted_risk)
        # target_dir = os.path.join(base_target_folder, percent_folder,
        #                           vehicle_input_folder, accepted_risk_folder)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        all_file_names = os.listdir(source_dir)
        all_csv_files = [file for file in all_file_names if
                         file.endswith('csv')]
        _, max_file_number = self.find_min_max_file_number(
            '', 'att', vehicle_types, controlled_percentages, vehicles_per_lane,
            accepted_risk)
        max_att_files = [file for file in all_file_names if
                         file.endswith(str(max_file_number) + '.att')]
        files_to_copy = all_csv_files + max_att_files
        for file_name in files_to_copy:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)


def create_percent_folder_name(percentage: List[int],
                               vehicle_type: List[VehicleType]) -> str:
    """Creates the name of the folder which contains results for the
    given percentage of controlled vehicles (not the full path)"""

    if sum(percentage) == 0:
        return '0_percent_'
    vehicle_type_names = [v.name.lower() for v in vehicle_type]
    percentage_folder = ''
    for v, p in sorted(zip(vehicle_type_names, percentage)):
        if p > 0:
            percentage_folder += str(int(p)) + '_percent_' + v + '_'

    return percentage_folder[:-1]  # remove last '_'


def get_networks_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].networks_folder


def get_shared_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].shared_folder


def create_vehs_per_lane_folder_name(vehicles_per_lane: Union[int, str]) -> str:
    """Creates the name of the folder which contains results for the
    given vehicle per lane input (not the full path)"""

    if not vehicles_per_lane:
        return ''
    if isinstance(vehicles_per_lane, str):
        vehs_per_lane_folder = vehicles_per_lane
    else:
        vehs_per_lane_folder = str(int(vehicles_per_lane)) + '_vehs_per_lane'
    return vehs_per_lane_folder


def create_accepted_risk_folder_name(accepted_risk: int) -> str:
    """
    Creates the name of the folder which contains the results for the given
    maximum accepted lane change risk
    :param accepted_risk: simulation's maximum accepted lane change risk
    :return: folder name as: [accepted_risk]_accepted_risk
    """
    return ('' if accepted_risk is None
            else str(accepted_risk) + '_accepted_risk')
