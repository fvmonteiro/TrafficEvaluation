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

        percent_folder = create_percent_folder_name(
            controlled_percentage, vehicle_type)
        vehicle_input_folder = create_vehs_per_lane_folder_name(
            vehicles_per_lane)
        accepted_risk_folder = create_accepted_risk_folder_name(
            accepted_risk)
        return os.path.join(results_base_folder, percent_folder,
                            vehicle_input_folder, accepted_risk_folder)

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

        base_source_folder = os.path.join(
            get_networks_folder(),
            self.get_results_relative_address())
        base_target_folder = os.path.join(
            get_shared_folder(),
            self.get_results_relative_address())
        percent_folder = create_percent_folder_name(
            controlled_percentages, vehicle_types)
        vehicle_input_folder = create_vehs_per_lane_folder_name(
            vehicles_per_lane)
        accepted_risk_folder = create_accepted_risk_folder_name(
            accepted_risk)
        source_dir = os.path.join(base_source_folder, percent_folder,
                                  vehicle_input_folder, accepted_risk_folder)
        target_dir = os.path.join(base_target_folder, percent_folder,
                                  vehicle_input_folder, accepted_risk_folder)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        all_file_names = os.listdir(source_dir)
        for file_name in all_file_names:
            file_extension = file_name.split('.')[-1]
            if file_extension in {'csv', 'att'}:
                shutil.copy(os.path.join(source_dir, file_name), target_dir)


# def get_network_name_from_scenario(scenario_name: str):
#     return _scenario_info[scenario_name].network
#
#
# def get_file_name_from_scenario(scenario_name: str):
#     return _scenario_info[scenario_name].file_name


# def get_file_name_from_network_name(network_name: str):
#     if network_name in _scenario_info:
#         return _scenario_info[network_name].file_name
#     for existing_scenario in _scenario_info.values():
#         if network_name == existing_scenario.network:
#             return existing_scenario.file_name
#     raise ValueError('Network "{}" is not in the list of valid '
#                      'simulations\nCheck whether the network exists  '
#                      'and add it to the VissimInterface attribute '
#                      'existing_networks'.
#                      format(network_name))


# def get_network_name_from_file_name(network_file:str):
#     for key in _scenario_info:
#         if _scenario_info[key].file_name == network_file:
#             return key
#     raise ValueError('Simulation name for this file was not found')


# def get_results_relative_address(network_name: str):
#     return os.path.join(get_network_file_relative_address(network_name),
#                         _scenario_info[network_name].results_folder)


# def get_network_file_relative_address(network_name: str):
#     # The network files are in folder with the same name as the file
#     return _scenario_info[network_name].file_name


# def get_data_folder(network_results_folder: str,
#                     vehicle_type: List[VehicleType],
#                     controlled_percentage: List[int],
#                     vehicles_per_lane: int,
#                     accepted_risk: int) -> str:
#     """
#     Creates a string with the full path of the simulation results data
#     folder. If all parameters are None, returns the test data folder
#
#     :param network_results_folder: Result's folder for a given network
#     :param vehicle_type: list of enums to indicate the vehicle (controller) type
#     :param controlled_percentage: Percentage of autonomous vehicles
#      in the simulation. Current possible values: 0:25:100
#     :param vehicles_per_lane: Vehicle input per lane on VISSIM. Possible
#      values depend on the controlled_vehicles_percentage: 500:500:2500
#     :param accepted_risk: maximum lane changing risk in m/s
#     :return: string with the folder where the data is
#     """
#     if (vehicle_type is None and controlled_percentage is None
#             and vehicles_per_lane is None):
#         return os.path.join(network_results_folder, 'test')
#
#     percent_folder = create_percent_folder_name(
#         controlled_percentage, vehicle_type)
#     vehicle_input_folder = create_vehs_per_lane_folder_name(
#         vehicles_per_lane)
#     accepted_risk_folder = create_accepted_risk_folder_name(
#         accepted_risk)
#     return os.path.join(network_results_folder, percent_folder,
#                         vehicle_input_folder, accepted_risk_folder)


# def get_on_ramp_links(network_name: str):
#     return _scenario_info[network_name].on_ramp_link
#
#
# def get_off_ramp_links(network_name: str):
#     return _scenario_info[network_name].off_ramp_link
#
#
# def get_merging_links(network_name: str):
#     return _scenario_info[network_name].merging_link


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
