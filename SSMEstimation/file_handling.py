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
    """Contains information about different VISSIM networks"""
    file_name: str
    relative_folder: str
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

_network_info = {
    'in_and_out': _NetworkInfo('highway_in_and_out_lanes', '',
                               [2, 10001], [10003, 5], [3]),
    'in_and_merge': _NetworkInfo('highway_in_and_merge', '',
                                 [2, 10001], [],
                                 [3]),
    'i710': _NetworkInfo('I710-MultiSec-3mi', '', [], [], []),
    'us101': _NetworkInfo('US_101', '', [4], [5], []),
    'traffic_lights': _NetworkInfo('traffic_lights_study',
                                   'traffic_lights_study', [], [], []),
    'platoon_lane_change': _NetworkInfo('platoon_lane_change',
                                        'platoon_lane_change',
                                        [2, 10001], [1003, 5], [3])
}


def get_networks_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].networks_folder


def get_shared_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].shared_folder


def copy_results_from_multiple_scenarios(
        network_name: str,
        percentages_per_vehicle_types: List[Dict[VehicleType, int]],
        vehicles_per_lane: List[int],
        accepted_risks: List[int]):
    for item in percentages_per_vehicle_types:
        vehicle_types = list(item.keys())
        percentages = list(item.values())
        for vi in vehicles_per_lane:
            for ar in accepted_risks:
                copy_result_files(network_name, vehicle_types,
                                  percentages, vi, ar)


def copy_result_files(network_name: str, vehicle_types: List[VehicleType],
                      controlled_percentages: List[int], vehicles_per_lane: int,
                      accepted_risk: int):
    """Copies data collections, link segments, lane change data, SSMs,
    Risky Maneuvers and Violations files to a similar folder structure in
    Google Drive. """

    # network_relative_address = get_relative_address_from_network_name(
    #     network_name)
    # percentage_folder = create_percent_folder_name(
    #     [controlled_percentages], [vehicle_types])
    # vehicles_per_lane_folder = (
    #     create_vehs_per_lane_folder_name(
    #         vehicles_per_lane))
    # source_dir_old = os.path.join(get_networks_folder(),
    #                              network_relative_address,
    #                           percentage_folder, vehicles_per_lane_folder)
    # target_dir = os.path.join(get_shared_folder(), network_relative_address,
    #                           percentage_folder, vehicles_per_lane_folder)
    base_source_folder = os.path.join(
        get_networks_folder(),
        get_relative_address_from_network_name(network_name))
    base_target_folder = os.path.join(
        get_shared_folder(),
        get_relative_address_from_network_name(network_name))
    source_dir = get_data_folder(base_source_folder, vehicle_types,
                                 controlled_percentages, vehicles_per_lane,
                                 accepted_risk)
    target_dir = get_data_folder(base_target_folder, vehicle_types,
                                 controlled_percentages, vehicles_per_lane,
                                 accepted_risk)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    all_file_names = os.listdir(source_dir)
    for file_name in all_file_names:
        file_extension = file_name.split('.')[-1]
        if file_extension in {'csv', 'att', 'spw'}:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)


def get_file_name_from_network_name(network_name):
    if network_name in _network_info:
        network_name = _network_info[network_name].file_name
    elif network_name in {i.file_name for i in _network_info.values()}:
        pass
    else:
        raise ValueError('Network "{}" is not in the list of valid '
                         'simulations\nCheck whether the network exists  '
                         'and add it to the VissimInterface attribute '
                         'existing_networks'.
                         format(network_name))
    return network_name


def get_network_name_from_file_name(network_file):
    for key in _network_info:
        if _network_info[key].file_name == network_file:
            return key
    raise ValueError('Simulation name for this file was not found')
    # return list(_network_names_map.keys())[
    #     list(_network_names_map.values()).index(network_file)]


def get_relative_address_from_network_name(network_name):
    return os.path.join(_network_info[network_name].relative_folder,
                        _network_info[network_name].file_name)
    # return os.path.join(_network_relative_folders_map[network_name],
    #                     _network_names_map[network_name])


def get_data_folder(network_results_folder: str,
                    vehicle_type: List[VehicleType],
                    controlled_percentage: List[int],
                    vehicles_per_lane: int,
                    accepted_risk: int) -> str:
    """
    Creates a string with the full path of the simulation results data
    folder. If all parameters are None, returns the test data folder

    :param network_results_folder: Result's folder for a given network
    :param vehicle_type: list of enums to indicate the vehicle (controller) type
    :param controlled_percentage: Percentage of autonomous vehicles
     in the simulation. Current possible values: 0:25:100
    :param vehicles_per_lane: Vehicle input per lane on VISSIM. Possible
     values depend on the controlled_vehicles_percentage: 500:500:2500
    :param accepted_risk: maximum lane changing risk in m/s
    :return: string with the folder where the data is
    """
    if (vehicle_type is None and controlled_percentage is None
            and vehicles_per_lane is None):
        return os.path.join(network_results_folder, 'test')

    percent_folder = create_percent_folder_name(
        controlled_percentage, vehicle_type)
    vehicle_input_folder = create_vehs_per_lane_folder_name(
        vehicles_per_lane)
    accepted_risk_folder = create_accepted_risk_folder_name(
        accepted_risk)
    return os.path.join(network_results_folder, percent_folder,
                        vehicle_input_folder, accepted_risk_folder)


def get_on_ramp_links(network_name: str):
    return _network_info[network_name].on_ramp_link


def get_off_ramp_links(network_name: str):
    return _network_info[network_name].off_ramp_link


def get_merging_links(network_name: str):
    return _network_info[network_name].merging_link


def create_percent_folder_name(percentage: List[int],
                               vehicle_type: List[VehicleType]) -> str:
    """Creates the name of the folder which contains results for the
    given percentage of controlled vehicles (not the full path)"""
    # if not isinstance(percentage, list):
    #     percentage = [percentage]
    # if not isinstance(vehicle_type, list):
    #     vehicle_type = [vehicle_type]

    if sum(percentage) == 0:
        return '0_percent_'
    vehicle_type_names = [v.name.lower() for v in vehicle_type]
    percentage_folder = ''
    for v, p in sorted(zip(vehicle_type_names, percentage)):
        if p > 0:
            percentage_folder += str(int(p)) + '_percent_' + v + '_'

    return percentage_folder[:-1]  # remove last '_'


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
