import os
import shutil
from typing import List, Union

from vehicle import VehicleType

_network_names_map = {'in_and_out': 'highway_in_and_out_lanes',
                      'in_and_merge': 'highway_in_and_merge',
                      'i710': 'I710-MultiSec-3mi',
                      'us101': 'US_101',
                      'traffic_lights': 'traffic_lights_study',
                      'platoon_lane_change': 'platoon_lane_change'}

_network_relative_folders_map = {'in_and_out': '', 'in_and_merge': '',
                                 'i710': '', 'us101': '',
                                 'traffic_lights': 'traffic_lights_study',
                                 'platoon_lane_change': 'platoon_lane_change'}


def get_networks_folder() -> str:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
        return ('C:\\Users\\fvall\\Documents\\Research'
                '\\AV_TrafficSimulation\\VISSIM_networks')
    else:
        return ('C:\\Users\\fvall\\Documents\\Research'
                '\\TrafficSimulation\\VISSIM_networks')


def get_shared_folder() -> str:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
        return ('C:\\Users\\fvall\\Google Drive\\Safety in Mixed '
                'Traffic\\data_exchange')
    else:
        return 'G:\\My Drive\\Safety in Mixed Traffic\\data_exchange'


def copy_results_from_multiple_scenarios(network_name: str,
                                         vehicle_types: List[VehicleType],
                                         controlled_percentages: List[int],
                                         vehicles_per_lane: List[int]):
    for vt in vehicle_types:
        for p in controlled_percentages:
            for vi in vehicles_per_lane:
                copy_result_files(network_name, vt, p, vi)


def copy_result_files(network_name: str, vehicle_types: VehicleType,
                      controlled_percentages: int, vehicles_per_lane: int):
    """Copies data collections, link segments, SSMs, Risky Maneuvers and
    Violations files to a similar folder structure in Google Drive. """

    network_relative_address = get_relative_address_from_network_name(
        network_name)
    percentage_folder = create_percent_folder_name(
        [controlled_percentages], [vehicle_types])
    vehicles_per_lane_folder = (
        create_vehs_per_lane_folder_name(
            vehicles_per_lane))
    source_dir = os.path.join(get_networks_folder(), network_relative_address,
                              percentage_folder, vehicles_per_lane_folder)

    target_dir = os.path.join(get_shared_folder(), network_relative_address,
                              percentage_folder, vehicles_per_lane_folder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    all_file_names = os.listdir(source_dir)
    for file_name in all_file_names:
        file_extension = file_name.split('.')[-1]
        if file_extension in {'csv', 'att'}:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)


def get_file_name_from_network_name(network_name):
    if network_name in _network_names_map:
        network_name = _network_names_map[
            network_name]
    elif network_name in _network_names_map.values():
        pass
    else:
        raise ValueError('Network "{}" is not in the list of valid '
                         'simulations\nCheck whether the network exists  '
                         'and add it to the VissimInterface attribute '
                         'existing_networks'.
                         format(network_name))
    return network_name


def get_network_name_from_file_name(network_file):
    return list(_network_names_map.keys())[
        list(_network_names_map.values()).index(network_file)]


def get_relative_address_from_network_name(network_name):
    return os.path.join(_network_relative_folders_map[network_name],
                        _network_names_map[network_name])


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
