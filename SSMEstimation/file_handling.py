import os
from typing import Union

_network_names_map = {'in_and_out': 'highway_in_and_out_lanes',
                      'in_and_merge': 'highway_in_and_merge',
                      'i710': 'I710-MultiSec-3mi',
                      'us101': 'US_101',
                      'traffic_lights': 'traffic_lights_study'}

_network_relative_folders_map = {'in_and_out': '', 'in_and_merge': '',
                                 'i710': '', 'us101': '',
                                 'traffic_lights': 'traffic_lights_study'}


def get_networks_folder():
    if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
        return ('C:\\Users\\fvall\\Documents\\Research'
                '\\AV_TrafficSimulation\\VISSIM_networks')
    else:
        return ('C:\\Users\\fvall\\Documents\\Research'
                '\\TrafficSimulation\\VISSIM_networks')


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


def create_percent_folder_name(percentage: Union[int, str],
                               vehicle_type: str) -> str:
    """Creates the name of the folder which contains results for the
    given percentage of controlled vehicles (not the full path)"""

    if isinstance(percentage, str):
        percentage_folder = percentage
    else:
        percentage_folder = str(int(percentage)) + '_percent_'
        percentage_folder += vehicle_type if percentage > 0 else ''

    return percentage_folder


def create_vehs_per_lane_folder_name(vehs_per_lane: int):
    """Creates the name of the folder which contains results for the
    given vehicle per lane input (not the full path)"""
    if isinstance(vehs_per_lane, str):
        vehs_per_lane_folder = vehs_per_lane
    else:
        vehs_per_lane_folder = str(int(vehs_per_lane)) + '_vehs_per_lane'
    return vehs_per_lane_folder
