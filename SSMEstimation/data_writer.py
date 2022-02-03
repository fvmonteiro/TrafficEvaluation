import os

import numpy as np
import pandas as pd

from vehicle import Vehicle, VehicleType
from vissim_interface import VissimInterface


class DataWriter:

    file_extension = '.csv'

    def __init__(self, data_type_identifier: str, network_name: str,
                 vehicle_type: VehicleType):
        network_name = VissimInterface.get_file_name_from_network_name(
            network_name)
        self.file_base_name = network_name + '_' + data_type_identifier
        self.network_data_dir = os.path.join(
            VissimInterface.get_networks_folder(), network_name)
        self.vehicle_type = vehicle_type.name.lower()

    @staticmethod
    def _save_as_csv(data: pd.DataFrame, folder_path: str,
                     file_name: str):

        full_address = os.path.join(folder_path, file_name)
        try:
            data.to_csv(full_address, index=False)
        except FileNotFoundError:
            print('Couldn''t save at', full_address, '. Saving at: ',
                  VissimInterface.get_networks_folder(), 'instead.')
            data.to_csv(os.path.join(VissimInterface.get_networks_folder(),
                                     file_name), index=False)


class SSMDataWriter(DataWriter):
    """Helps saving aggregated SSM results to files"""
    _data_type_identifier = 'SSM Results'

    def __init__(self, network_name: str, vehicle_type: VehicleType):
        DataWriter.__init__(self, self._data_type_identifier, network_name,
                            vehicle_type)

    def save_as_csv(self, data: pd.DataFrame,
                    controlled_vehicles_percentage: int,
                    vehicles_per_lane: int):

        max_sim_number = data['simulation_number'].iloc[-1]
        num_str = '_' + str(max_sim_number).rjust(3, '0')
        file_name = self.file_base_name + num_str + self.file_extension

        percentage_folder = VissimInterface.create_percent_folder_name(
            controlled_vehicles_percentage, self.vehicle_type)
        vehicles_per_lane_folder = (
            VissimInterface.create_vehs_per_lane_folder_name(
                vehicles_per_lane))
        folder_path = os.path.join(self.network_data_dir, percentage_folder,
                                   vehicles_per_lane_folder)
        self._save_as_csv(data, folder_path, file_name)


class RiskyManeuverWriter(SSMDataWriter):
    _data_type_identifier = 'Risky Maneuvers'

    def __init__(self, network_name: str, vehicle_type: VehicleType):
        DataWriter.__init__(self, self._data_type_identifier, network_name,
                            vehicle_type)


class MergedDataWriter(DataWriter):
    _data_type_identifier = 'Merged Data'

    def __init__(self, network_name: str, vehicle_type: VehicleType):
        DataWriter.__init__(self, self._data_type_identifier, network_name,
                            vehicle_type)

    def save_as_csv(self, data: pd.DataFrame,
                    controlled_vehicles_percentage: int):

        file_name = self.file_base_name + self.file_extension
        percentage_folder = VissimInterface.create_percent_folder_name(
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
    def create_data(vx=20, delta_v=0):
        """
        Creates simple data for tests.

        Only two vehicles, each one with constant speed. The kinematics do not
        have to be correct, i.e., we do not guarantee x = x0 + vt, since this is
        not important for the SSM tests.

        :param vx: Follower longitudinal velocity
        :param delta_v: Difference between leader and follower velocity.
         Note the order: vL - vF
        :return: pandas dataframe with the synthetic data
        """
        max_gap = 10 ** 2
        max_time = 10 ** 2
        gap_interval = 0.01
        vissim_max_delta_x = 250
        n_points = int(max_gap / gap_interval) + 1

        follower = dict()
        leader = dict()
        follower['time'] = np.round(np.linspace(0, max_time, n_points), 2)
        leader['time'] = follower['time']
        follower['veh_id'] = np.ones(n_points, dtype=int)
        leader['veh_id'] = follower['veh_id'] + 1
        follower['veh_type'] = Vehicle.NGSIM_CAR_ID * np.ones(n_points,
                                                              dtype=int)
        leader['veh_type'] = Vehicle.NGSIM_CAR_ID * np.ones(n_points,
                                                            dtype=int)
        follower['lane'] = np.ones(n_points, dtype=int)
        leader['lane'] = np.ones(n_points, dtype=int)
        follower['x'] = np.zeros(n_points)
        leader['x'] = np.round(np.linspace(0, max_gap, n_points), 2)
        follower['vx'] = vx * np.ones(n_points)
        leader['vx'] = vx * np.ones(n_points) + delta_v
        follower['y'] = np.zeros(n_points)
        leader['y'] = np.zeros(n_points)
        follower['leader_id'] = leader['veh_id']
        leader['leader_id'] = leader['veh_id']  # if vehicle has no leader,
        # we set it as its own leader
        follower['delta_x'] = leader['x'] - follower['x']
        leader['delta_x'] = vissim_max_delta_x * np.ones(n_points)
        data = dict()
        for key in follower:
            data[key] = np.hstack((follower[key], leader[key]))

        return pd.DataFrame(data)
