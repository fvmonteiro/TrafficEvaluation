import os
import pandas as pd
# from sodapy import Socrata
from vissim_interface import VissimInterface


class DataReader:
    relevant_columns = set(['time', 'veh_id', 'veh_type', 'lane', 'x', 'vx', 'y', 'leader_id', 'delta_x', 'delta_v'])

    def __init__(self, data_dir, base_file_name):
        self.data_dir = data_dir
        self.base_file_name = base_file_name
        self.file_name = self.base_file_name

    def load_data(self):
        pass

    def load_max_decel_data(self):
        pass

    def load_from_csv(self):
        """
        Loads csv vehicle records from simulations of a chosen network
        :return: dictionary of dataframes
        """
        # TODO: instead of looking for all files that match, get only one
        sim_output = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and (self.file_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                sim_output.append(pd.read_csv(file))
        return sim_output.pop()

    def select_relevant_columns(self, data):
        columns_to_drop = []
        for col in data.columns:
            if col not in self.relevant_columns:
                columns_to_drop.append(col)
        data.drop(columns=columns_to_drop, inplace=True)


class VissimDataReader(DataReader):

    def __init__(self, sim_name):
        self.column_names = ['time', 'veh_id', 'veh_type', 'link', 'lane', 'x', 'vx', 'y',
                             'leader_id', 'delta_x']
        DataReader.__init__(self, VissimInterface.networks_folder, sim_name)

    def load_data(self, sim_number=1):
        """
        Loads raw vehicle records from simulations of a chosen network
        :return: pandas dataframes
        """
        # Create a three-character string with trailing zeros and then sim_nums (e.g.: 004, 015, 326)
        num_str = str(sim_number).rjust(3, '0')
        self.file_name = self.base_file_name + '_' + num_str + '.fzp'
        try:
            with open(os.path.join(self.data_dir, self.file_name), 'r') as file:
                # Skip header lines
                for line in file:
                    if line.startswith('$VEHICLE'):
                        break
                sim_output = pd.read_csv(file, sep=';', names=self.column_names, nrows=1000)
                print('Still in tests: loading only 1000 first rows')
        except OSError:
            print('{}: File not found'.format(type(self)))
            sim_output = pd.DataFrame()

        self.select_relevant_columns(sim_output)
        return sim_output

    def load_max_decel_data(self):
        """
        Loads data describing maximum deceleration distribution per vehicle type and velocity
        :return: pandas dataframe with double index
        """
        return pd.read_csv(os.path.join(self.data_dir, 'max_decel_data.csv'), index_col=['veh_type', 'vel'])


class NGSIMDataReader(DataReader):
    ngsim_dir = 'C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation\\NGSIM_original\\'
    location_switch = {'us-101': 'US-101-LosAngeles-CA\\us-101-vehicle-trajectory-data'}
    interval_switch = {1: '0750am-0805am', 2: '0805am-0820am', 3: '0820am-0835am'}
    # ['time', 'veh_id', 'veh_type', 'lane', 'x', 'vx', 'y', 'leader_id', 'delta_x', 'delta_v']
    ngsim_to_reader_naming = {'Global_Time': 'time', 'Vehicle_ID': 'veh_id', 'v_Class': 'veh_type',
                              'Local_Y': 'x', 'v_Vel': 'vx', 'Local_X': 'y',
                              'Preceding': 'leader_id', 'Space_Hdwy': 'delta_x'}

    def __init__(self, location):
        try:
            data_dir = os.path.join(self.ngsim_dir, self.location_switch[location])
            file_name = 'trajectories-'
        except KeyError:
            print('{}: KeyError: location {} not defined'.format(type(self), location))
            data_dir = None
            file_name = None
        DataReader.__init__(self, data_dir, file_name)

    def load_data(self, interval=1):

        if interval not in self.location_switch:
            print('Requested interval not available')
            return pd.DataFrame()

        self.file_name = self.base_file_name + self.interval_switch[interval] + '.csv'
        try:
            with open(os.path.join(self.data_dir, self.file_name), 'r') as file:
                data = pd.read_csv(file)  # , nrows=1000, skiprows=range(1, 10**4))
                # print('Still in tests: loading only 100 rows')
                data.rename(columns=self.ngsim_to_reader_naming, inplace=True)
        except OSError:
            print('{}: File not found'.format(type(self)))
            data = pd.DataFrame()

        self.select_relevant_columns(data)
        return data


class OnLineDataReader(DataReader):
    """"Class to read NGSIM data online, commands from https://dev.socrata.com/foundry/data.transportation.gov/8ect-6jqj
    Data details in:
    https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj"""

    def __init__(self):
        url = "data.transportation.gov"
        database_identifier = "8ect-6jqj"
        DataReader.__init__(self, url, database_identifier)

    def load_data(self, location='us-101', limit='2000'):
        """
        :param location: peachtree, i-80, us-101, lankershim
        :param limit: max number of rows
        :return: pandas dataframe
        """
        # The get function can receive SQL-like parameters to better select the data

        # Unauthenticated client only works with public data sets. Note 'None' in place of application token,
        # and no username or password:
        client = Socrata(self.data_dir, None)
        # Results, returned as JSON from API / converted to Python list of dictionaries by sodapy.
        results = client.get(self.file_name, location=location, limit=limit)
        # Convert to pandas DataFrame
        results_df = pd.DataFrame.from_records(results)

        return results_df
