import os
import pandas as pd
from sodapy import Socrata
from classesForSSM import VissimInterface


# TODO: should this be an abstract class?
class DataReader:
    column_names = None

    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name

    def load_data(self):
        pass

    def load_from_csv(self):
        """
        Loads csv vehicle records from simulations of a chosen network
        :return: dictionary of dataframes
        """
        sim_output = dict()
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and (self.file_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                sim_output[os.path.splitext(file_name)[0]] = pd.read_csv(file)
        return sim_output


class VissimDataReader(DataReader):

    def __init__(self, file_name):
        DataReader.__init__(self, VissimInterface.networks_folder, file_name)
        self.column_names = ('time', 'number', 'veh type', 'link', 'lane', 'x', 'vx', 'y', 'leader number', 'delta x')

    def load_data(self):
        """
        Loads raw vehicle records from simulations of a chosen network
        :return: dictionary of dataframes
        """
        sim_output = dict()
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.fzp') and (self.file_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                # Skip header lines
                for line in file:
                    if line.startswith('$VEHICLE'):
                        break
                sim_output[os.path.splitext(file_name)[0]] = pd.read_csv(file, sep=';', names=self.column_names)

        return sim_output

    def load_max_decel_data(self):
        """
        Loads data describing maximum deceleration distribution per vehicle type and velocity
        :return: dataframe with double index
        """
        return pd.read_csv(os.path.join(self.data_dir, 'max_decel_data.csv'), index_col=['veh_type', 'vel'])


class NGSIMDataReader(DataReader):
    ngsim_dir = 'C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation\\NGSIM\\'
    location_switch = {'us-101': 'US - 101 - LosAngeles - CA\\us - 101 - vehicle - trajectory - data'}
    interval_switch = {1: '0750am-0805am', 2: '0805am-0820am', 3: '0820am-0835am'}

    def __init__(self, location, interval):
        try:
            data_dir = os.path.join(self.location_switch[location], self.interval_switch[interval])
            file_name = 'trajectories-' + self.interval_switch[interval] + '.csv'
        except KeyError:
            print('[NGSIMDataReader] KeyError: location {}, interval {} not found'.format(location, interval))
            data_dir = None
            file_name = None
        DataReader.__init__(self, data_dir, file_name)

    def load_data(self):
        pass


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
