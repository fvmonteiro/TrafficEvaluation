import pywintypes
import win32com.client as com
import os
import pandas as pd
import warnings
import numpy as np
from scipy.stats import truncnorm


class VissimInterface:
    # vissim_networks_folder = '.\\..\\VISSIM_networks'  # relative path not working but preferred
    networks_folder = 'C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation\\VISSIM_networks'
    vissim_net_ext = '.inpx'
    vissim_layout_ext = '.layx'
    column_names = ('time', 'number', 'veh type', 'link', 'lane', 'x', 'vx', 'y', 'leader number', 'delta x')

    # Note: list attributes should be defined inside the constructor. Otherwise they are shared by all instances of
    # the class

    def __init__(self, network_file, layout_file=None):
        self.network_file = network_file
        self.layout_file = network_file if layout_file is None else layout_file
        self.vissim = None

    def open_simulation(self):
        """
        Opens a VISSIM scenario and optionally sets some parameters.
        :return: None
        """

        net_full_path = os.path.join(self.networks_folder, self.network_file + self.vissim_net_ext)
        vissim_id = "Vissim.Vissim"  # "VISSIM.Vissim.1000" # Useful if more than one Vissim version installed

        if os.path.isfile(net_full_path):
            # Connect to the COM server, which opens a new Vissim window
            # vissim = com.gencache.EnsureDispatch("Vissim.Vissim")  # if connecting for the first time ever
            print("Client: Creating a Vissim instance")
            self.vissim = com.Dispatch(vissim_id)
            print("Client: Loading file")
            self.vissim.LoadNet(net_full_path)
            layout_full_path = os.path.join(self.networks_folder, self.layout_file + self.vissim_layout_ext)
            if os.path.isfile(layout_full_path):
                self.vissim.LoadLayout(layout_full_path)
            else:
                print('Client: Layout file {} not found.'.format(net_full_path))
        else:
            print('Client: File {} not found. Exiting program.'.format(net_full_path))
            # sys.exit()  # better to use this?
            return

    def run_toy_scenario(self, sim_params=None, veh_volumes=None):
        """
        Runs a VISSIM scenario. Vehicle results are automatically saved.
        :param sim_params: dictionary with {param_name: param_value}. Check VISSIM COM docs for possible parameters
        :param veh_volumes: vehicle input (veh/hr). Dictionary like {veh input name: input value}
        :return: None
        """
        # TODO: after initial tests, we should save results according to the simulation parameters, that is, create
        #  folders for each vehicle input volume and change file names to reflect the random seed used

        if self.vissim is None:
            self.open_simulation()
        if sim_params is None:  # Simulation will use previously set parameters
            sim_params = dict()
        if veh_volumes is None:  # No change to previously used vehicle inputs
            veh_volumes = dict()

        # Setting simulation parameters
        print("Client: Setting simulation parameters")
        for param_name, param_value in sim_params.items():
            self.vissim.Simulation.SetAttValue(param_name, param_value)

        evaluation = self.vissim.Evaluation
        # First we check if vehicle records are being exported and get a chance to set it to true if we forgot
        veh_rec_saved = evaluation.AttValue('VehRecWriteFile')  # ensure we are saving vehicle records
        if not veh_rec_saved:
            print('Current  configuration does not export vehicle records\n'
                  'You can open VISSIM now and change this.')
            user_decision = input('Press [y] to continue and run simulation or [n] to stop execution')
            if user_decision != 'y':
                self.vissim = None
                return

        # If vehicle records are being exported, we check whether the necessary variables are selected
        if evaluation.GetAttValue('VehRecWriteFile'):
            needed_variables = {'SIMSEC', 'NO', 'LANE\LINK\\NO', 'LANE\INDEX', 'POS', 'SPEED', 'POSLAT', 'FOLLOWDIST',
                                'VEHTYPE', 'LEADTARGNO'}
            att_selection_container = evaluation.VehRecAttributes
            recorded_variables = set()
            for att_selection in att_selection_container:
                recorded_variables.add(att_selection.AttValue('AttributeID'))
            if not needed_variables.issubset(recorded_variables):
                missing = needed_variables.difference(recorded_variables)
                warnings.warn('Current evaluation configuration does not export:{}'
                              '\nPlease open VISSIM and select those attributes.'.format(str(missing)))
                user_decision = input('Press [y] to continue and run simulation or [n] to stop execution')
                if user_decision != 'y':
                    self.vissim = None
                    return

        # Vehicle inputs - change to different function
        veh_inputs = self.vissim.Net.VehicleInputs.GetAll()  # TODO: check if GetAll() is necessary
        for vi in veh_inputs:
            vi_name = vi.AttValue('Name')
            if vi_name in veh_volumes:
                print('Vehicle input {} set to {}'.format(vi_name, veh_volumes[vi_name]))
                veh_inputs[vi].SetAttValue('Volume(1)', veh_volumes[vi_name])

        evaluation.SetAttValue('VehRecFromTime', 0)  # when simulation gets longer, ignore warm-up time
        # Run
        print('Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Simulation done.')
        self.close_vissim()

    def run_i710_simulation(self, sim_params, idx_scenario, demand, save_veh_rec=False):
        """
        Run PTV VISSIM simulation using given arguments
        :param sim_params: dictionary with {param_name: param_value}. Check VISSIM COM docs for possible parameters
        :param idx_scenario: index of simulation scenarios[int]
        :param demand: vehicle input (veh/hr) [int]
        :param save_veh_rec: Defines if VISSIM should save a vehicle record file [boolean]
        """

        # Definition of scenarios
        scenarios = [{'link': 9, 'lane': 2, 'coordinate': 10, 'incident_start_time': 30, 'incident_end_time': 30},
                     {'link': 9, 'lane': 2, 'coordinate': 10, 'incident_start_time': 30, 'incident_end_time': 3570},
                     {'link': 9, 'lane': 2, 'coordinate': 10, 'incident_start_time': 600, 'incident_end_time': 1800}]

        # Incident time period
        incident_start_time = scenarios[idx_scenario]['incident_start_time']
        incident_end_time = scenarios[idx_scenario]['incident_end_time']

        if self.vissim is None:
            self.open_simulation()
        # COM lines
        try:
            # Setting simulation parameters
            print("Client: Setting simulation parameters")
            for param_name, param_value in sim_params.items():
                self.vissim.Simulation.SetAttValue(param_name, param_value)

            self.vissim.Evaluation.SetAttValue('VehRecWriteFile', save_veh_rec)

            # Set vehicle input
            net = self.vissim.Net
            veh_inputs = net.VehicleInputs
            veh_input = veh_inputs.ItemByKey(1)
            veh_input.SetAttValue('Volume(1)', demand)

            # Get link and vehicle objects
            links = net.Links
            vehs = net.Vehicles

            # Make sure that all lanes are open
            for link in links:
                # Open all lanes
                for lane in link.Lanes:
                    lane.SetAttValue('BlockedVehClasses', '')

            bus_no = 2  # No. of buses used to block the lane
            bus_array = [0] * bus_no

            # Get total simulation time and set a break point
            sim_time = self.vissim.Simulation.AttValue('SimPeriod')
            self.vissim.Simulation.SetAttValue("SimBreakAt", incident_start_time)

            # Start simulation
            print("Scenario:", idx_scenario)
            print("Random Seed:", self.vissim.Simulation.AttValue('RandSeed'))
            print("Client: Starting simulation")
            self.vissim.Simulation.RunContinuous()

            # Create incident (simulation stops automatically at the SimBreakAt value)
            if sim_time >= incident_start_time:
                print('Client: Creating incident')
                for i in range(bus_no):  # xrange
                    bus_array[i] = vehs.AddVehicleAtLinkPosition(300, scenarios[idx_scenario]['link'],
                                                                 scenarios[idx_scenario]['lane'],
                                                                 scenarios[idx_scenario]['coordinate'] + 20 * i, 0,
                                                                 0)
                self.vissim.Simulation.SetAttValue("SimBreakAt", incident_end_time)
                print('Client: Running again')
                self.vissim.Simulation.RunContinuous()

            # Remove the incident
            if sim_time >= incident_end_time:
                print('Client: Removing incident')
                for veh in bus_array:
                    vehs.RemoveVehicle(veh.AttValue('No'))
                # open all closed lanes
                for link in links:
                    for lane in link.Lanes:
                        lane.SetAttValue('BlockedVehClasses', '')
                print('Client: Running again')
                self.vissim.Simulation.RunContinuous()

            print('Simulation done')

        except pywintypes.com_error as err:
            self.vissim = None
            print("err=", err)

            # Error
            # The specified configuration is not defined within VISSIM.
            #
            # Description
            # Some methods for evaluations results need a previously configuration for data collection.
            # The error occurs when requesting results that have not been previously configured.
            # For example, using the GetSegmentResult() method of the ILink interface to request
            # density results can end up with this error if the density has not been requested within the configuration

    def get_max_decel_data(self):
        """
        Checks the vehicle types used in simulation and creates a Panda dataframe with max deceleration values by
        vehicle type and speed
        :return: None
        """

        if self.vissim is None:
            self.open_simulation()

        # Stats of the uniform distribution (0, 1) from which VISSIM draws numbers
        vissim_mu = 0.5
        vissim_sigma = 0.15

        # Initializations
        veh_comp_container = self.vissim.Net.VehicleCompositions
        data = list()
        known_veh_type = set()

        for veh_comp in veh_comp_container:
            for rel_flow in veh_comp.VehCompRelFlows:
                veh_type = rel_flow.VehType
                if veh_type.AttValue('No') not in known_veh_type:
                    known_veh_type.add(veh_type.AttValue('No'))
                    for point in veh_type.MaxDecelFunc.DecelFuncDataPts:
                        # Scaling from original random variable to max deceleration random variable
                        scale_above = (point.AttValue('yMax') - point.AttValue('Y')) / (vissim_mu - 0)
                        std_above = scale_above * vissim_sigma
                        scale_below = (point.AttValue('Y') - point.AttValue('yMin')) / (vissim_mu - 0)
                        std_below = scale_below * vissim_sigma
                        std_mean = (std_above + std_below) / 2
                        if abs(std_above - std_below) > 0.1 * std_mean:
                            warnings.warn('Std Devs above and below median vary significantly. ' +
                                          'Averaging them could generate unrealistic results.')
                        # Formula for adjusting parameters to standardized truncated gaussian
                        a = (point.AttValue('yMin') - point.AttValue('Y')) / std_mean
                        b = (point.AttValue('yMax') - point.AttValue('Y')) / std_mean
                        data.append([veh_type.AttValue('No'), point.AttValue('X'), point.AttValue('Y'), std_mean, a, b])

        max_decel_df = pd.DataFrame(data=data, columns=['veh_type', 'vel', 'mean', 'std', 'norm_min', 'norm_max'])
        max_decel_df.set_index(['veh_type', 'vel'], inplace=True)
        max_decel_df.to_csv(os.path.join(self.networks_folder, 'max_decel_data.csv'))

        return max_decel_df

    def close_vissim(self):
        self.vissim = None


class DataReader:
    column_names = VissimInterface.column_names

    # ('time', 'number', 'veh type', 'link', 'lane', 'x', 'vx', 'y', 'leader number', 'delta x')

    def __init__(self, data_dir, sim_name):
        self.data_dir = data_dir
        self.sim_name = sim_name
        self.sim_output = None  # dictionary of dataframes
        self.max_decel = None

    def load_data_from_vissim(self):
        self.sim_output = dict()
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.fzp') and (self.sim_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                # Skip header lines
                for line in file:
                    if line.startswith('$VEHICLE'):
                        break
                self.sim_output[os.path.splitext(file_name)[0]] = pd.read_csv(file, sep=';', names=self.column_names)

    def load_data_from_csv(self):
        self.sim_output = dict()
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and (self.sim_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                self.sim_output[os.path.splitext(file_name)[0]] = pd.read_csv(file)

    def load_max_decel_data(self):
        self.max_decel = pd.read_csv(os.path.join(self.data_dir, 'max_decel_data.csv'), index_col=['veh_type', 'vel'])

    def get_single_dataframe(self):
        # Used for tests and checks
        for df in self.sim_output.values():
            return df

    def post_process_output(self):
        """
        Adds data to the dataframe. For now, just relative speed
        :return: None
        """

        warm_up_time = 10  # [s]

        # 1. Compute relative velocity to the vehicle's leader (own vel minus leader vel)
        # Note: we need this function because VISSIM's output 'SpeedDiff' is not always correct. It has been observed to
        # equal the vehicle's own speed at the previous time step.
        for name, df in self.sim_output.items():
            if 'delta v' in df.columns:
                print('Dataframe {} already has delta v'.format(name))
                pass
            df.set_index('time', inplace=True)
            df.drop(index=[i for i in df.loc[df.index < warm_up_time].index], inplace=True)
            df['leader number'].fillna(df['number'], inplace=True, downcast='infer')
            n_veh = max(df['number'])
            identity = np.eye(n_veh)

            for t in np.unique(df.index):
                adj_matrix = np.zeros([n_veh, n_veh])
                vel_vector = np.zeros(n_veh)
                veh_idx = [i - 1 for i in df.loc[t, 'number']]
                adj_matrix[veh_idx, df.loc[t, 'leader number'].values - 1] = -1
                vel_vector[veh_idx] = df.loc[t, 'vx']
                df.loc[t, 'delta v'] = np.matmul((identity + adj_matrix), vel_vector)[veh_idx]
                # df.loc[t, 'leader type'] =

        # TODO: 2. Add maximum deceleration at that time instant. Decide how to define max brake

    def save_to_csv(self):
        for filename, df in self.sim_output.items():
            df.to_csv(os.path.join(self.data_dir, filename + '.csv'))


class SSMAnalyzer:

    def __init__(self):
        pass

    # TODO: check the computations below for a couple of vehicles and ensure they're properly coded
    @staticmethod
    def include_ttc(df_dict):
        """
        Includes Time To Collision (TTC) as a column in dataframes in the dictionary
        :param df_dict: dictionary of dataframes loaded using a DataReader
        :return: None
        """
        # TTC = deltaX/deltaV if follower is faster; otherwise infinity

        for df in df_dict.values():
            df['TTC'] = df['delta x'] / df['delta v']
            df.loc[df['delta v'] < 0, 'TTC'] = float('inf')
            ttc_mean = df.loc[df['TTC'] < float('inf'), 'TTC'].mean()
            print('Mean TTC: ', ttc_mean)

    @staticmethod
    def include_drac(df_dict):
        """
        Includes Deceleration Rate to Avoid Collision (DRAC) as a column in dataframes in the dictionary
        :param df_dict: dictionary of dataframes loaded using a DataReader
        :return: None
        """
        # DRAC = deltaV^2/(2.deltaX), if follower is faster; otherwise zero

        for df in df_dict.values():
            df['DRAC'] = df['delta v'] ** 2 / 2 / df['delta x']
            df.loc[df['delta v'] < 0, 'DRAC'] = 0
            drac_mean = df.loc[df['DRAC'] > 0, 'DRAC'].mean()
            print('Mean DRAC: ', drac_mean)

    @staticmethod
    def include_cpi(df_dict, max_decel_df, default_vissim=True):
        """
        Includes Crash Probability Index (CPI) as a column in dataframes in the dictionary
        :param df_dict: dictionary of dataframes loaded using a DataReader
        :param max_decel_df: dataframe specifying maximum deceleration per vehicle type and speed
        :param default_vissim: boolean to identify if data was generated using default VISSIM deceleration parameters
        :return: None
        """
        # CPI = Prob(DRAC > MADR), where MADR is the maximum available deceleration rate
        # Formally, we should check the truncated Gaussian parameters for each velocity. However, the default VISSIM
        # max decel is a linear function of the velocity and the other three parameters are constant. We make use of
        # this to speed up this function.

        for df in df_dict.values():
            df['CPI'] = 0
            veh_type_array = np.unique(df['veh type'])
            for veh_type in veh_type_array:
                idx = (df['veh type'] == veh_type) & (df['DRAC'] > 0)
                if not default_vissim:
                    a_array = []
                    b_array = []
                    madr_array = []
                    std_array = []
                    for vel in df.loc[idx, 'vx']:
                        row = max_decel_df.loc[veh_type, round(vel, -1)]
                        a_array.append(row['norm_min'])
                        b_array.append(row['norm_max'])
                        madr_array.append(-1 * row['mean'])
                        std_array.append(row['std'])
                    df.loc[idx, 'CPI'] = truncnorm.cdf(df.loc[idx, 'DRAC'], a=a_array, b=b_array,
                                                       loc=madr_array, scale=std_array)
                else:
                    first_row = max_decel_df.loc[veh_type, 0]
                    possible_vel = max_decel_df.loc[veh_type].index
                    min_vel = 0
                    max_vel = max(possible_vel)
                    decel_min_vel = max_decel_df.loc[veh_type, min_vel]['mean']
                    decel_max_vel = max_decel_df.loc[veh_type, max_vel]['mean']
                    madr_array = decel_min_vel + (decel_max_vel - decel_min_vel) / max_vel * df.loc[idx, 'vx']
                    df.loc[idx, 'CPI'] = truncnorm.cdf(df.loc[idx, 'DRAC'], a=first_row['norm_min'],
                                                       b=first_row['norm_max'], loc=(-1) * madr_array,
                                                       scale=first_row['std'])

    @staticmethod
    def include_safe_gaps(df_dict, max_decel_df, rho=0.2, free_flow_vel=30):
        """
        Includes safe gap and time headway-based gap as columns in dataframes in the dictionary
        :param df_dict: dictionary of dataframes loaded using a DataReader
        :param max_decel_df: dataframe specifying maximum deceleration per vehicle type and speed (NOT BEING USED)
        :param rho: expected maximum relative velocity, defined as vE - vL <= rho.vE
        :param free_flow_vel: should be given in m/s
        :return:
        """
        # Safe gap is the worst-case collision-free gap (as close as possible to minimum gap to avoid collision under
        # emergency braking)
        # Time headway-based gap (th gap) is the linear overestimation of the safe gap

        # TODO: braking parameters set based on typical values. Consider extracting from VISSIM
        # TODO: get leader max brake from its type
        # TODO: consider case where (ego max brake) > (leader max brake)

        # Veh types:
        veh_types = set()
        for df in df_dict.values():
            [veh_types.add(v_type) for v_type in np.unique(df['veh type'])]

        # Parameters
        mps_to_kph = 3.6
        accel_t0 = 0.5
        max_brake = {100: 6.5, 200: 5.5}
        max_jerk = {100: 50, 200: 30}
        tau_d = 0.2
        veh_params = pd.DataFrame(columns=['max_brake', 'lambda0', 'lambda1'])
        for key in max_brake.keys():
            tau_j = (accel_t0 + max_brake[key]) / max_jerk[key]
            lambda1 = (accel_t0 + max_brake[key]) * (tau_d + tau_j / 2)
            lambda0 = -(accel_t0 + max_brake[key]) / 2 * (tau_d ** 2 + tau_d * tau_j + tau_j ** 2 / 3)
            veh_params.loc[key] = [max_brake[key], lambda0, lambda1]

        for df in df_dict.values():
            df['safe gap'] = 0
            change_idx = df['number'] != df['leader number']
            ego_vel = df['vx'] / mps_to_kph
            leader_vel = ego_vel - df['delta v'] / mps_to_kph
            for v_type in veh_types:
                max_brake_ego, lambda0, lambda1 = veh_params.loc[v_type]
                gamma = 1  # max_brake_ego/max_brake_leader
                time_headway = ((1 / 2 - (1 - rho) ** 2 / 2 / gamma) * free_flow_vel + lambda1) / max_brake_ego
                standstill_gap = lambda1 ** 2 / 2 / max_brake_ego + lambda0

                df.loc[(df['veh type'] == v_type) & change_idx, 'safe gap'] = \
                    ego_vel ** 2 / 2 / max_brake_ego - leader_vel ** 2 / 2 / veh_params.loc[v_type, 'max_brake'] \
                    + lambda1 * ego_vel / max_brake_ego + lambda1 ** 2 / 2 / max_brake_ego + lambda0
                df.loc[(df['veh type'] == v_type) & change_idx, 'time headway gap'] = \
                    time_headway * ego_vel + standstill_gap
        print(df.loc[df['safe gap'] > 0, ['delta v', 'safe gap', 'time headway gap']])
