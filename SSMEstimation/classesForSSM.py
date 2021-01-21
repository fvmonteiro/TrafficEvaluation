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
                self.close_vissim()
                return

        # If vehicle records are being exported, we check whether the necessary variables are selected
        if evaluation.GetAttValue('VehRecWriteFile'):
            needed_variables = {'SIMSEC', 'NO', 'VEHTYPE', 'LANE\LINK\\NO', 'LANE\INDEX', 'POS', 'SPEED', 'POSLAT',
                                'LEADTARGNO', 'FOLLOWDIST'}
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
                    self.close_vissim()
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

            # Get total simulation time and set a break point
            sim_time = self.vissim.Simulation.AttValue('SimPeriod')
            self.vissim.Simulation.SetAttValue("SimBreakAt", incident_start_time)

            # Start simulation
            print("Scenario:", idx_scenario)
            print("Random Seed:", self.vissim.Simulation.AttValue('RandSeed'))
            print("Client: Starting simulation")
            self.vissim.Simulation.RunContinuous()

            # Create incident (simulation stops automatically at the SimBreakAt value)
            bus_no = 2  # No. of buses used to block the lane
            bus_array = [None] * bus_no  # TODO: used to be [0] * bus_no. May no longer work
            if sim_time >= incident_start_time:
                print('Client: Creating incident')
                for i in range(bus_no):
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
            self.close_vissim()
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
                print('Saving data for vehicle type: {}'.format(veh_type.AttValue('Name')))
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
        # self.sim_output = None  # dictionary of dataframes
        # self.max_decel = None

    def load_from_vissim(self):
        """
        Loads raw vehicle records from simulations of a chosen network
        :return: dictionary of dataframes
        """
        sim_output = dict()
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.fzp') and (self.sim_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                # Skip header lines
                for line in file:
                    if line.startswith('$VEHICLE'):
                        break
                # TODO: only keep the temp_column_names until simulation on I710 is run again
                temp_column_names = ('time', 'number', 'link', 'lane', 'x', 'vx', 'y', 'delta x', 'veh type',
                                     'leader number')
                sim_output[os.path.splitext(file_name)[0]] = pd.read_csv(file, sep=';', names=temp_column_names)

        return sim_output

    def load_from_csv(self):
        """
        Loads csv vehicle records from simulations of a chosen network
        :return: dictionary of dataframes
        """
        sim_output = dict()
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and (self.sim_name in file_name):
                file = open(os.path.join(self.data_dir, file_name), 'r')
                sim_output[os.path.splitext(file_name)[0]] = pd.read_csv(file)
        return sim_output

    def load_max_decel_data(self):
        """
        Loads data describing maximum deceleration distribution per vehicle type and velocity
        :return: dataframe with double index
        """
        return pd.read_csv(os.path.join(self.data_dir, 'max_decel_data.csv'), index_col=['veh_type', 'vel'])


class DataAnalyzer:

    def __init__(self, data_dir, sim_name, raw=False):
        self.reader = DataReader(data_dir, sim_name)
        if not raw:
            print('Loading CSV file')
            self.veh_records = self.reader.load_from_csv()
        if raw or len(self.veh_records) == 0:
            print('Loading data saved by VISSIM')
            self.veh_records = self.reader.load_from_vissim()
        self.max_decel_df = None

    def post_process_output(self):
        """
        Process fzp vehicle record data file generated by VISSIM
        :return: None
        """

        # Only use samples
        n_discarded = 10  # number of vehicles

        for name, df in self.veh_records.items():
            # Define warm up time as moment when first 10 vehicles have left simulation
            warm_up_time = max(df.loc[df['number'] <= n_discarded, 'time'])

            # Compute relative velocity to the vehicle's leader (own vel minus leader vel) Note: we need this
            # function because VISSIM's output 'SpeedDiff' is not always correct. It has been observed to equal the
            # vehicle's own speed at the previous time step.
            if 'delta v' in df.columns:
                print('Dataframe {} already has delta v'.format(name))
                continue
            print('Computing delta v for file: {}'.format(name))
            df.set_index('time', inplace=True)
            df.drop(index=[i for i in df.loc[df.index <= warm_up_time].index], inplace=True)
            df['leader number'].fillna(df['number'], inplace=True, downcast='infer')
            n_veh = max(df['number'])
            identity = np.eye(n_veh)

            sim_times = np.unique(df.index)
            percent = 0.1
            for i in range(len(sim_times)):
                t = sim_times[i]
                adj_matrix = np.zeros([n_veh, n_veh])
                vel_vector = np.zeros(n_veh)
                this_time = df.loc[t, 'number']

                veh_idx = [i - 1 for i in this_time]
                adj_matrix[veh_idx, df.loc[t, 'leader number'].values - 1] = -1
                vel_vector[veh_idx] = df.loc[t, 'vx']
                df.loc[t, 'delta v'] = np.matmul((identity + adj_matrix), vel_vector)[veh_idx]
                if i == int(percent * len(sim_times)):
                    print('{}% done'.format(int(i / len(sim_times) * 100)))
                    percent += 0.1

            df.reset_index(inplace=True)

        # TODO: 2. Add maximum deceleration at that time instant. Decide how to define max brake

    def save_to_csv(self, data_dir):
        for filename, df in self.veh_records.items():
            df.to_csv(os.path.join(data_dir, filename + '.csv'), index=False)

    # TODO: check the computations below for a couple of vehicles and ensure they're properly coded
    def include_ttc(self):
        """
        Includes Time To Collision (TTC) as a column in dataframes in the dictionary
        :return: None
        """
        # TTC = deltaX/deltaV if follower is faster; otherwise infinity
        for df in self.veh_records.values():
            if 'delta v' not in df.columns:
                self.post_process_output()
            df['TTC'] = float('inf')
            valid_ttc_idx = df['delta v'] > 0
            df.loc[valid_ttc_idx, 'TTC'] = df.loc[valid_ttc_idx, 'delta x'] / df.loc[valid_ttc_idx, 'delta v']
            # df.loc[df['delta v'] <= 0, 'TTC'] = float('inf')

    def include_drac(self):
        """
        Includes Deceleration Rate to Avoid Collision (DRAC) as a column in dataframes in the dictionary
        :return: None
        """
        # DRAC = deltaV^2/(2.deltaX), if follower is faster; otherwise zero

        for df in self.veh_records.values():
            if 'delta v' not in df.columns:
                self.post_process_output()
            df['DRAC'] = 0
            valid_drac_idx = df['delta v'] > 0
            df.loc[valid_drac_idx, 'DRAC'] = \
                df.loc[valid_drac_idx, 'delta v'] ** 2 / 2 / df.loc[valid_drac_idx, 'delta x']
            # df.loc[df['delta v'] < 0, 'DRAC'] = 0
            # drac_mean = df.loc[df['DRAC'] > 0, 'DRAC'].mean()

    def include_cpi(self, default_vissim=True):
        """
        Includes Crash Probability Index (CPI) as a column in dataframes in the dictionary
        :param default_vissim: boolean to identify if data was generated using default VISSIM deceleration parameters
        :return: None
        """
        # CPI = Prob(DRAC > MADR), where MADR is the maximum available deceleration rate
        # Formally, we should check the truncated Gaussian parameters for each velocity. However, the default VISSIM
        # max decel is a linear function of the velocity and the other three parameters are constant. We make use of
        # this to speed up this function.

        if self.max_decel_df is None:
            self.max_decel_df = self.reader.load_max_decel_data()

        veh_type_array = np.unique(self.max_decel_df.index.get_level_values('veh_type'))
        for df in self.veh_records.values():
            if 'DRAC' not in df.columns:
                self.include_drac()
            df['CPI'] = 0
            # veh_type_array = np.unique(df['veh type'])
            for veh_type in veh_type_array:
                idx = (df['veh type'] == veh_type) & (df['DRAC'] > 0)
                if not default_vissim:
                    a_array = []
                    b_array = []
                    madr_array = []
                    std_array = []
                    for vel in df.loc[idx, 'vx']:
                        row = self.max_decel_df.loc[veh_type, round(vel, -1)]
                        a_array.append(row['norm_min'])
                        b_array.append(row['norm_max'])
                        madr_array.append(-1 * row['mean'])
                        std_array.append(row['std'])
                    df.loc[idx, 'CPI'] = truncnorm.cdf(df.loc[idx, 'DRAC'], a=a_array, b=b_array,
                                                       loc=madr_array, scale=std_array)
                else:
                    first_row = self.max_decel_df.loc[veh_type, 0]
                    possible_vel = self.max_decel_df.loc[veh_type].index
                    min_vel = 0
                    max_vel = max(possible_vel)
                    decel_min_vel = self.max_decel_df.loc[veh_type, min_vel]['mean']
                    decel_max_vel = self.max_decel_df.loc[veh_type, max_vel]['mean']
                    madr_array = decel_min_vel + (decel_max_vel - decel_min_vel) / max_vel * df.loc[idx, 'vx']
                    df.loc[idx, 'CPI'] = truncnorm.cdf(df.loc[idx, 'DRAC'], a=first_row['norm_min'],
                                                       b=first_row['norm_max'], loc=(-1) * madr_array,
                                                       scale=first_row['std'])

    def include_safe_gaps(self, rho=0.2, free_flow_vel=30):
        """
        Includes safe gap and time headway-based gap as columns in dataframes in the dictionary
        :param rho: expected maximum relative velocity, defined as vE - vL <= rho.vE
        :param free_flow_vel: should be given in m/s
        :return: None
        """
        # Safe gap is the worst-case collision-free gap (as close as possible to minimum gap to avoid collision under
        # emergency braking)
        # Time headway-based gap (th gap) is the linear overestimation of the safe gap

        # TODO: braking parameters set based on typical values. Consider extracting from VISSIM
        # TODO: get leader max brake from its type
        # TODO: consider case where (ego max brake) > (leader max brake)

        if self.max_decel_df is None:
            self.max_decel_df = self.reader.load_max_decel_data()

        # Veh types:
        veh_type_array = np.unique(self.max_decel_df.index.get_level_values('veh_type'))

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

        for df in self.veh_records.values():
            df['safe gap'] = 0
            change_idx = df['number'] != df['leader number']
            ego_vel = df['vx'] / mps_to_kph
            leader_vel = ego_vel - df['delta v'] / mps_to_kph
            for veh_type in veh_type_array:
                max_brake_ego, lambda0, lambda1 = veh_params.loc[veh_type]
                gamma = 1  # max_brake_ego/max_brake_leader
                time_headway = ((1 / 2 - (1 - rho) ** 2 / 2 / gamma) * free_flow_vel + lambda1) / max_brake_ego
                standstill_gap = lambda1 ** 2 / 2 / max_brake_ego + lambda0
                veh_type_idx = df['veh type'] == veh_type
                df.loc[veh_type_idx & change_idx, 'safe gap'] = \
                    ego_vel ** 2 / 2 / max_brake_ego - leader_vel ** 2 / 2 / max_brake_ego \
                    + lambda1 * ego_vel / max_brake_ego + lambda1 ** 2 / 2 / max_brake_ego + lambda0
                df[df['safe gap'] < 0] = 0
                df['DTSG'] = df['delta x'] - df['safe gap']
                df.loc[(df['veh type'] == veh_type) & change_idx, 'time headway gap'] = \
                    time_headway * ego_vel + standstill_gap
