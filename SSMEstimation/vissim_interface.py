import os
import pandas as pd
import pywintypes
import warnings
import win32com.client as com


class VissimInterface:
    # vissim_networks_folder = '.\\..\\VISSIM_networks'  # relative path not
    # working but preferred
    # IN_AND_OUT = 'highway_in_and_out_lanes'
    # I710 = 'I710-MultiSec-3mi'
    # US_101 = 'US_101'
    networks_folder = ('C:\\Users\\fvall\\Documents\\Research'
                       '\\TrafficSimulation\\VISSIM_networks')
    vissim_net_ext = '.inpx'
    vissim_layout_ext = '.layx'
    existing_networks = {'toy': 'highway_in_and_out_lanes',
                         'i710': 'I710-MultiSec-3mi',
                         'us101': 'US_101'}

    def __init__(self, network_file, layout_file=None):
        if network_file in self.existing_networks:
            network_file = self.existing_networks[network_file]
        elif network_file in self.existing_networks.values():
            pass
        else:
            raise ValueError('Network "{}" is not in the list of valid '
                             'simulations\nCheck whether the network exists  '
                             'and it to the class attribute existing_networks'.
                             format(network_file))
        self.network_file = network_file
        self.layout_file = network_file if layout_file is None else layout_file
        self.vissim = None

    def open_simulation(self):
        """
        Opens a VISSIM scenario and optionally sets some parameters.
        :return: None
        """

        net_full_path = os.path.join(self.networks_folder,
                                     self.network_file + self.vissim_net_ext)
        vissim_id = "Vissim.Vissim"  # "VISSIM.Vissim.1000" # Useful if more
        # than one Vissim version installed

        if os.path.isfile(net_full_path):
            # Connect to the COM server, which opens a new Vissim window
            # vissim = com.gencache.EnsureDispatch("Vissim.Vissim")  # if
            # connecting for the first time ever
            print("Client: Creating a Vissim instance")
            self.vissim = com.Dispatch(vissim_id)
            print("Client: Loading file")
            self.vissim.LoadNet(net_full_path)
            layout_full_path = os.path.join(self.networks_folder,
                                            self.layout_file
                                            + self.vissim_layout_ext)
            if os.path.isfile(layout_full_path):
                self.vissim.LoadLayout(layout_full_path)
            else:
                print('Client: Layout file {} not found.'.
                      format(net_full_path))
        else:
            print('Client: File {} not found. Exiting program.'.
                  format(net_full_path))
            # sys.exit()  # better to use this?
            return

    # RUN NETWORK #
    def run_toy_scenario(self, in_flow_input=None, main_flow_input=None,
                         sim_params=None, save_veh_rec=True,
                         save_ssam_file=True):
        """
        Runs a VISSIM scenario. Vehicle results are automatically saved.
        :param sim_params: dictionary with {param_name: param_value}. Check
        VISSIM COM docs for possible parameters
        :param veh_volumes: vehicle input (veh/hr). Dictionary like
        {veh input name: input value}
        :return: None
        """
        # TODO: after initial tests, we should save results according to the
        #  simulation parameters, that is, create folders for each vehicle
        #  input volume and change file names to reflect the random seed used

        if self.vissim is None:
            self.open_simulation()
        if sim_params is None:  # Simulation will use previously set parameters
            sim_params = dict()
        veh_volumes = dict()
        if in_flow_input is not None:
            veh_volumes['in_flow'] = in_flow_input
        if main_flow_input is not None:
            veh_volumes['main_flow'] = main_flow_input

        print("Client: Setting simulation parameters")
        for param_name, param_value in sim_params.items():
            try:
                self.vissim.Simulation.SetAttValue(param_name, param_value)
            except AttributeError as err:
                self.close_vissim()
                print("Issue setting parameter {} at value {}".
                      format(param_name, param_value))
                print("err=", err)

        self.set_saving_preferences(save_veh_rec, save_ssam_file)
        if save_veh_rec and not self.check_saved_variables():
            return
        self.set_vehicle_inputs_by_name(veh_volumes)

        self.vissim.Evaluation.SetAttValue('VehRecFromTime', 0)
        # when simulation gets longer, ignore warm-up time
        # Run
        print('Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Simulation done.')

    def run_us_101_simulation(self, highway_input=8200, ramp_input=450,
                              speed_limit_per_lane=None,
                              save_veh_rec=False, save_ssam_file=False):
        """
        Run PTV VISSIM simulation of the US 101
        :param highway_input: vehicle input at the beginning of the highway
        :param ramp_input: vehicle input at the beginning of the highway
        :param speed_limit_per_lane: reduced speed area values. Must be a 5
        element list from fastest to slowest lane
        :param save_veh_rec: Defines if VISSIM should save a vehicle record
        file [boolean]
        :param save_ssam_file: Defines if VISSIM should save the file to use
         with the SSAM software [boolean]
        """

        veh_inputs = {'highway_input': highway_input,
                      'ramp_input': ramp_input}
        if speed_limit_per_lane is None:
            speed_limit_per_lane = [40, 30, 25, 25, 25]

        if self.vissim is None:
            self.open_simulation()

        print("Client: Setting simulation parameters")
        self.set_saving_preferences(save_veh_rec, save_ssam_file)
        if save_veh_rec and not self.check_saved_variables():
            return
        self.set_vehicle_inputs_by_name(veh_inputs)

        reduced_speed_areas = self.vissim.Net.ReducedSpeedAreas
        for i in range(len(reduced_speed_areas)):
            reduced_speed_areas[i].SetAttValue(
                'DesSpeedDistr(10)', speed_limit_per_lane[i])

        print('Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Simulation done.')

    def run_i710_simulation(self, sim_params, idx_scenario, demand,
                            save_veh_rec=False, save_ssam_file=False):
        """
        Run PTV VISSIM simulation using given arguments
        :param sim_params: dictionary with {param_name: param_value}. Check
        VISSIM COM docs for possible parameters
        :param idx_scenario: index of simulation scenarios[int]
        :param demand: vehicle input (veh/hr) [int]
        :param save_veh_rec: Defines if VISSIM should save a vehicle record
        file [boolean]
        :param save_ssam_file: Defines if VISSIM should save the file to use
         with the SSAM software [boolean]
        """

        # Definition of scenarios
        scenarios = [{'link': 9, 'lane': 2, 'coordinate': 10,
                      'incident_start_time': 30, 'incident_end_time': 30},
                     {'link': 9, 'lane': 2, 'coordinate': 10,
                      'incident_start_time': 30, 'incident_end_time': 10000},
                     {'link': 9, 'lane': 2, 'coordinate': 10,
                      'incident_start_time': 780, 'incident_end_time': 1980}]

        # Incident time period
        incident_start_time = scenarios[idx_scenario]['incident_start_time']
        incident_end_time = scenarios[idx_scenario]['incident_end_time']

        if self.vissim is None:
            self.open_simulation()

        # COM lines
        self.vissim.SuspendUpdateGUI()
        print("Client: Setting simulation parameters")
        self.set_saving_preferences(save_veh_rec, save_ssam_file)
        if save_veh_rec and not self.check_saved_variables():
            # TODO: include some message
            return
        for param_name, param_value in sim_params.items():
            self.vissim.Simulation.SetAttValue(param_name, param_value)
        # Set vehicle input
        net = self.vissim.Net
        veh_inputs = net.VehicleInputs
        veh_input = veh_inputs.ItemByKey(1)
        veh_input.SetAttValue('Volume(1)', demand)
        # TODO: can probably remove the try block (or substitute the exception)
        try:
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
            self.vissim.Simulation.SetAttValue("SimBreakAt",
                                               incident_start_time)

            # Start simulation
            print("Scenario:", idx_scenario)
            print("Random Seed:", self.vissim.Simulation.AttValue('RandSeed'))
            print("Client: Starting simulation")
            self.vissim.Simulation.RunContinuous()

            # Create incident (simulation stops automatically at the
            # SimBreakAt value)
            bus_no = 2  # No. of buses used to block the lane
            bus_array = [0] * bus_no
            if sim_time >= incident_start_time:
                print('Client: Creating incident')
                for i in range(bus_no):
                    bus_array[i] = vehs.AddVehicleAtLinkPosition(
                                        300, scenarios[idx_scenario]['link'],
                                        scenarios[idx_scenario]['lane'],
                                        scenarios[idx_scenario]['coordinate']
                                        + 20 * i, 0, 0)
                self.vissim.Simulation.SetAttValue("SimBreakAt",
                                                   incident_end_time)
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

            # Description
            # Some methods for evaluations results need a previously
            # configuration for data collection. The error occurs when
            # requesting results that have not been previously configured.
            # For example, using the GetSegmentResult() method of the ILink
            # interface to request density results can end up with this error
            # if the density has not been requested within the configuration
        finally:
            self.vissim.ResumeUpdateGUI()

    def test_lane_access(self):
        """Code to figure out how to read lane by lane information"""
        if self.vissim is None:
            self.open_simulation()

        self.set_saving_preferences(False, False)
        simulation_time_sec = 150
        current_time = 0
        t_data_sec = 30

        # This line is different from Yihang's code, because his code doesn't
        # use the GetAll method. You can select only the links that interest you
        # instead of GetAll
        links = self.vissim.Net.Links.GetAll()
        # Run each time step at a time
        print('Simulation starting.')
        while current_time <= simulation_time_sec:
            self.vissim.Simulation.RunSingleStep()
            current_time = self.vissim.Simulation.AttValue('SimSec')
            if current_time % t_data_sec == 0:
                for link in links:
                    for lane in link.Lanes.GetAll():
                        print('Link number: {}; Lane index: {}; '
                              'Total vehs: {}'.format(
                            link.AttValue('No'), lane.AttValue('Index'),
                            lane.Vehs.Count))
        print('Simulation done.')

    # MULTIPLE SCENARIO RUN #
    def run_us_101_with_different_speed_limits(self, possible_speeds=None):
        if possible_speeds is None:
            possible_speeds = [25, 35, 40]

        n_lanes = 5
        self.open_simulation()
        self.vissim.SuspendUpdateGUI()

        for i in range(len(possible_speeds)-1):
            speed_limit_per_lane = [possible_speeds[0]]*n_lanes
            for j in range(n_lanes):
                speed_limit_per_lane[:j] = [possible_speeds[i+1]]*j
                print('Speed reduction values: ', speed_limit_per_lane)
                self.run_us_101_simulation(
                    speed_limit_per_lane=speed_limit_per_lane)

        self.vissim.ResumeUpdateGUI()

    # HELPER FUNCTIONS #
    def get_max_decel_data(self):
        """
        Checks the vehicle types used in simulation, creates a Panda
        dataframe with max deceleration values by vehicle type and speed, and
        saves the dataframe as csv
        :return: None
        """

        if self.vissim is None:
            self.open_simulation()

        # Stats of the uniform distribution (0, 1) from which VISSIM draws
        # numbers
        vissim_mu = 0.5
        vissim_sigma = 0.15

        # Initializations
        veh_comp_container = self.vissim.Net.VehicleCompositions
        data = list()
        known_veh_type = set()

        for veh_comp in veh_comp_container:
            for rel_flow in veh_comp.VehCompRelFlows:
                veh_type = rel_flow.VehType
                print('Saving data for vehicle type: {}'.
                      format(veh_type.AttValue('Name')))
                if veh_type.AttValue('No') not in known_veh_type:
                    known_veh_type.add(veh_type.AttValue('No'))
                    for point in veh_type.MaxDecelFunc.DecelFuncDataPts:
                        # Scaling from original random variable to max
                        # deceleration random variable
                        scale_above = (point.AttValue('yMax')
                                       - point.AttValue('Y')) / vissim_mu
                        std_above = scale_above * vissim_sigma
                        scale_below = (point.AttValue('Y')
                                       - point.AttValue('yMin')) / vissim_mu
                        std_below = scale_below * vissim_sigma
                        std_mean = (std_above + std_below) / 2
                        if abs(std_above - std_below) > 0.1 * std_mean:
                            warnings.warn('Std Devs above and below median '
                                          'vary significantly. Averaging '
                                          'them could generate unrealistic '
                                          'results.')
                        # Formula for adjusting parameters to standardized
                        # truncated gaussian
                        a = (point.AttValue('yMin')
                             - point.AttValue('Y')) / std_mean
                        b = (point.AttValue('yMax')
                             - point.AttValue('Y')) / std_mean
                        data.append([veh_type.AttValue('No'),
                                     point.AttValue('X'), point.AttValue('Y'),
                                     std_mean, a, b])

        max_decel_df = pd.DataFrame(data=data,
                                    columns=['veh_type', 'vel', 'mean', 'std',
                                             'norm_min', 'norm_max'])
        max_decel_df.set_index(['veh_type', 'vel'], inplace=True)
        max_decel_df.to_csv(os.path.join(self.networks_folder,
                                         'max_decel_data.csv'))

        return max_decel_df

    def set_saving_preferences(self, save_veh_rec=False, save_ssam_file=False):
        self.vissim.Evaluation.SetAttValue('VehRecWriteFile', save_veh_rec)
        self.vissim.Evaluation.SetAttValue('SSAMWriteFile', save_ssam_file)

    def check_saved_variables(self):
        """Checks whether all the necessary vehicle variables are set to be
        saved
        :return: boolean indicating whether to continue run"""
        evaluation = self.vissim.Evaluation
        if evaluation.AttValue('VehRecWriteFile'):
            needed_variables = {'SIMSEC', 'NO', 'VEHTYPE', 'LANE\\LINK\\NO',
                                'LANE\\INDEX', 'SPEED', 'LEADTARGNO',
                                'COORDFRONTX', 'COORDFRONTY',
                                'COORDREARX', 'COORDREARY'}
            att_selection_container = evaluation.VehRecAttributes
            recorded_variables = set()
            for att_selection in att_selection_container:
                recorded_variables.add(att_selection.AttValue('AttributeID'))
            if not needed_variables.issubset(recorded_variables):
                missing = needed_variables.difference(recorded_variables)
                warnings.warn('Current evaluation configuration does not '
                              'export:{} \nPlease open VISSIM and select '
                              'those attributes.'.format(str(missing)))
                user_decision = input('Press [y] to continue and run '
                                      'simulation or [n] to stop execution')
                if user_decision != 'y':
                    self.close_vissim()
                    return False
        return True

    def set_vehicle_inputs_by_name(self, new_veh_inputs=None):
        """
        Sets the several vehicle inputs in the simulation by name.
        :param new_veh_inputs: vehicle input (veh/hr). Dictionary like
        {veh input name: input value}
        """

        if new_veh_inputs is None:
            new_veh_inputs = dict()

        veh_inputs = self.vissim.Net.VehicleInputs.GetAll()
        for vi in veh_inputs:
            vi_name = vi.AttValue('Name')
            if vi_name in new_veh_inputs:
                vi.SetAttValue('Volume(1)', new_veh_inputs[vi_name])
                print('Vehicle input {} set to {}'.
                      format(vi_name, vi.AttValue('Volume(1)')))
                new_veh_inputs.pop(vi_name)
            else:
                print('Vehicle input {} left unchanged at {}'.
                      format(vi_name, vi.AttValue('Volume(1)')))

        for vi_key in new_veh_inputs:
            print('Vehicle input {} was passed as parameter, but not '
                  'found in simulation'.format(vi_key))

    def close_vissim(self):
        self.vissim = None
