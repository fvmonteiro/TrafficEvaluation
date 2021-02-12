import os
import pandas as pd
import pywintypes
import warnings
import win32com.client as com


class VissimInterface:
    # vissim_networks_folder = '.\\..\\VISSIM_networks'  # relative path not working but preferred
    networks_folder = 'C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation\\VISSIM_networks'
    vissim_net_ext = '.inpx'
    vissim_layout_ext = '.layx'

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
                     {'link': 9, 'lane': 2, 'coordinate': 10, 'incident_start_time': 30, 'incident_end_time': 10000},
                     {'link': 9, 'lane': 2, 'coordinate': 10, 'incident_start_time': 780, 'incident_end_time': 1980}]

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
