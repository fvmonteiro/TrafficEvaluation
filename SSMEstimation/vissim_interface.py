import os
import time
import warnings

import pandas as pd
import win32com.client as com

from Vehicle import Vehicle


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
    evaluation_periods = {'highway_in_and_out_lanes': 1800,
                          'I710-MultiSec-3mi': 3600,
                          'US_101': 1800}

    def __init__(self):
        self.network_file = None
        self.layout_file = None
        self.vissim = None
        self.open_vissim()

    @staticmethod
    def get_file_name_from_network_name(network_name):
        if network_name in VissimInterface.existing_networks:
            network_name = VissimInterface.existing_networks[network_name]
        elif network_name in VissimInterface.existing_networks.values():
            pass
        else:
            raise ValueError('Network "{}" is not in the list of valid '
                             'simulations\nCheck whether the network exists  '
                             'and add it to the VissimInterface attribute '
                             'existing_networks'.
                             format(network_name))
        return network_name

    def open_vissim(self):
        # Connect to the COM server, which opens a new Vissim window
        # vissim = com.gencache.EnsureDispatch("Vissim.Vissim")  # if
        # connecting for the first time ever
        vissim_id = "Vissim.Vissim"  # "VISSIM.Vissim.1000" # Useful if more
        # than one Vissim version installed
        print("Client: Creating a Vissim instance")
        self.vissim = com.Dispatch(vissim_id)

    def load_simulation(self, network_name: str, layout_file: str = None):
        """ Loads a VISSIM network and optionally sets it to save vehicle
        records and ssam files.

        :param network_name: Network name. Either the actual file name or the
         network nickname. Currently available: toy, i710, us101
        :param layout_file: Optionally defines the layout file for the network
        :return: boolean indicating if simulation was properly loaded
        """

        self.network_file = VissimInterface.get_file_name_from_network_name(
            network_name)
        self.layout_file = network_name if layout_file is None else layout_file
        net_full_path = os.path.join(self.networks_folder,
                                     self.network_file + self.vissim_net_ext)

        if os.path.isfile(net_full_path):
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
            print('Client: File {} not found.'.
                  format(net_full_path))
            # sys.exit()  # better to use this?
            return False

        self.create_results_directory()
        return True

    # RUNNING NETWORKS --------------------------------------------------------#

    def run_toy_scenario(self, in_flow_input: int = None,
                         main_flow_input: int = None):
        """
        Runs a VISSIM scenario. Vehicle results are automatically saved.

        :param in_flow_input: vehicle input (veh/h) of the in ramp (optional)
        :param main_flow_input: vehicle input (veh/h) of the main road
         (optional)
        :return: None
        """
        # TODO: after initial tests, we should save results according to the
        #  simulation parameters, that is, create folders for each vehicle
        #  input volume and change file names to reflect the random seed used

        if (self.vissim.AttValue('InputFile')
                != (self.existing_networks['toy'] + self.vissim_net_ext)):
            print('You must load the toy scenario before running it')
            return

        veh_volumes = dict()
        if in_flow_input is not None:
            veh_volumes['in_flow'] = in_flow_input
        if main_flow_input is not None:
            veh_volumes['main_flow'] = main_flow_input
        if len(veh_volumes) > 0:
            self.set_vehicle_inputs_by_name(veh_volumes)

        self.vissim.Evaluation.SetAttValue('VehRecFromTime', 0)
        # when simulation gets longer, ignore warm-up time
        # Run
        print('Client: Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Client: Simulation done.')

    def run_us_101_simulation(self, highway_input: int = 8200,
                              ramp_input: int = 450,
                              speed_limit_per_lane: list = None):
        """
        Run PTV VISSIM simulation of the US 101

        :param highway_input: vehicle input at the beginning of the highway
        :param ramp_input: vehicle input at the beginning of the highway
        :param speed_limit_per_lane: reduced speed area values. Must be a 5
         element list from fastest to slowest lane
        """

        if (self.vissim.AttValue('InputFile')
                != (self.existing_networks['us101'] + self.vissim_net_ext)):
            print('You must load the us101 scenario before running it')
            return

        veh_inputs = {'highway_input': highway_input,
                      'ramp_input': ramp_input}
        if speed_limit_per_lane is None:
            speed_limit_per_lane = [40, 30, 25, 25, 25]

        self.set_vehicle_inputs_by_name(veh_inputs)

        reduced_speed_areas = self.vissim.Net.ReducedSpeedAreas
        for i in range(len(reduced_speed_areas)):
            reduced_speed_areas[i].SetAttValue(
                'DesSpeedDistr(10)', speed_limit_per_lane[i])

        print('Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Simulation done.')

    def run_i710_simulation(self, scenario_idx: int, demand=5500):
        """
        Run PTV VISSIM simulation using given arguments

        :param scenario_idx: index of simulation scenarios. Follows:
         0: No block, 1: All time block, 2: Temporary block
        :param demand: vehicle input (veh/hr)
        """

        if (self.vissim.AttValue('InputFile')
                != (self.existing_networks['i710'] + self.vissim_net_ext)):
            print('You must load the i710 scenario before running it')
            return

        # Definition of scenarios
        scenarios = [{'link': 9, 'lane': 2, 'coordinate': 10,
                      'incident_start_time': 30, 'incident_end_time': 30},
                     {'link': 9, 'lane': 2, 'coordinate': 10,
                      'incident_start_time': 30, 'incident_end_time': 10000},
                     {'link': 9, 'lane': 2, 'coordinate': 10,
                      'incident_start_time': 780, 'incident_end_time': 1980}]

        # Incident time period
        incident_start_time = scenarios[scenario_idx]['incident_start_time']
        incident_end_time = scenarios[scenario_idx]['incident_end_time']

        # COM lines
        self.vissim.SuspendUpdateGUI()

        # Set vehicle input
        net = self.vissim.Net
        veh_inputs = net.VehicleInputs
        veh_input = veh_inputs.ItemByKey(1)
        veh_input.SetAttValue('Volume(1)', demand)

        # Get link and vehicle objects
        links = net.Links
        vehicles = net.Vehicles

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
        print("Scenario:", scenario_idx)
        print("Random Seed:", self.vissim.Simulation.AttValue('RandSeed'))
        print("Client: Starting simulation")
        self.vissim.Simulation.RunContinuous()

        # Create incident (simulation stops automatically at the
        # SimBreakAt value)
        bus_no = 2  # No. of buses used to block the lane
        bus_array = []
        if sim_time >= incident_start_time:
            print('Client: Creating incident')
            for i in range(bus_no):
                bus_array.append(vehicles.AddVehicleAtLinkPosition(
                    300, scenarios[scenario_idx]['link'],
                    scenarios[scenario_idx]['lane'],
                    scenarios[scenario_idx]['coordinate']
                    + 20 * i, 0, 0))
            self.vissim.Simulation.SetAttValue("SimBreakAt",
                                               incident_end_time)
            print('Client: Running again')
            self.vissim.Simulation.RunContinuous()

        # Remove the incident
        if sim_time >= incident_end_time:
            print('Client: Removing incident')
            for veh in bus_array:
                vehicles.RemoveVehicle(veh.AttValue('No'))
            # open all closed lanes
            for link in links:
                for lane in link.Lanes:
                    lane.SetAttValue('BlockedVehClasses', '')
            print('Client: Running again')
            self.vissim.Simulation.RunContinuous()

        print('Simulation done')
        self.vissim.ResumeUpdateGUI()

    def test_lane_access(self):
        """Code to figure out how to read lane by lane information"""

        if not self.load_simulation('i710'):
            return
        self.set_evaluation_outputs(False, False, False, False)

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
                        print('Link number: {}; Lane index: {}; Total vehs: {}'.
                              format(link.AttValue('No'),
                                     lane.AttValue('Index'),
                                     lane.Vehs.Count))
        print('Simulation done.')

    # MULTIPLE SCENARIO RUN ---------------------------------------------------#

    def run_with_increasing_demand(self, input_increase_per_lane: int,
                                   initial_input_per_lane: int = 500,
                                   max_input_per_lane: int = 2500,
                                   runs_per_input: int = 10,
                                   simulation_period: int = None):
        """ Runs a simulation several times with different seeds, then
        increases the input and reruns the simulation.

        :param input_increase_per_lane: increased vehicle input per
         lane per set of runs
        :param initial_input_per_lane: first tested input per lane
        :param max_input_per_lane: maximum value of input per lane
        :param runs_per_input: how many runs with the same input and varying
         seed
        :param simulation_period: Parameter used for debugging, when we want
        shorter runs. If left at None, the code uses the predefined period
        chosen for longer evaluations.
         """

        if not self.is_some_network_loaded():
            print('Must load a network before running.')
            return
        self.check_saved_variables()

        self.vissim.Evaluation.SetAttValue('KeepPrevResults', 'KEEPALL')
        if not simulation_period:  # None, zero or empty string
            self.set_simulation_period(
                self.evaluation_periods[self.network_file])
        else:
            self.set_simulation_period(simulation_period)
        self.set_number_of_runs(runs_per_input)

        print('Starting series of runs, with initial seed {} and increment {}'.
              format(self.vissim.Simulation.AttValue('RandSeed'),
                     self.vissim.Simulation.AttValue('RandSeedIncr')))

        self.vissim.SuspendUpdateGUI()
        n_inputs = 0
        start_time = time.perf_counter()
        # 'highway_in_and_out_lanes', 'I710-MultiSec-3mi', 'US_101'
        for input_per_lane in range(initial_input_per_lane,
                                    max_input_per_lane + 1,
                                    input_increase_per_lane):
            self.set_uniform_vehicle_input_for_all_lanes(input_per_lane)

            if self.network_file == 'highway_in_and_out_lanes':
                self.run_toy_scenario()
            elif self.network_file == 'I710-MultiSec-3mi':
                self.run_i710_simulation(scenario_idx=2)
            elif self.network_file == 'US_101':
                self.run_us_101_simulation()
            else:
                print("Error: trying to evaluate unknown scenario")
            n_inputs += 1

        end_time = time.perf_counter()
        print('Total time: {}s to run {} simulations including {} different '
              'vehicle inputs'.format(end_time - start_time,
                                      n_inputs * runs_per_input, n_inputs))
        self.vissim.ResumeUpdateGUI()

    def run_with_increasing_autonomous_penetration(
            self, autonomous_percentage_increase,
            initial_autonomous_percentage,
            final_autonomous_percentage,
            input_increase_per_lane=500,
            initial_input_per_lane=500,
            max_input_per_lane=2500,
            runs_per_input=10,
            is_debugging=False):
        """Runs scenarios with increasing demand for varying penetration
        values of autonomous vehicles"""

        # if is_debugging and (final_autonomous_percentage -
        #         initial_autonomous_percentage) > 20:
        #     print

        self.set_evaluation_outputs(True, True, True, True, 60, 30)
        # Any values for initial random seed and increment are good, as long
        # as they are the same over each set of simulations.
        self.set_random_seed(7)
        self.set_random_seed_increment(1)
        results_base_folder = os.path.join(self.networks_folder,
                                           self.network_file)

        for autonomous_percentage in range(initial_autonomous_percentage,
                                           final_autonomous_percentage + 1,
                                           autonomous_percentage_increase):
            if not is_debugging:  # save all results in different folders
                # For each autonomous percentage, we reset VISSIM's simulation
                # count
                self.reset_saved_simulations(warning_active=False)
                # Then we set the proper folder to save the results
                results_folder = os.path.join(
                    results_base_folder,
                    str(autonomous_percentage) + '_percent_autonomous')
                self.vissim.Evaluation.SetAttValue('EvalOutDir', results_folder)
            else:
                self.use_debug_folder_for_results()
            # Finally, we set the percentage and run the simulation
            self.set_autonomous_percentage(autonomous_percentage)
            self.run_with_increasing_demand(
                input_increase_per_lane=input_increase_per_lane,
                initial_input_per_lane=initial_input_per_lane,
                max_input_per_lane=max_input_per_lane,
                runs_per_input=runs_per_input)

    def run_us_101_with_different_speed_limits(self, possible_speeds=None):
        if (self.vissim.AttValue('InputFile')
                != (self.existing_networks['us101'] + self.vissim_net_ext)):
            print('You must load the us101 scenario before running it')
            return

        n_lanes = 5
        if possible_speeds is None:
            possible_speeds = [25, 35, 40]

        self.vissim.SuspendUpdateGUI()
        for i in range(len(possible_speeds) - 1):
            speed_limit_per_lane = [possible_speeds[0]] * n_lanes
            for j in range(n_lanes):
                speed_limit_per_lane[:j] = [possible_speeds[i + 1]] * j
                print('Speed reduction values: ', speed_limit_per_lane)
                self.run_us_101_simulation(
                    speed_limit_per_lane=speed_limit_per_lane)

        self.vissim.ResumeUpdateGUI()

    # MODIFYING SCENARIO ------------------------------------------------------#
    def set_evaluation_outputs(self,
                               save_vehicle_record: bool = False,
                               save_ssam_file: bool = False,
                               activate_data_collections: bool = False,
                               activate_link_evaluation: bool = False,
                               warm_up_time: int = 0,
                               data_frequency: int = 30):
        """ Sets evaluation output options for various possible VISSIM outputs

        :param save_vehicle_record: Defines if VISSIM should save a vehicle
         record file
        :param save_ssam_file: Defines if VISSIM should save the file to use
         with the SSAM software
        :param activate_data_collections: Tells VISSIM whether to activate
         measurements from data collection points. Note that the autosave
         option must be manually selected in VISSIM
        :param activate_link_evaluation: Tells VISSIM whether to activate
         measurements in links. Note that the auto save option must be
         manually selected in VISSIM
        :param warm_up_time: simulation second in which the vehicle records,
        data collections and link evaluation start
        :param data_frequency: Duration of the evaluation intervals in which
         the data collections and link evaluation results are aggregated"""

        if not self.vissim.AttValue('InputFile'):
            print('Cannot change output options because no simulation is '
                  'open')
            return

        evaluation = self.vissim.Evaluation
        evaluation.SetAttValue('VehRecWriteFile', save_vehicle_record)
        evaluation.SetAttValue('VehRecFromTime', warm_up_time)
        evaluation.SetAttValue('SSAMWriteFile', save_ssam_file)
        evaluation.SetAttValue('DataCollCollectData',
                               activate_data_collections)
        evaluation.SetAttValue('DataCollFromTime', warm_up_time)
        evaluation.SetAttValue('DataCollInterval', data_frequency)
        evaluation.SetAttValue('LinkResCollectData',
                               activate_link_evaluation)
        evaluation.SetAttValue('LinkResFromTime', warm_up_time)
        evaluation.SetAttValue('LinkResInterval', data_frequency)

    def set_simulation_parameters(self, sim_params: dict):
        """
        Sets parameters accessible through the Simulation member of a Vissim
        instance.

        :param sim_params: dictionary with {param_name: param_value}. Check
         VISSIM COM docs for possible parameters
        :return: None
        """
        for param_name, param_value in sim_params.items():
            print("Client: setting parameter {} to value {}".
                  format(param_name, param_value))
            try:
                self.vissim.Simulation.SetAttValue(param_name, param_value)
            except AttributeError as err:
                self.close_vissim()
                print("Failed to set parameter")
                print("err=", err)

    def set_simulation_period(self, period):
        """Sets the period of the simulation"""
        sim_params = {'SimPeriod': period}
        self.set_simulation_parameters(sim_params)

    def set_random_seed(self, seed):
        """Sets the simulation random seed"""
        sim_params = {'RandSeed': seed}
        self.set_simulation_parameters(sim_params)

    def set_random_seed_increment(self, seed_increment):
        """Sets the random seed increment when running several simulations"""
        sim_params = {'RandSeedIncr': seed_increment}
        self.set_simulation_parameters(sim_params)

    def set_number_of_runs(self, number_of_runs):
        """Sets the total number of runs performed in a row"""
        sim_params = {'NumRuns': number_of_runs}
        self.set_simulation_parameters(sim_params)

    def create_time_intervals(self, period):
        """Programmatically creates time intervals in VISSIM so that
        measurements can be taken periodically. This avoids having to stop
        the simulation to save the results as they come.
        It seems like VISSIM already provides an easy way to do this, so this
        function is no longer necessary

        :param period: time intervals will be created with equal 'period'
         duration
         """

        print('WARNING: this function does''t do anything for now')

        n_intervals = self.vissim.Simulation.AttValue(
            'SimPeriod') // period + 1

        # I still don't know how to find the TimeIntervalSet which contains
        # the TimeIntervalContainer relative to the measurements
        time_interval_set_container = self.vissim.Net.TimeIntervalSets
        for ti_set in time_interval_set_container:
            ti_container = ti_set.TimeInts
            if True:  # create condition to find the proper TimeIntervalSet
                continue
            # print('initial number of time intervals: ', len(ti_container))
            # for i in range(len(ti_container), n_intervals ):
            #     ti_container.AddTimeInterval(i)
            # print('final number of time intervals: ', len(ti_container))

        # Save
        # self.vissim.SaveNet()

    def set_vehicle_inputs_by_name(self, new_veh_inputs: dict):
        """
        Sets the several vehicle inputs in the simulation by name.

        :param new_veh_inputs: vehicle inputs (veh/h) identified following
         {veh input name: input value}
        """

        veh_input_container = self.vissim.Net.VehicleInputs.GetAll()
        for veh_input in veh_input_container:
            vi_name = veh_input.AttValue('Name')
            if vi_name in new_veh_inputs:
                veh_input.SetAttValue('Volume(1)', new_veh_inputs[vi_name])
                print('Client: Vehicle input {} set to {}'.
                      format(vi_name, veh_input.AttValue('Volume(1)')))
                new_veh_inputs.pop(vi_name)
            else:
                print('Client: Vehicle input {} left unchanged at {}'.
                      format(vi_name, veh_input.AttValue('Volume(1)')))

        for vi_key in new_veh_inputs:
            print('Client: Vehicle input {} was passed as parameter, but not '
                  'found in simulation'.format(vi_key))

    def set_uniform_vehicle_input_for_all_lanes(self, input_per_lane: int,
                                                time_interval=1):
        """ Looks at all vehicle inputs and sets each of them equal to
        input_per_lane * n_lanes

        :param input_per_lane: vehicles per hour per lane
        :param time_interval: time interval of the vehicle input that should be
         changed.
         """

        veh_input_container = self.vissim.Net.VehicleInputs
        for veh_input in veh_input_container:
            if time_interval > len(veh_input.TimeIntVehVols):
                print('Trying to change vehicle input at a time interval '
                      'number {} when vehicle input number {} only has {} '
                      'time intervals for now. Skipping the command'.
                      format(time_interval, veh_input.AttValue('No'),
                             len(veh_input.TimeIntVehVols)))

            n_lanes = veh_input.Link.AttValue('NumLanes')
            total_input = input_per_lane * n_lanes
            if time_interval > 1:
                # 'Continuous' of time intervals after the first must be
                # false otherwise they cannot be changed
                veh_input.SetAttValue('Cont(' + str(time_interval) + ')', 0)
            veh_input.SetAttValue('Volume(' + str(time_interval) + ')',
                                  total_input)
            print('Client: Vehicle input {} set to {}'.
                  format(veh_input.AttValue('Name'),
                         veh_input.AttValue('Volume(1)')))

    def set_vehicle_inputs_composition(self, composition_name: str):
        """Sets all the desired composition to all the vehicle inputs in the
        network. We prefer to work with the composition name (instead of its
        number) because its easier to keep the name constant over several
        networks.

        :param composition_name: The name of the composition in VISSIM."""

        # Find the composition number based on its name
        veh_compositions_container = self.vissim.Net.VehicleCompositions
        for veh_composition in veh_compositions_container:
            # Find the right vehicle composition object
            if veh_composition.AttValue('Name') == composition_name:
                vehicle_composition_number = veh_composition.AttValue('No')
                break
        else:  # no break
            raise ValueError('Client: Composition name not found in {} '
                             'network.'.format(self.network_file))

        # Set the desired composition to all vehicle inputs in the simulation
        veh_input_container = self.vissim.Net.VehicleInputs
        for veh_input in veh_input_container:
            veh_input.SetAttValue('VehComp(1)', vehicle_composition_number)
            print('Client: Composition of vehicle input {} set to {}'.
                  format(veh_input.AttValue('Name'), composition_name))

    def set_autonomous_percentage(self, autonomous_percentage: float,
                                  composition_name: str = 'with_autonomous'):
        """ Looks for the specified vehicle composition and sets the
        percentage of autonomous vehicles in it. The rest of the composition
        will be made of 'human' driven vehicles.
        Assumption worth noting: the vehicle composition in VISSIM already
        exists and it contains only two vehicle types: 100 (Car) and 110
        (Autonomous Car).

        :param autonomous_percentage: This should be expressed either as
         an integer between 0 and 100 or as a float in the range [0, 1) . If a
         percentage below 1 is desired, the user must pass it as a fraction
         (example 0.005).
        :param composition_name: the composition to be altered. In the
         current simulations it is called 'with_autonomous'
        """
        # Adjust from [0, 1) to [0, 100]
        if autonomous_percentage < 1:
            autonomous_percentage *= 100

        if autonomous_percentage <= 0:
            self.set_vehicle_inputs_composition('all_human')
        elif autonomous_percentage >= 100:
            self.set_vehicle_inputs_composition('all_autonomous')
        else:
            self.set_vehicle_inputs_composition('with_autonomous')
            desired_flows = {
                Vehicle.VISSIM_CAR_ID: 100 - autonomous_percentage,
                Vehicle.VISSIM_AUTONOMOUS_CAR_ID: autonomous_percentage}

            veh_compositions_container = self.vissim.Net.VehicleCompositions
            for veh_composition in veh_compositions_container:
                # Find the right vehicle composition object
                if veh_composition.AttValue('Name') == composition_name:
                    # Modify the relative flows
                    for relative_flow in veh_composition.VehCompRelFlows:
                        flow_vehicle_type = int(
                            relative_flow.AttValue('VehType'))
                        if flow_vehicle_type in desired_flows:
                            relative_flow.SetAttValue(
                                'RelFlow', desired_flows[flow_vehicle_type])
            print('Client: input flows are {}% autonomous.'.
                  format(autonomous_percentage))

    # HELPER FUNCTIONS --------------------------------------------------------#
    def is_some_network_loaded(self):
        if self.vissim.AttValue('InputFile') != '':
            # In case we loaded the simulation through VISSIM's interface:
            if self.network_file is None:
                self.network_file = (
                    self.vissim.AttValue('InputFile').split('.')[0])
                self.create_results_directory()
            return True
        else:
            return False

    def reset_saved_simulations(self, warning_active: bool = True):
        """Deletes the data from previous simulations in VISSIM's lists. If the
        directory where files are saved is not changed, previously saved
        files might be overwritten.

        :param warning_active: If True, asks the user for confirmation before
         deleting data."""

        # Double check we're not doing anything stupid

        if warning_active:
            print('You are trying to reset the current simulation count.\n',
                  'This might lead to previous results being overwritten.')
            user_decision = input('Press [y] to confirm and reset the '
                                  'count or [n] to keep previous results\n'
                                  '(the program will continue running in any '
                                  'case).')
            if user_decision == 'y':
                print('You chose to RESET the current simulation count.')
            else:
                print('You chose to KEEP the current simulation count.')
                return

        # VISSIM does not provide a direct method to erase list results and
        # reset simulation number. The workaround is to change the evaluation
        # configuration to not keep results of previous simulations and then
        # run a single simulation step.
        print('Client: Resetting simulation count...')
        result_folder = self.vissim.Evaluation.AttValue('EvalOutDir')
        self.use_debug_folder_for_results()
        self.vissim.Evaluation.SetAttValue('KeepPrevResults', 'KEEPNONE')
        self.vissim.Simulation.RunSingleStep()
        self.vissim.Simulation.Stop()
        self.vissim.Evaluation.SetAttValue('KeepPrevResults', 'KEEPALL')
        self.vissim.Evaluation.SetAttValue('EvalOutDir', result_folder)

    def use_debug_folder_for_results(self):
        debug_log_folder = os.path.join(self.networks_folder, self.network_file,
                                        'test')
        self.vissim.Evaluation.SetAttValue('EvalOutDir', debug_log_folder)

    def get_max_decel_data(self):
        """Checks the vehicle types used in simulation, creates a Panda
        dataframe with max deceleration values by vehicle type and speed, and
        saves the dataframe as csv

        :return: None
        """

        if not self.is_some_network_loaded():
            print("No network loaded")
            if self.network_file is None:
                print("No network file defined to load either")
                return
            print("Loading network: ", self.network_file)
            self.load_simulation(self.network_file)

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

    def check_saved_variables(self):
        """If the simulation is set to export vehicle records, the method
        checks whether all the necessary vehicle variables are set to be
        saved. Returns True otherwise.

        :return: boolean indicating whether to continue run"""

        if not self.vissim.Evaluation.AttValue('VehRecWriteFile'):
            return True

        needed_variables = {'SIMSEC', 'NO', 'VEHTYPE', 'LANE\\LINK\\NO',
                            'LANE\\INDEX', 'LEADTARGNO',
                            'SPEED', 'SPEEDDIFF',
                            'COORDFRONTX', 'COORDFRONTY',
                            'COORDREARX', 'COORDREARY'}
        att_selection_container = self.vissim.Evaluation.VehRecAttributes
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
                print('You chose not to go forward with the simulation.')
                return False
            else:
                print('You chose to go forward with the simulation.')
                return True
        else:
            print('All necessary variables set to be saved.')
            return True

    def create_results_directory(self):
        if self.network_file:
            results_folder = os.path.join(self.networks_folder,
                                          self.network_file)
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)

    def close_vissim(self):
        self.vissim = None
