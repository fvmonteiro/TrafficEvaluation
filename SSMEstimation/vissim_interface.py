import os
import time
# from typing import Union
from typing import List, Dict
import warnings

import pandas as pd
# import pywintypes
import win32com.client as com

import data_writer
import file_handling
import readers
import vehicle
from vehicle import Vehicle, VehicleType


class VissimInterface:
    vissim_net_ext = '.inpx'
    vissim_layout_ext = '.layx'

    _evaluation_periods = {'in_and_out': 1800,
                           'in_and_merge': 1800,
                           'platoon_lane_change': 1800,
                           'i710': 3600,
                           'us101': 1800,
                           'traffic_lights': 1800}

    _initial_random_seed = 7

    def __init__(self):
        self.networks_folder = file_handling.get_networks_folder()
        self.network_name = None
        # self.layout_file = None
        self.network_relative_address = None
        self.vissim = None
        self.open_vissim()

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
         network nickname. Currently available: in_and_out, in_and_merge, i710,
         us101, traffic_lights
        :param layout_file: Optionally defines the layout file for the network
        :return: boolean indicating if simulation was properly loaded
        """

        self.network_name = network_name
        self.network_relative_address = (
            file_handling.get_relative_address_from_network_name(network_name))
        net_full_path = os.path.join(self.networks_folder,
                                     self.network_relative_address
                                     + self.vissim_net_ext)

        if os.path.isfile(net_full_path):
            print("Client: Loading file")
            self.vissim.LoadNet(net_full_path)
            if layout_file is not None:
                layout_full_path = os.path.join(self.networks_folder,
                                                layout_file
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

        self.create_network_results_directory()
        return True

    # RUNNING NETWORKS --------------------------------------------------------#

    def run_simple_scenario(self, scenario: str,
                            in_flow_input: int = None,
                            main_flow_input: int = None):
        """
        Runs the one of the toy scenarios: highway_in_and_out_lanes or
        highway_in_and_merge. Vehicle results are automatically saved.

        :param scenario: in_and_out or in_and_merge
        :param in_flow_input: vehicle input (veh/h) of the in ramp (optional)
        :param main_flow_input: vehicle input (veh/h) of the main road
         (optional)
        :return: None
        """

        if not self.is_network_loaded(scenario):
            return

        veh_volumes = dict()
        if in_flow_input is not None:
            veh_volumes['in_flow'] = in_flow_input
        if main_flow_input is not None:
            veh_volumes['main_flow'] = main_flow_input
        if len(veh_volumes) > 0:
            self.set_vehicle_inputs(veh_volumes)

        self.vissim.Evaluation.SetAttValue('VehRecFromTime', 0)
        # when simulation gets longer, ignore warm-up time
        # Run
        print('Client: Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Client: Simulation done.')

    def run_in_and_out_scenario(self, in_flow_input: int = None,
                                main_flow_input: int = None):
        """
        Runs the highway_in_and_out_lanes VISSIM scenario. Vehicle results are
        automatically saved.

        :param in_flow_input: vehicle input (veh/h) of the in ramp (optional)
        :param main_flow_input: vehicle input (veh/h) of the main road
         (optional)
        :return: None
        """
        self.run_simple_scenario('in_and_out', in_flow_input, main_flow_input)

    def run_in_and_merge_scenario(self, in_flow_input: int = None,
                                  main_flow_input: int = None):
        """
        Runs the highway_in_and_merge VISSIM scenario. Vehicle results are
        automatically saved.

        :param in_flow_input: vehicle input (veh/h) of the in ramp (optional)
        :param main_flow_input: vehicle input (veh/h) of the main road
         (optional)
        :return: None
        """
        self.run_simple_scenario('in_and_merge', in_flow_input, main_flow_input)

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

        if not self.is_network_loaded('us101'):
            return

        veh_inputs = {'highway_input': highway_input,
                      'ramp_input': ramp_input}
        if speed_limit_per_lane is None:
            speed_limit_per_lane = [40, 30, 25, 25, 25]

        self.set_vehicle_inputs(veh_inputs)

        reduced_speed_areas = self.vissim.Net.ReducedSpeedAreas
        for i in range(len(reduced_speed_areas)):
            reduced_speed_areas[i].SetAttValue(
                'DesSpeedDistr(10)', speed_limit_per_lane[i])

        print('Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Simulation done.')

    def run_i710_simulation(self, scenario_idx: int, demand=None):
        """
        Run PTV VISSIM simulation of the I-710 with possible accident.

        :param scenario_idx: index of simulation scenarios. Follows:
         0: No block, 1: All time block, 2: Temporary block
        :param demand: vehicle input (veh/hr)
        """

        # TODO: we can maybe find a way of running this through Vissim too,
        #  so it would be easier to test this scenario. We need to Run Script
        #  File or Event Based from the Vissim interface for that

        if not self.is_network_loaded('i710'):
            return

        # Definition of scenarios
        scenarios = [{'link': 5, 'upstream_link': 10010, 'lane': 2,
                      'coordinate': 10, 'incident_start_time': 30,
                      'incident_end_time': 30},
                     {'link': 5, 'upstream_link': 10010, 'lane': 2,
                      'coordinate': 10, 'incident_start_time': 30,
                      'incident_end_time': 10000},
                     {'link': 5, 'upstream_link': 10010, 'lane': 2,
                      'coordinate': 10, 'incident_start_time': 780,
                      'incident_end_time': 1980}]

        # Incident parameters
        incident_start_time = scenarios[scenario_idx]['incident_start_time']
        incident_end_time = scenarios[scenario_idx]['incident_end_time']
        incident_link = scenarios[scenario_idx]['link']
        incident_upstream_link = scenarios[scenario_idx]['upstream_link']
        incident_lane = scenarios[scenario_idx]['lane']

        if demand is not None:
            print('NOT YET TESTED')
            veh_inputs = {'main': demand}
            self.set_vehicle_inputs(veh_inputs)

        # Get link and vehicle objects
        net = self.vissim.Net
        links = net.Links
        vehicles = net.Vehicles

        # Get total simulation time and number of runs
        simulation = self.vissim.Simulation
        sim_time = simulation.AttValue('SimPeriod')
        n_runs = simulation.AttValue('NumRuns')

        # Make sure that all lanes are open
        for link in links:
            # Open all lanes
            for lane in link.Lanes:
                lane.SetAttValue('BlockedVehClasses', '')

        # Start simulation
        print("Scenario:", scenario_idx)
        print("Random Seed:", simulation.AttValue('RandSeed'))
        print("Client: Starting simulation")
        # Set break point for first run and run until accident
        simulation.SetAttValue("SimBreakAt", incident_start_time)
        simulation.RunContinuous()
        run_counter = 0
        while run_counter < n_runs:
            # Create incident (simulation stops automatically at the
            # SimBreakAt value)
            current_time = simulation.AttValue('SimSec')
            bus_no = 2  # No. of buses used to block the lane
            bus_array = []
            if current_time >= incident_start_time:
                print('Client: Creating incident')
                for i in range(bus_no):
                    bus_array.append(vehicles.AddVehicleAtLinkPosition(
                        300, incident_link, incident_lane,
                        scenarios[scenario_idx]['coordinate'] + 20 * i, 0, 0))
                # Tell vehicles to change lanes by blocking the accident lane
                for lane in links.ItemByKey(incident_upstream_link).Lanes:
                    if lane.AttValue('Index') == incident_lane:
                        lane.SetAttValue('BlockedVehClasses', '10')

                # Set the accident end break point or set the next accident
                # start break point
                if sim_time >= incident_end_time:
                    simulation.SetAttValue("SimBreakAt", incident_end_time)
                else:
                    simulation.SetAttValue("SimBreakAt", incident_start_time)
                print('Client: Running again')
                simulation.RunContinuous()

            # Make sure that all lanes are open after the incident (either in
            # the same run or in the next run)
            for link in links:
                for lane in link.Lanes:
                    lane.SetAttValue('BlockedVehClasses', '')

            # Remove the incident
            current_time = simulation.AttValue('SimSec')
            if current_time >= incident_end_time:
                print('Client: Removing incident')
                for veh in bus_array:
                    vehicles.RemoveVehicle(veh.AttValue('No'))
                # open all closed lanes
                for link in links:
                    for lane in link.Lanes:
                        lane.SetAttValue('BlockedVehClasses', '')
                # Set accident start break point for next the simulation run
                simulation.SetAttValue("SimBreakAt", incident_start_time)
                print('Client: Running again')
                simulation.RunContinuous()

            run_counter += 1
            print('runs finished: ', run_counter)

        print('Simulation done')

    def run_traffic_lights_scenario(self, vehicle_input=None):
        """
        Run PTV VISSIM simulation traffic_lights_study.

        :param vehicle_input: Total vehicles per hour entering the scenario
        :return: None
        """
        if not self.is_network_loaded('traffic_lights'):
            return
        if vehicle_input is not None:
            self.set_vehicle_inputs({'main': vehicle_input})
        # Run
        print('Client: Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('Client: Simulation done.')

    def run_platoon_scenario(self, vehicle_input=None):
        """

        :param vehicle_input:
        :return:
        """
        if not self.is_network_loaded('platoon_lane_change'):
            return

        simulation = self.vissim.Simulation
        self.set_simulation_period(60)
        self.set_number_of_runs(2)
        sim_time = simulation.AttValue('SimPeriod')
        n_runs = simulation.AttValue('NumRuns')

        platoon_size = 3  # number of vehicles
        first_platoon_time = 10
        platoon_creation_period = 10

        run_counter = 0
        simulation.SetAttValue("SimBreakAt", first_platoon_time)
        simulation.RunContinuous()
        while run_counter < n_runs:
            platoon_counter = 0
            platoon_creation_time = first_platoon_time
            print('Client: simulation', run_counter + 1, 'started')
            while platoon_creation_time < sim_time:
                platoon_counter += 1
                platoon_creation_time += platoon_creation_period
                print('Client: Creating platoon ', platoon_counter)
                self.create_platoon(platoon_size)
                if sim_time > platoon_creation_time:
                    simulation.SetAttValue("SimBreakAt", platoon_creation_time)
                else:
                    simulation.SetAttValue("SimBreakAt", first_platoon_time)
                print('Client: Running again')
                simulation.RunContinuous()
            run_counter += 1
            print('runs finished:', run_counter)
        print('Simulation done')

    def test_lane_access(self):
        """Code to figure out how to read lane by lane information"""

        if not self.load_simulation('i710'):
            return
        self.set_evaluation_outputs()  # all false by default

        simulation_time_sec = 150
        current_time = 0
        t_data_sec = 30

        # This line is different from Yihang's code, because his code doesn't
        # use the GetAll method. You can select only the links that interest you
        # instead of GetAll
        links = self.vissim.Net.Links.GetAll()
        # Run each time step at a time
        print('Simulation starting.')
        simulation = self.vissim.Simulation
        while current_time <= simulation_time_sec:
            simulation.RunSingleStep()
            current_time = simulation.AttValue('SimSec')
            if current_time % t_data_sec == 0:
                for link in links:
                    for lane in link.Lanes.GetAll():
                        print('Link number: {}; Lane index: {}; Total vehs: {}'.
                              format(link.AttValue('No'),
                                     lane.AttValue('Index'),
                                     lane.Vehs.Count))
        print('Simulation done.')

    # MULTIPLE SCENARIO RUN ---------------------------------------------------#

    def run_with_increasing_demand(self, inputs_per_lane: List[int],
                                   runs_per_input: int = 10,
                                   simulation_period: int = None):
        """ Runs a simulation several times with different seeds, then
        increases the input and reruns the simulation.

        :param inputs_per_lane: vehicles per hour per lane
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
                self._evaluation_periods[self.network_name])
        else:
            self.set_simulation_period(simulation_period)
        self.set_number_of_runs(runs_per_input)

        simulation = self.vissim.Simulation
        print('Starting series of runs, with initial seed {} and increment {}'.
              format(simulation.AttValue('RandSeed'),
                     simulation.AttValue('RandSeedIncr')))

        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
        self.vissim.SuspendUpdateGUI()
        n_inputs = 0
        start_time = time.perf_counter()
        for ipl in inputs_per_lane:
            self.set_uniform_vehicle_input_for_all_lanes(ipl)

            # For each input, we reset VISSIM's simulation count
            self.reset_saved_simulations(warning_active=False)
            # Then we set the proper folder to save the results
            current_folder = self.vissim.Evaluation.AttValue('EvalOutDir')
            head, tail = os.path.split(current_folder)
            if 'percent' in tail or 'test' in tail:
                head = os.path.join(head, tail)
            veh_input_folder = str(ipl) + '_vehs_per_lane'
            results_folder = os.path.join(head, veh_input_folder)
            self.set_results_folder(results_folder)

            if self.network_name == 'in_and_out':
                self.run_in_and_out_scenario()
            elif self.network_name == 'in_and_merge':
                self.run_in_and_merge_scenario()
            elif self.network_name == 'i710':
                self.run_i710_simulation(scenario_idx=1)
            elif self.network_name == 'us101':
                self.run_us_101_simulation()
            elif self.network_name == 'traffic_lights':
                self.run_traffic_lights_scenario()
            else:
                print("Error: trying to evaluate unknown scenario")
            n_inputs += 1

        end_time = time.perf_counter()
        print('Total time: {}s to run {} simulations including {} different '
              'vehicle inputs'.format(end_time - start_time,
                                      n_inputs * runs_per_input, n_inputs))
        self.vissim.ResumeUpdateGUI()
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 0)

    def run_with_varying_controlled_percentage(
            self,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            inputs_per_lane: List[int],
            runs_per_input: int = 10,
            simulation_period: int = None,
            is_debugging: bool = False):
        """Runs scenarios with increasing demand for varying percentage
        values of controlled vehicles

        :param percentages_per_vehicle_types: List of dictionaries. Each
         dictionary should define the percentages of controlled vehicles as
         VehicleType: int.
        :param inputs_per_lane: vehicles per hour per lane
        :param runs_per_input: how many runs with the same input and varying
         seed
        :param simulation_period: Parameter used for debugging, when we want
        shorter runs. If left at None, the code uses the predefined period
        chosen for longer evaluations.
        :param is_debugging: if true, results are saved to a test folder
        """

        if (self.network_name in {'in_and_out', 'in_and_merge', 'us101'}
                or is_debugging):
            warm_up_minutes = 1
            save_ssam = True
        elif self.network_name in {'i710', 'traffic_lights'}:
            warm_up_minutes = 10
            save_ssam = False
        else:
            raise ValueError('Warm up time not set for network ',
                             self.network_name)
        self.set_evaluation_outputs(True, save_ssam, True, True, True,
                                    warm_up_minutes * 60, 30)
        # Any values for initial random seed and increment are good, as long
        # as they are the same over each set of simulations.
        self.set_random_seed(self._initial_random_seed)
        self.set_random_seed_increment(1)
        results_base_folder = os.path.join(self.networks_folder,
                                           self.network_relative_address)

        for item in percentages_per_vehicle_types:
            vehicle_types = list(item.keys())
            percentages = list(item.values())
            if is_debugging:  # run 2 short simulations
                print('Debug mode: running 2 short simulations with single '
                      'input per lane value.')
                self.use_debug_folder_for_results()
                self.reset_saved_simulations(warning_active=False)
                input_increase_per_lane = 100  # any value > 1
                initial_input_per_lane = 1000
                max_input_per_lane = initial_input_per_lane
                runs_per_input = 2
                simulation_period = 360
            else:  # save all results in different folders
                # For each percentage, we reset VISSIM's simulation count
                self.reset_saved_simulations(warning_active=False)
                # Then we set the proper folder to save the results
                results_folder = os.path.join(
                    results_base_folder,
                    file_handling.create_percent_folder_name(
                        percentages, vehicle_types))
                self.set_results_folder(results_folder)
            # Finally, we set the percentage and run the simulation
            self.set_controlled_vehicles_percentage(percentages,
                                                    list(vehicle_types))
            self.run_with_increasing_demand(
                inputs_per_lane,
                runs_per_input=runs_per_input,
                simulation_period=simulation_period)
            # TODO: we can save DataCollectionResults from here. Get the
            #  container from INet.DataCollectionMeasurements. It looks like
            #  we'll have to read one time interval and one simulation result
            #  at a time.

    def run_us_101_with_different_speed_limits(self, possible_speeds=None):
        if not self.is_network_loaded('us101'):
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
                               save_lane_changes: bool = False,
                               warm_up_time: int = 0,
                               data_frequency: int = 30):
        """
        Sets evaluation output options for various possible VISSIM outputs.
        If no arguments are defined, assumes all are false.

        :param save_vehicle_record: Defines if VISSIM saves the vehicle
         record file
        :param save_ssam_file: Defines if VISSIM saves the file to use
         with the SSAM software
        :param activate_data_collections: Tells VISSIM whether to activate
         measurements from data collection points. Note that the auto save
         option must be manually selected in VISSIM
        :param activate_link_evaluation: Tells VISSIM whether to activate
         measurements in links. Note that the auto save option must be
         manually selected in VISSIM
        :param save_lane_changes: Defines if VISSIM saves lane changing data
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
        evaluation.SetAttValue('LaneChangesWriteFile', save_lane_changes)
        evaluation.SetAttValue('LaneChangesFromTime', warm_up_time)

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

    def set_results_folder(self, results_folder):
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        self.vissim.Evaluation.SetAttValue('EvalOutDir', results_folder)

    # def create_time_intervals(self, period):
    #     """Programmatically creates time intervals in VISSIM so that
    #     measurements can be taken periodically. This avoids having to stop
    #     the simulation to save the results as they come.
    #     It seems like VISSIM already provides an easy way to do this, so this
    #     function is no longer necessary
    #
    #     :param period: time intervals will be created with equal 'period'
    #      duration
    #      """
    #
    #     print('WARNING: this function does''t do anything for now')
    #
    #     # n_intervals = self.vissim.Simulation.AttValue(
    #     #     'SimPeriod') // period + 1
    #     #
    #     # # I still don't know how to find the TimeIntervalSet which contains
    #     # # the TimeIntervalContainer relative to the measurements
    #     # time_interval_set_container = self.vissim.Net.TimeIntervalSets
    #     # for ti_set in time_interval_set_container:
    #     #     ti_container = ti_set.TimeInts
    #     #     if True:  # create condition to find the proper TimeIntervalSet
    #     #         print('initial number of time intervals: ',
    #     #             len(ti_container))
    #     #         for i in range(len(ti_container), n_intervals):
    #     #             ti_container.AddTimeInterval(i)
    #     #         print('final number of time intervals: ', len(ti_container))
    #     #
    #     # # Save
    #     # self.vissim.SaveNet()

    def set_vehicle_inputs(self, new_veh_inputs: dict):
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

    def set_vehicle_inputs_composition(self, composition_number: int):
        """Sets all the desired composition to all the vehicle inputs in the
        network. We prefer to work with the composition name (instead of its
        number) because its easier to keep the name constant over several
        networks.

        :param composition_number: The number of the composition in VISSIM.
        """

        veh_compositions_container = self.vissim.Net.VehicleCompositions
        veh_input_container = self.vissim.Net.VehicleInputs
        for veh_input in veh_input_container:
            veh_input.SetAttValue('VehComp(1)', composition_number)
            print('Client: Composition of vehicle input {} set to {}'.
                  format(veh_input.AttValue('Name'),
                         veh_compositions_container.ItemByKey(
                             composition_number).AttValue('Name')))
        # return composition_number

    def set_controlled_vehicles_percentage(
            self, percentages: List[int], vehicle_types: List[VehicleType]):
        """
        Looks for the specified vehicle composition and sets the
        percentage of autonomous vehicles in it. The rest of the composition
        will be made of 'human' driven vehicles.
        Assumption worth noting: the vehicle composition in VISSIM already
        exists and it contains two vehicle types: regular car and the
        controlled vehicle type

        :param percentages: This should be expressed either as
         an integer between 0 and 100.
        :param vehicle_types: enum to indicate the vehicle (controller) type
        """
        # The percentage of non-controlled vehicles must be human
        total_controlled_percentage = sum(percentages)
        if total_controlled_percentage == 0:
            vehicle_types = [VehicleType.HUMAN_DRIVEN]
            percentages = [100]
        elif total_controlled_percentage < 100:
            vehicle_types.append(VehicleType.HUMAN_DRIVEN)
            percentages.append(100 - total_controlled_percentage)
        composition_number = self.find_matching_vehicle_composition(
            vehicle_types)
        self.set_vehicle_inputs_composition(composition_number)

        # Modify the relative flows
        desired_flows = dict()
        for i in range(len(vehicle_types)):
            vehicle_type_id = Vehicle.ENUM_TO_VISSIM_ID[vehicle_types[i]]
            desired_flows[vehicle_type_id] = percentages[i]
        veh_composition = self.vissim.Net.VehicleCompositions.ItemByKey(
            composition_number)
        for relative_flow in veh_composition.VehCompRelFlows:
            flow_vehicle_type = int(relative_flow.AttValue('VehType'))
            if (flow_vehicle_type in desired_flows
                    and desired_flows[flow_vehicle_type] > 0):
                relative_flow.SetAttValue('RelFlow',
                                          desired_flows[flow_vehicle_type])
                print('Client: veh type {} at {}%.'.
                      format(flow_vehicle_type,
                             relative_flow.AttValue('RelFlow')))

    def set_traffic_lights(self):
        """
        Sets simulation traffic lights based on information on a csv file.
        The function modifies signal controller files (.sig) as well as signal
        controllers and signal heads in the VISSIM file.

        Assumes simulation has a single link, at least one signal controller
        file already created (.sig) and as many signal heads as rows in the
        csv file times the number of lanes.
        :return:
        """
        csv_reader = readers.TrafficLightSourceReader(self.network_name)
        source_data = csv_reader.load_data()
        signal_controller_container = self.vissim.Net.SignalControllers
        signal_head_container = self.vissim.Net.SignalHeads
        lane_container = self.vissim.Net.Links[0].Lanes

        if signal_controller_container.Count > source_data.shape[0]:
            print('FYI, more signal controllers in the simulation than in the '
                  'source file.')

        for _, row_unsure_type in source_data.iterrows():
            row = row_unsure_type.astype(int)
            sc_id = row['id']  # starts at 1

            self.set_signal_controller_times_in_file(sc_id, row['red duration'],
                                                     row['green duration'],
                                                     row['amber duration'])
            self.set_signal_controller(sc_id, signal_controller_container)
            self.set_signal_head(sc_id, lane_container, row['position'],
                                 signal_head_container)
        self.vissim.SaveNet()

    def set_signal_controller_times_in_file(
            self, sc_id: int, red_duration: int, green_duration: int,
            amber_duration: int) -> None:
        """
        Opens a signal controller file (.sig), edits the traffic light times,
        and saves the file.

        :param sc_id: Signal controller id
        :param red_duration: time traffic light stays red in seconds
        :param green_duration: time traffic light stays green in seconds
        :param amber_duration: time traffic light stays amber in seconds
        :return:
        """

        sc_reader = readers.SignalControllerFileReader(self.network_name)
        sc_editor = data_writer.SignalControllerTreeEditor()
        try:
            sc_file_tree = sc_reader.load_data(sc_id)
        except ValueError:
            if sc_id == 1:
                raise OSError("Network {} doesn't have any signal "
                              "controller files (sig) defined yet"
                              .format(self.network_name))
            # If there is no file yet, we get the first file as template,
            # modify it and save with a different name
            sc_file_tree = sc_reader.load_data(1)
            sc_editor.set_signal_controller_id(sc_file_tree, sc_id)

        sc_editor.set_times(sc_file_tree, red_duration,
                            green_duration, amber_duration)
        sc_editor.save_file(sc_file_tree, self.networks_folder,
                            self.network_relative_address + str(sc_id))

    def set_signal_controller(self, sc_id, signal_controller_container=None):
        """
        Sets the file named [simulation_name]sc_id.sig as the supply file for
        signal controller with id sc_id. Creates a new signal controller if one
        with sc_id doesn't exist, but this requires manually saving the file
        through VISSIM.

        :param sc_id: Signal controller id
        :param signal_controller_container: VISSIM object containing all the
         signal controllers
        :return:
        """
        if signal_controller_container is None:
            signal_controller_container = self.vissim.Net.SignalControllers

        ask_for_user_action = False
        if not signal_controller_container.ItemKeyExists(sc_id):
            ask_for_user_action = True
            signal_controller_container.AddSignalController(sc_id)
        signal_controller = signal_controller_container.ItemByKey(sc_id)
        signal_controller.SetAttValue('SupplyFile2', os.path.join(
            self.networks_folder,
            self.network_relative_address + str(sc_id) + '.sig'))
        if ask_for_user_action:
            input('You must manually open the signal controller window of '
                  'VISSIM and save the file.\nOtherwise, the code will '
                  'produce and error. Press any key when done.')

    def set_signal_head(self, sc_id: int, lane_container, position: int,
                        signal_head_container=None):
        """
        Sets as many signal heads as lanes in the lane_container to the given
        position and with the given signal controller. Assumes that each signal
        controller only has one signal group and that the network has a
        single link.

        :param sc_id: Signal controller id
        :param lane_container: VISSIM object containing all the lanes of some
         link
        :param position: longitudinal position of the signal head in the link
        :param signal_head_container: VISSIM object containing all the signal
         heads
        :return:
        """

        if signal_head_container is None:
            signal_head_container = self.vissim.Net.SignalHeads
        n_lanes = lane_container.Count
        for lane in lane_container:
            lane_idx = lane.AttValue('Index')
            signal_head_key = (sc_id - 1) * n_lanes + lane_idx
            if not signal_head_container.ItemKeyExists(signal_head_key):
                signal_head_container.AddSignalHead(signal_head_key,
                                                    lane,
                                                    position)
                signal_head = signal_head_container.ItemByKey(
                    signal_head_key)
            else:
                signal_head = signal_head_container.ItemByKey(
                    signal_head_key)
                signal_head.SetAttValue('Lane', '1-' + str(lane_idx))
                signal_head.SetAttValue('Pos', position)
            signal_head.SetAttValue('SG', str(sc_id) + '-1')

    def create_platoon(self, platoon_size):

        desired_speed = 100  # km/h
        vehicle_type = vehicle.VehicleType.CONNECTED
        platoon_vehicle = vehicle.Vehicle(vehicle_type)
        platoon_vehicle.free_flow_velocity = desired_speed / 3.6
        h, d = platoon_vehicle.compute_vehicle_following_parameters(
            leader_max_brake=platoon_vehicle.max_brake, rho=0.05)
        safe_gap = h * desired_speed / 3.6 + d

        vehicles = self.vissim.Net.Vehicles
        vissim_vehicle_type = vehicle.Vehicle.ENUM_TO_VISSIM_ID[vehicle_type]
        in_ramp_link = 2
        lane = 1
        position = 10
        interaction = True  # optional
        for i in range(platoon_size):
            added_vehicle = vehicles.AddVehicleAtLinkPosition(
                vissim_vehicle_type, in_ramp_link, lane, position,
                desired_speed, interaction)
            position += added_vehicle.AttValue('Length') + safe_gap

    # HELPER FUNCTIONS --------------------------------------------------------#

    def find_matching_vehicle_composition(
            self, vehicle_types: List[VehicleType]) -> int:
        """
        Finds the vehicle composition that has exactly the same vehicle types
        listed in the parameter

        :param vehicle_types: List of VehicleType enums
        :return: The vehicle composition number
        """
        vehicle_type_ids = set([Vehicle.ENUM_TO_VISSIM_ID[vt] for vt in
                                vehicle_types])
        veh_compositions_container = self.vissim.Net.VehicleCompositions
        for veh_composition in veh_compositions_container:
            counter = 0
            relative_flows_container = veh_composition.VehCompRelFlows
            # We can skip compositions with different number of relative flows
            if len(relative_flows_container) != len(vehicle_types):
                continue
            for relative_flow in relative_flows_container:
                flow_vehicle_type = int(relative_flow.AttValue('VehType'))
                if flow_vehicle_type not in vehicle_type_ids:
                    continue
                counter += 1
            if counter == len(vehicle_types):
                return veh_composition.AttValue('No')
        raise ValueError('Client: Composition not found in {} '
                         'network.'.format(self.network_name))

    def find_vehicle_composition_by_name(self, name) -> int:
        """
        Finds the vehicle composition name with the given name
        :param name: Vehicle composition name
        :return: Vehicle composition number
        """
        veh_compositions_container = self.vissim.Net.VehicleCompositions
        for veh_composition in veh_compositions_container:
            if veh_composition.AttValue('Name').lower() == name:
                return veh_composition.AttValue('No')
        raise ValueError('Client: Composition name not found in {} '
                         'network.'.format(self.network_name))

    def is_some_network_loaded(self):
        if self.vissim.AttValue('InputFile') != '':
            # In case we loaded the simulation through VISSIM's interface:
            if self.network_name is None:
                network_file = (
                    self.vissim.AttValue('InputFile').split('.')[0])
                self.network_name = (
                    file_handling.get_network_name_from_file_name(
                        network_file))
                # self.create_network_results_directory()
            return True
        else:
            return False

    def is_network_loaded(self, network_name: str):
        if (self.vissim.AttValue('InputFile')
                != (file_handling.get_file_name_from_network_name(network_name)
                    + self.vissim_net_ext)):
            print('You must load the ', network_name,
                  ' scenario before running it')
            return False
        return True

    def reset_saved_simulations(self, warning_active: bool = True):
        """
        Deletes the data from previous simulations in VISSIM's lists. If the
        directory where files are saved is not changed, previously saved
        files might be overwritten.

        :param warning_active: If True, asks the user for confirmation before
         deleting data.
        """

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

        print('Client: Resetting simulation count...')
        for simRun in self.vissim.Net.SimulationRuns:
            self.vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)
        # Old implementation
        # result_folder = self.vissim.Evaluation.AttValue('EvalOutDir')
        # self.use_debug_folder_for_results()
        # self.vissim.Evaluation.SetAttValue('KeepPrevResults', 'KEEPNONE')
        # self.vissim.Simulation.RunSingleStep()
        # self.vissim.Simulation.Stop()
        # self.vissim.Evaluation.SetAttValue('KeepPrevResults', 'KEEPALL')
        # self.set_results_folder(result_folder)

    def use_debug_folder_for_results(self):
        debug_log_folder = os.path.join(self.networks_folder,
                                        self.network_relative_address, 'test')
        self.set_results_folder(debug_log_folder)

    def get_max_decel_data(self):
        """Checks the vehicle types used in simulation, creates a Panda
        dataframe with max deceleration values by vehicle type and speed, and
        saves the dataframe as csv

        :return: None
        """

        if not self.is_some_network_loaded():
            print("No network loaded")
            if self.network_name is None:
                print("No network file defined to load either")
                return
            print("Loading network: ", self.network_name)
            self.load_simulation(self.network_name)

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
                            'LANE\\INDEX', 'POS', 'SPEED', 'POSLAT',
                            'SPEEDDIFF', 'SIMRUN',
                            'COORDFRONTX', 'COORDFRONTY',
                            'COORDREARX', 'COORDREARY',
                            'LENGTH',
                            'ACCELERATION', 'LNCHG'}
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

    def create_network_results_directory(self):
        if self.network_relative_address:
            results_folder = os.path.join(self.networks_folder,
                                          self.network_relative_address)
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)

    def close_vissim(self):
        self.vissim = None
