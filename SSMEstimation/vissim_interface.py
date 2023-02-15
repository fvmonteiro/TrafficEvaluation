from dataclasses import dataclass
from enum import Enum
import os
import time
from typing import List, Dict, Callable, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import pywintypes
import win32com.client as com

import data_writer
from file_handling import FileHandler
import readers
import vehicle
from vehicle import Vehicle, VehicleType, PlatoonLaneChangeStrategy


@dataclass
class _ScenarioInfo:
    evaluation_period: int
    warm_up_minutes: int
    run_function: Callable


class _UDANumber(Enum):
    risk_to_leaders = 9
    risk_to_follower = 10
    use_linear_lane_change_gap = 11
    platoon_lane_change_strategy = 13
    verbose_simulation = 98
    logged_vehicle = 99


class VissimInterface:
    vissim_net_ext = '.inpx'
    vissim_layout_ext = '.layx'

    # risk_to_leaders_uda_number = 9
    # risk_to_follower_uda_number = 10
    # use_linear_lane_change_gap_uda_number = 11
    # platoon_lane_change_strategy_uda_number = 13
    # verbose_simulation_uda_number = 98
    # logged_vehicle_uda_number = 99

    _initial_random_seed = 7

    def __init__(self, vissim=None):
        self.file_handler = None
        self.network_info = None

        self._all_networks_info = {
            'in_and_out': _ScenarioInfo(1800, 1, self.run_in_and_out_scenario),
            'in_and_merge': _ScenarioInfo(1800, 1,
                                          self.run_in_and_merge_scenario),
            'platoon_mandatory_lane_change': _ScenarioInfo(1200, 1,
                                                 self.run_platoon_scenario),
            'platoon_discretionary_lane_change': _ScenarioInfo(
                1200, 1, self.run_platoon_scenario),
            'i710': _ScenarioInfo(3600, 10, self.run_i710_simulation),
            'us101': _ScenarioInfo(1800, 1, self.run_us_101_simulation),
            'traffic_lights': _ScenarioInfo(1800, 10,
                                            self.run_traffic_lights_scenario),
        }

        if vissim is None:
            self.vissim = None
            self.open_vissim()
        else:
            self.vissim = vissim

    def open_vissim(self):
        # Connect to the COM server, which opens a new Vissim window
        # vissim = com.gencache.EnsureDispatch("Vissim.Vissim")  # if
        # connecting for the first time ever
        vissim_id = "Vissim.Vissim"  # "VISSIM.Vissim.1000" # Useful if more
        # than one Vissim version installed
        print("[Client] Trying to create a Vissim instance")
        for i in range(5):
            try:
                self.vissim = com.Dispatch(vissim_id)
                print("[Client] Vissim instance created")
                return True
            except pywintypes.com_error:
                print("[Client] Failed attempt #" + str(i + 1))
        return False

    def load_simulation(self, scenario_name: str, layout_file: str = None):
        """ Loads a VISSIM network and optionally sets it to save vehicle
        records and ssam files.

        :param scenario_name: Currently available: in_and_out_*,
         in_and_merge, i710, us101, traffic_lights,
         platoon_mandatory_lane_change, platoon_discretionary_lane_change
        :param layout_file: Optionally defines the layout file for the network
        :return: boolean indicating if simulation was properly loaded
        """

        self.file_handler = FileHandler(scenario_name)
        self.network_info = self._all_networks_info[
            self.file_handler.get_network_name()]

        network_address = self.file_handler.get_network_file_folder()
        network_file = self.file_handler.get_file_name()
        net_full_path = os.path.join(network_address,
                                     network_file + self.vissim_net_ext)
        if os.path.isfile(net_full_path):
            print("[Client] Loading file")
            self.vissim.LoadNet(net_full_path)
            if layout_file is not None:
                layout_full_path = os.path.join(network_address,
                                                layout_file
                                                + self.vissim_layout_ext)
                if os.path.isfile(layout_full_path):
                    self.vissim.LoadLayout(layout_full_path)
                else:
                    print('[Client] Layout file {} not found.'.
                          format(net_full_path))
        else:
            print('[Client] File {} not found.'.
                  format(net_full_path))
            # sys.exit()  # better to use this?
            return False

        self.create_network_results_directory()
        return True

    def create_file_handler(self, scenario_name: str):
        """
        Creates a file handler instance, which is used determine where
        results are saved. The function is mostly useful during debugging when
        we want to avoid calling load_simulation several times.
        """
        if self.file_handler is not None:
            print('This object already has a file handler.')
            return
        self.file_handler = FileHandler(scenario_name)
        self.network_info = self._all_networks_info[
            self.file_handler.get_network_name()]

    # RUNNING NETWORKS --------------------------------------------------------#

    def run_simple_scenario(self,
                            in_flow_input: int = None,
                            main_flow_input: int = None):
        """
        Runs the one of the toy scenarios: highway_in_and_out_lanes or
        highway_in_and_merge. Vehicle results are automatically saved.

        :param in_flow_input: vehicle input (veh/h) of the in ramp (optional)
        :param main_flow_input: vehicle input (veh/h) of the main road
         (optional)
        :return: None
        """

        if not self.is_correct_network_loaded():
            return

        veh_volumes = dict()
        if in_flow_input is not None:
            veh_volumes['in_flow'] = in_flow_input
        if main_flow_input is not None:
            veh_volumes['main_flow'] = main_flow_input
        if len(veh_volumes) > 0:
            self.set_vehicle_inputs(veh_volumes)

        self.vissim.Evaluation.SetAttValue('VehRecFromTime', 0)
        # Run
        print('[Client] Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('[Client] Simulation done.')

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
        scenario_name = self.file_handler.scenario_name
        if scenario_name.endswith('risk_in_gap'):
            self.set_use_linear_lane_change_gap(False)
        else:
            self.set_use_linear_lane_change_gap(True)
        self.run_simple_scenario(in_flow_input, main_flow_input)

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
        self.run_simple_scenario(in_flow_input, main_flow_input)

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

        if not self.is_correct_network_loaded():
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

    def run_i710_simulation(self, scenario_idx: int = 1, demand=None):
        """
        Run PTV VISSIM simulation of the I-710 with possible accident.

        :param scenario_idx: index of simulation scenarios. Follows:
         0: No block, 1: All time block, 2: Temporary block
        :param demand: vehicle input (veh/hr)
        """

        if not self.is_correct_network_loaded():
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
        print("[Client] Starting simulation")
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
                print('[Client] Creating incident')
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
                print('[Client] Running again')
                simulation.RunContinuous()

            # Make sure that all lanes are open after the incident (either in
            # the same run or in the next run)
            for link in links:
                for lane in link.Lanes:
                    lane.SetAttValue('BlockedVehClasses', '')

            # Remove the incident
            current_time = simulation.AttValue('SimSec')
            if current_time >= incident_end_time:
                print('[Client] Removing incident')
                for veh in bus_array:
                    vehicles.RemoveVehicle(veh.AttValue('No'))
                # open all closed lanes
                for link in links:
                    for lane in link.Lanes:
                        lane.SetAttValue('BlockedVehClasses', '')
                # Set accident start break point for next the simulation run
                simulation.SetAttValue("SimBreakAt", incident_start_time)
                print('[Client] Running again')
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
        if not self.is_correct_network_loaded():
            return
        if vehicle_input is not None:
            self.set_vehicle_inputs({'main': vehicle_input})
        # Run
        print('[Client] Simulation starting.')
        self.vissim.Simulation.RunContinuous()
        print('[Client] Simulation done.')

    def run_platoon_scenario(self,
                             platoon_size: int,
                             platoon_speed: int,
                             first_platoon_time: int = 10,
                             platoon_creation_period: int = 60,
                             n_platoons: int = None,
                             is_platoon_autonomous: bool = True):
        """

        :param platoon_size: Number of vehicles in the platoon
        :param platoon_speed: Desired speed of platoon vehicles in km/h
        :param first_platoon_time: Time in seconds when platoons are created
         in the simulation
        :param platoon_creation_period: Interval in seconds between creation
         of platoons
        :param n_platoons: Total number of platoons to be created. If None,
        then creates platoons repeatedly until the end of the simulation
        :param is_platoon_autonomous: If true, platoon vehicles are
        autonomous. Otherwise, they are human driven.
        :return: Nothing
        """
        if not self.is_correct_network_loaded():
            return

        simulation = self.vissim.Simulation
        run_counter = 0
        sim_time = simulation.AttValue('SimPeriod')
        n_runs = simulation.AttValue('NumRuns')
        if n_platoons is None:
            n_platoons = sim_time // platoon_creation_period + 1
        # print('Sim period is ', sim_time)
        # print('n platoons is', n_platoons)
        # print('[Client] Setting first ever break point at ',
        #       first_platoon_time)
        simulation.SetAttValue("SimBreakAt", first_platoon_time)
        simulation.RunContinuous()
        while run_counter < n_runs:
            run_counter += 1
            platoon_counter = 0
            platoon_creation_time = first_platoon_time
            continue_loop_condition = (platoon_creation_time < sim_time
                                       and platoon_counter < n_platoons)
            print('[Client] Simulation', run_counter, 'started')
            while continue_loop_condition:
                platoon_counter += 1
                platoon_creation_time += platoon_creation_period
                continue_loop_condition = (platoon_creation_time < sim_time
                                           and platoon_counter < n_platoons)
                print('[Client] Creating platoon', platoon_counter,
                      'at time', simulation.SimulationSecond)
                self.create_platoon(platoon_size, platoon_speed)
                if continue_loop_condition:
                    # print('[Client] Setting break point at ',
                    #       platoon_creation_time)
                    simulation.SetAttValue("SimBreakAt", platoon_creation_time)
                else:
                    # print('[Client] Setting break point for next run at ',
                    #       first_platoon_time)
                    simulation.SetAttValue("SimBreakAt", first_platoon_time)
                # print('[Client] Running again')
                simulation.RunContinuous()
            # print('runs finished:', run_counter)

    def run_platoon_scenario_sample(
            self,  platoon_size: int,
            lane_change_strategy: PlatoonLaneChangeStrategy,
            vehicle_input: int, orig_and_dest_lane_speeds: Tuple[int, str],
            simulation_period=60, number_of_runs=2,
            first_platoon_time=10, platoon_creation_period=30,
            random_seed=None, is_fast_mode=False,
            is_simulation_verbose=False, logged_veh_id=None):
        """
        For initial test. Allows us to perform single runs of the platoon
        scenario with given parameters.
        :param vehicle_input: Vehicles per hour entering the main road
        :param lane_change_strategy: 0: No Strategy; 1: Synchronous;
         2: Leader First; 3: Last Vehicle First; 4: Leader First and Invert
        :return:
        """
        if not self.is_correct_network_loaded():
            return
        if random_seed is None:
            random_seed = self._initial_random_seed
        self.set_random_seed(random_seed)

        self.set_simulation_period(simulation_period)
        self.set_number_of_runs(number_of_runs)
        self.set_platoon_lane_change_strategy(lane_change_strategy)
        if vehicle_input is not None:
            veh_volumes = {'left_lane': vehicle_input,
                           'right_lane': vehicle_input}
            self.set_vehicle_inputs(veh_volumes)
        self.set_verbose_simulation(is_simulation_verbose)
        if logged_veh_id is not None:
            self.set_logged_vehicle_id(logged_veh_id)
        if is_fast_mode:
            self.vissim.Graphics.CurrentNetworkWindow.SetAttValue(
                "QuickMode", 1)

        platoon_speed = orig_and_dest_lane_speeds[0]

        self.run_platoon_scenario(
            platoon_size, platoon_speed,
            first_platoon_time=first_platoon_time,
            platoon_creation_period=platoon_creation_period,
            is_platoon_autonomous=True)
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue(
            "QuickMode", 0)

        print('Simulation done')

    def test_lane_access(self):
        """Code to figure out how to read lane by lane information"""

        if not self.load_simulation('i710'):
            return
        self.set_evaluation_options()  # all false by default

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

    def run_multiple_scenarios(
            self,
            vehicle_percentages: List[Dict[VehicleType, int]],
            inputs_per_lane: List[int],
            accepted_risks: List[int] = None,
            runs_per_scenario: int = 10,
            simulation_period: int = None,
            is_debugging: bool = False
    ):
        # Set-up simulation parameters
        if accepted_risks is None:
            accepted_risks = [0]
        if is_debugging:
            warm_up_minutes = 0
            runs_per_scenario = 1
            simulation_period = 360
        else:
            warm_up_minutes = self.network_info.warm_up_minutes
        if simulation_period is None:
            simulation_period = self.network_info.evaluation_period

        self.set_evaluation_options(True, False, True, True, True,
                                    warm_up_minutes * 60,
                                    data_frequency=5)
        self.set_random_seed(self._initial_random_seed)
        self.set_random_seed_increment(1)
        self.set_simulation_period(simulation_period)
        self.set_number_of_runs(runs_per_scenario)
        if not is_debugging:
            self.vissim.Graphics.CurrentNetworkWindow.SetAttValue(
                "QuickMode", 1)
            self.vissim.SuspendUpdateGUI()

        print("Starting multiple-scenario run.")
        for vp in vehicle_percentages:
            self.set_controlled_vehicles_percentage(vp)
            for ipl in inputs_per_lane:
                self.set_uniform_vehicle_input_for_all_lanes(ipl)
                # accepted risk does not impact simulations without any AVs
                relevant_risks = (accepted_risks if sum(vp.values()) > 0
                                  else [0])
                for ar in relevant_risks:
                    self.reset_saved_simulations(warning_active=False)
                    self.set_accepted_lane_change_risk_to_leaders(ar)
                    self.set_accepted_lane_change_risk_to_follower(ar / 2)
                    # self.file_handler.get_vissim_data_folder()

                    if is_debugging:
                        results_folder = (
                            self.file_handler.get_vissim_test_folder())
                        # self.use_debug_folder_for_results()
                    else:
                        results_folder = (
                            self.file_handler.get_vissim_data_folder(
                                vp, ipl, accepted_risk=ar))
                    is_folder_set = self.set_results_folder(results_folder)
                    print("Starting series of {} runs with duration {}".format(
                        runs_per_scenario, simulation_period
                    ))
                    start_time = time.perf_counter()
                    self.network_info.run_function()
                    end_time = time.perf_counter()
                    _print_run_time_with_unit(
                        start_time, end_time, runs_per_scenario)
                    if not is_folder_set:
                        self.file_handler.copy_all_files_from_temp_folder(
                            results_folder)
        self.vissim.ResumeUpdateGUI()
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 0)

    def run_capacity_tests_platoon_lane_change_scenario(self):
        """
        Runs the scenario only with human driven vehicles and increasing main
        road input to find out the scenarios capacity.
        """
        strategy = PlatoonLaneChangeStrategy.human_driven
        vehicle_type = VehicleType.HUMAN_DRIVEN
        main_road_speeds = ['slower', 'faster']
        orig_and_dest_lane_speeds = [(90, speed) for speed in main_road_speeds]
        vehicle_inputs = [i for i in range(500, 2501, 500)]
        self.run_multiple_platoon_lane_change_scenarios(
            [strategy], [vehicle_type], orig_and_dest_lane_speeds,
            vehicle_inputs)

    def run_multiple_platoon_lane_change_scenarios(
            self,
            strategies: List[PlatoonLaneChangeStrategy],
            main_road_vehicle_types: List[VehicleType],
            orig_and_dest_lane_speeds: List[Tuple[int, str]],
            input_per_lane: List[int],
            runs_per_scenario: int = 5,
            is_debugging: bool = False
    ):
        """

        :param strategies: Integer from 0 to 4. 0: No strategy; 1: single
         body platoon; 2: leader first; 3: last vehicle first; 4: leader first
         and revert order
        :param main_road_vehicle_types: Human driven or connected
        :param orig_and_dest_lane_speeds: Tuple where the first element is the
         platoon desired speed and the second is a string indicating "slower",
         "same" or "faster" than the platoon
        :param input_per_lane: TBD
        :param runs_per_scenario:
        :param is_debugging: If true, runs the scenario only once for a
         short time period, and results are saved to a test folder.
        """
        # Set-up simulation parameters
        if is_debugging:
            warm_up_minutes = 0
            runs_per_scenario = 1
            simulation_period = 300
            first_platoon_time = 10
        else:
            warm_up_minutes = self.network_info.warm_up_minutes
            simulation_period = self.network_info.evaluation_period
            first_platoon_time = 180
            self.vissim.Graphics.CurrentNetworkWindow.SetAttValue(
                "QuickMode", 1)
            self.vissim.SuspendUpdateGUI()

        platoon_size = 4
        orig_lane_speed_distribution = 'same'
        self.set_evaluation_options(True, True, True, True, True,
                                    warm_up_minutes * 60,
                                    data_frequency=5)
        self.set_random_seed(self._initial_random_seed)
        self.set_random_seed_increment(1)
        self.set_simulation_period(simulation_period)
        self.set_number_of_runs(runs_per_scenario)
        self.set_verbose_simulation(False)

        print("Starting multiple-scenario run.")
        multiple_sim_start_time = time.perf_counter()
        for st in strategies:
            self.set_platoon_lane_change_strategy(st)
            is_platoon_autonomous = st != PlatoonLaneChangeStrategy.human_driven
            for vt in main_road_vehicle_types:
                veh_percent = {vt: 100}
                for speed_pair in orig_and_dest_lane_speeds:
                    platoon_desired_speed = speed_pair[0]
                    orig_lane_comp_number = (
                        self.find_composition_matching_speed_distributions(
                            {vt: orig_lane_speed_distribution}))
                    dest_lane_comp_number = (
                        self.find_composition_matching_speed_distributions(
                            {vt: speed_pair[1]}))
                    input_to_comp_map = {
                        'right_lane': orig_lane_comp_number,
                        'left_lane': dest_lane_comp_number
                    }
                    self.set_vehicle_inputs_compositions(input_to_comp_map)
                    for veh_input in input_per_lane:
                        self.reset_saved_simulations(warning_active=False)
                        self.set_uniform_vehicle_input_for_all_lanes(veh_input)
                        # veh_volumes = {'left_lane': veh_input,
                        #                'right_lane': veh_input}
                        # self.set_vehicle_inputs(veh_volumes)
                        if is_debugging:
                            results_folder = (
                                self.file_handler.get_vissim_test_folder())
                        else:
                            results_folder = (
                                self.file_handler.get_vissim_data_folder(
                                    vehicle_percentages=veh_percent,
                                    vehicle_input_per_lane=veh_input,
                                    platoon_lane_change_strategy=st,
                                    orig_and_dest_lane_speeds=speed_pair)
                            )
                        is_folder_set = self.set_results_folder(results_folder)
                        print("Starting series of {} runs with duration {}".
                              format(runs_per_scenario, simulation_period))
                        start_time = time.perf_counter()
                        self.run_platoon_scenario(
                            platoon_size, platoon_desired_speed,
                            first_platoon_time=first_platoon_time,
                            platoon_creation_period=60,
                            is_platoon_autonomous=is_platoon_autonomous)
                        end_time = time.perf_counter()
                        _print_run_time_with_unit(
                            start_time, end_time, runs_per_scenario)
                        if not is_folder_set:
                            print("Copying result files to their proper "
                                  "location")
                            self.file_handler.copy_all_files_from_temp_folder(
                                results_folder)
        self.vissim.ResumeUpdateGUI()
        self.vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 0)

        multiple_sim_end_time = time.perf_counter()
        total_runs = (len(strategies) * len(main_road_vehicle_types)
                      * len(orig_and_dest_lane_speeds)
                      * len(input_per_lane) * runs_per_scenario)
        _print_run_time_with_unit(multiple_sim_start_time,
                                  multiple_sim_end_time, total_runs)


    def _check_result_folder_length(self, results_folder):
        return len(results_folder + self.file_handler.scenario_name) > 230

    def run_us_101_with_different_speed_limits(self, possible_speeds=None):
        if not self.is_correct_network_loaded():
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

    def set_evaluation_options(self,
                               save_vehicle_record: bool = False,
                               save_ssam_file: bool = False,
                               activate_data_collections: bool = False,
                               activate_link_evaluation: bool = False,
                               save_lane_changes: bool = False,
                               warm_up_time: int = 0,
                               data_frequency: int = 30):
        """
        Sets evaluation output options for various possible VISSIM outputs.
        Sets VISSIM to keep results of all runs.
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

        self.check_saved_variables()
        self.vissim.Evaluation.SetAttValue('KeepPrevResults', 'KEEPALL')

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
            print("[Client] setting parameter {} to value {}".
                  format(param_name, param_value))
            try:
                self.vissim.Simulation.SetAttValue(param_name, param_value)
            except AttributeError as err:
                self.close_vissim()
                print("Failed to set parameter")
                print("err=", err)

    def set_simulation_period(self, period):
        """Sets the period of the simulation if different from None. Otherwise,
        sets the default period based on the simulation name
        """
        if period is None:
            period = self.network_info.evaluation_period
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

    def set_results_folder(self, results_folder: str) -> bool:
        """
        Creates the result folder if it does not exist. If the path is too long
        (which may cause problem when saving files), uses a temporary folder for
        results
        :returns: A boolean indicating if the original results_folder is used.
        """
        success = True
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if self._check_result_folder_length(results_folder):
            print("Result path it too long. Saving at a temporary location.")
            # results_folder = '\\\\?\\' + results_folder
            success = False
            results_folder = self.file_handler.get_temp_results_folder()
        self.vissim.Evaluation.SetAttValue('EvalOutDir', results_folder)
        return success

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
                print('[Client] Vehicle input {} set to {}'.
                      format(vi_name, veh_input.AttValue('Volume(1)')))
                new_veh_inputs.pop(vi_name)
            else:
                print('[Client] Vehicle input {} left unchanged at {}'.
                      format(vi_name, veh_input.AttValue('Volume(1)')))

        for vi_key in new_veh_inputs:
            print('[Client] Vehicle input {} was passed as parameter, but not '
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
            print('[Client] Vehicle input {} set to {}'.
                  format(veh_input.AttValue('Name'),
                         veh_input.AttValue('Volume(1)')))

    def set_all_vehicle_inputs_composition(self, composition_number: int):
        """
        Sets the desired composition to all the vehicle inputs in the
        network.

        :param composition_number: The number of the composition in VISSIM.
        """

        veh_compositions_container = self.vissim.Net.VehicleCompositions
        veh_input_container = self.vissim.Net.VehicleInputs
        for veh_input in veh_input_container:
            veh_input.SetAttValue('VehComp(1)', composition_number)
            print('[Client] Composition of vehicle input {} set to {}'.
                  format(veh_input.AttValue('Name'),
                         veh_compositions_container.ItemByKey(
                             composition_number).AttValue('Name')))

    def set_vehicle_inputs_compositions(
            self, input_to_composition_map: Dict[str, int]):
        """
        Sets the desired composition to each vehicle input in the network.

        :param input_to_composition_map: Dictionary describing the composition
         number of each vehicle input
        """

        veh_compositions_container = self.vissim.Net.VehicleCompositions
        veh_input_container = self.vissim.Net.VehicleInputs
        for veh_input in veh_input_container:
            veh_input_name = veh_input.AttValue('Name')
            if veh_input_name in input_to_composition_map:
                comp_number = input_to_composition_map[veh_input_name]
                veh_input.SetAttValue('VehComp(1)', comp_number)
                print('[Client] Composition of vehicle input {} set to {}'.
                      format(veh_input_name,
                             veh_compositions_container.ItemByKey(
                                 comp_number).AttValue('Name')))

    def set_controlled_vehicles_percentage(
            self, vehicle_percentages: Dict[VehicleType, int]) -> int:
        """
        Looks for the specified vehicle composition and sets the
        percentage of autonomous vehicles in it. The rest of the composition
        will be made of 'human' driven vehicles.
        Assumption worth noting: the vehicle composition in VISSIM already
        exists and it contains two vehicle types: regular car and the
        controlled vehicle type

        :param vehicle_percentages: Describes the percentages of controlled
         vehicles in the simulation.
        :returns: The composition number in case more operations need to be done
        """
        # The percentage of non-controlled vehicles must be human
        total_controlled_percentage = 0
        for vt, p in vehicle_percentages.items():
            if vt != VehicleType.HUMAN_DRIVEN:
                total_controlled_percentage += p
        # sum(vehicle_percentages.values())
        percentages_with_humans = {VehicleType.HUMAN_DRIVEN:
                                   100 - total_controlled_percentage}
        percentages_with_humans.update(vehicle_percentages)
        composition_number = self.find_composition_matching_percentages(
            percentages_with_humans)
        self.set_all_vehicle_inputs_composition(composition_number)

        # Modify the relative flows
        desired_flows = {Vehicle.ENUM_TO_VISSIM_ID[vt]: p for vt, p
                         in percentages_with_humans.items()}
        veh_composition = self.vissim.Net.VehicleCompositions.ItemByKey(
            composition_number)
        for relative_flow in veh_composition.VehCompRelFlows:
            flow_vehicle_type = int(relative_flow.AttValue('VehType'))
            if (flow_vehicle_type in desired_flows
                    and desired_flows[flow_vehicle_type] > 0):
                relative_flow.SetAttValue('RelFlow',
                                          desired_flows[flow_vehicle_type])
                print('[Client] veh type {} at {}%.'.
                      format(flow_vehicle_type,
                             relative_flow.AttValue('RelFlow')))
        return composition_number

    def set_vehicle_inputs_composition_by_name(self, composition_name: str):
        """
        :returns: Nothing. Modifies the open simulation.
        """
        composition_number = self.find_vehicle_composition_by_name(
            composition_name)
        self.set_all_vehicle_inputs_composition(composition_number)

    def set_accepted_lane_change_risk_to_leaders(self, accepted_risk: float):
        if accepted_risk is None:
            accepted_risk = 0
        self.set_uda_default_value(_UDANumber.risk_to_leaders,
                                   accepted_risk)

    def set_accepted_lane_change_risk_to_follower(self, accepted_risk: float):
        if accepted_risk is None:
            accepted_risk = 0
        self.set_uda_default_value(_UDANumber.risk_to_follower,
                                   accepted_risk)

    def set_use_linear_lane_change_gap(self, use_linear_lane_change_gap: bool):
        """
        Determines whether the accepted lane change gaps are computed using a
         linear overestimation of the non-linear value
        :param use_linear_lane_change_gap:
        :return:
        """
        self.set_uda_default_value(_UDANumber.use_linear_lane_change_gap,
                                   use_linear_lane_change_gap)

    def set_platoon_lane_change_strategy(
            self, platoon_lc_strategy: PlatoonLaneChangeStrategy):
        print("[Client] Setting platoon lane change strategy to",
              platoon_lc_strategy.value, "(" + platoon_lc_strategy.name + ")")
        self.set_uda_default_value(
            _UDANumber.platoon_lane_change_strategy,
            platoon_lc_strategy.value)

    def set_verbose_simulation(self, is_simulation_verbose: bool):
        self.set_uda_default_value(_UDANumber.verbose_simulation,
                                   is_simulation_verbose)

    def set_logged_vehicle_id(self, veh_id: int):
        self.set_uda_default_value(_UDANumber.logged_vehicle,
                                   veh_id)

    def set_uda_default_value(self, uda_number: _UDANumber,
                              uda_value: Union[bool, int, float]):
        """
        Sets the default value of a user defined attribute
        :param uda_number: number identifying the uda
        :param uda_value: default value for all vehicles in the simulation
        :return: None
        """
        print("[Client] Setting {} to {}".format(uda_number.name, uda_value))
        uda = self.vissim.Net.UserDefinedAttributes.ItemByKey(
            uda_number.value)
        uda.SetAttValue('DefValue', uda_value)

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
        network_name = self.file_handler.get_network_name()
        if network_name != 'traffic_lights':
            print('No traffic lights to set.')
            return

        csv_reader = readers.TrafficLightSourceReader(
            self.file_handler.scenario_name)
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
        sc_reader = readers.SignalControllerFileReader(
            self.file_handler.scenario_name)
        sc_editor = data_writer.SignalControllerTreeEditor()
        try:
            sc_file_tree = sc_reader.load_data(sc_id)
        except ValueError:
            if sc_id == 1:
                raise OSError("Network {} doesn't have any signal "
                              "controller files (sig) defined yet"
                              .format(self.file_handler.get_network_name()))
            # If there is no file yet, we get the first file as template,
            # modify it and save with a different name
            sc_file_tree = sc_reader.load_data(1)
            sc_editor.set_signal_controller_id(sc_file_tree, sc_id)

        sc_editor.set_times(sc_file_tree, red_duration,
                            green_duration, amber_duration)
        sc_editor.save_file(sc_file_tree,
                            self.file_handler.get_network_file_folder(),
                            self.file_handler.get_file_name()
                            + str(sc_id))

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
            self.file_handler.get_network_file_folder(),
            # file_handling.get_file_name_from_network_name(self.network_name)
            self.file_handler.get_file_name()
            + str(sc_id) + '.sig'))
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

    def create_platoon(self, platoon_size, desired_speed):
        """

        :param platoon_size: Number of vehicles in the platoon
        :param desired_speed: Desired speed of platoon vehicles in km/h
        :param is_platoon_autonomous: If true, the platoon is made of
        autonomous vehicles. Otherwise, the platoon is made of human driven
        vehicles
        """

        platoon_type = vehicle.VehicleType.PLATOON
        platoon_vehicle = vehicle.Vehicle(platoon_type)
        vissim_vehicle_type = vehicle.Vehicle.ENUM_TO_VISSIM_ID[platoon_type]
        platoon_vehicle.free_flow_velocity = desired_speed / 3.6
        h, d = platoon_vehicle.compute_vehicle_following_parameters(
            leader_max_brake=platoon_vehicle.max_brake, rho=0.05)
        platoon_safe_gap = h * desired_speed / 3.6 + d

        net = self.vissim.Net
        vehicles = net.Vehicles
        right_lane_link = 2
        lane = 1
        first_platoon_position = 10

        # Remove any vehicles occupying the platoon space
        vehicles_in_link = net.Links.ItemByKey(2).Vehs
        veh_length = 6  # just an estimate
        extra_margin = 10  # to give the platoon time to react to some slower
        # moving vehicle ahead
        min_position = (first_platoon_position
                        + platoon_size * (veh_length + platoon_safe_gap)
                        + extra_margin)
        filter_str = "[Pos]<=" + str(min_position)
        vehicles_to_delete = vehicles_in_link.FilteredBy(filter_str)
        for veh in vehicles_to_delete:
            vehicles.RemoveVehicle(veh)

        # Create the platoon
        interaction = True  # optional
        position = first_platoon_position
        for i in range(platoon_size):
            added_vehicle = vehicles.AddVehicleAtLinkPosition(
                vissim_vehicle_type, right_lane_link, lane, position,
                desired_speed, interaction)
            position += added_vehicle.AttValue('Length') + platoon_safe_gap

    # HELPER FUNCTIONS --------------------------------------------------------#

    def find_composition_matching_percentages(
            self, vehicle_percentages: Dict[VehicleType, int]) -> int:
        """
        Finds the vehicle composition that has exactly the same vehicle types
        listed in the parameter

        :param vehicle_percentages: Percent of each vehicle type as a dictionary
        :return: The vehicle composition number
        """
        vehicle_type_ids = set([Vehicle.ENUM_TO_VISSIM_ID[vt] for vt in
                                vehicle_percentages
                                if vehicle_percentages[vt] > 0])
        veh_compositions_container = self.vissim.Net.VehicleCompositions
        for veh_composition in veh_compositions_container:
            counter = 0
            relative_flows_container = veh_composition.VehCompRelFlows
            # We can skip compositions with different number of relative flows
            if len(relative_flows_container) != len(vehicle_type_ids):
                continue
            for relative_flow in relative_flows_container:
                flow_vehicle_type = int(relative_flow.AttValue('VehType'))
                if flow_vehicle_type not in vehicle_type_ids:
                    continue
                counter += 1
            if counter == len(vehicle_type_ids):
                return veh_composition.AttValue('No')
        raise ValueError('[Client] Composition with {} not found in {} '
                         'network.'.format(
                            vehicle_percentages,
                            self.file_handler.get_network_name()))

    def find_composition_matching_speed_distributions(
            self, desired_speed_distribution: Dict[VehicleType, str]) -> int:
        """
        Finds the vehicle composition that has exactly the same vehicle types
        listed in the parameter

        :param desired_speed_distribution: Speed distribution per vehicle type
        :return: The vehicle composition number
        """
        # vehicle_type_ids = set([Vehicle.ENUM_TO_VISSIM_ID[vt] for vt in
        #                         vehicle_percentages
        #                         if vehicle_percentages[vt] > 0])
        speed_dist = {Vehicle.ENUM_TO_VISSIM_ID[key]: value
                      for key, value in desired_speed_distribution.items()}
        veh_compositions_container = self.vissim.Net.VehicleCompositions
        for veh_composition in veh_compositions_container:
            relative_flows_container = veh_composition.VehCompRelFlows
            # We can skip compositions with different number of relative flows
            if len(relative_flows_container) != len(desired_speed_distribution):
                continue
            current_comp = dict()
            for relative_flow in relative_flows_container:
                flow_vehicle_type = int(relative_flow.AttValue('VehType'))
                flow_speed_distribution = relative_flow.DesSpeedDistr.AttValue(
                    'Name')
                current_comp[flow_vehicle_type] = flow_speed_distribution
            if current_comp == speed_dist:
                return veh_composition.AttValue('No')

        raise ValueError('[Client] Composition with {} not found in {} '
                         'network.'.format(
                            desired_speed_distribution,
                            self.file_handler.get_network_name()))

    def find_vehicle_composition_by_name(self, name: str) -> int:
        """
        Finds the vehicle composition name with the given name
        :param name: Vehicle composition name
        :return: Vehicle composition number
        """
        veh_compositions_container = self.vissim.Net.VehicleCompositions
        for veh_composition in veh_compositions_container:
            if veh_composition.AttValue('Name').lower() == name:
                return veh_composition.AttValue('No')
        raise ValueError('[Client] Composition name not found in {} '
                         'network.'.format(
                          self.file_handler.get_network_name()))

    def set_vehicle_composition_desired_speed(self, composition_number: int,
                                              speed_distribution_name: str):
        veh_composition = self.vissim.Net.VehicleCompositions.ItemByKey(
            composition_number)
        speed_distribution = self.get_speed_distribution_by_name(
            speed_distribution_name)
        for relative_flow in veh_composition.VehCompRelFlows:
            relative_flow.SetAttValue('DesSpeedDistr', speed_distribution)
        print('[Client] Desired speed distribution of all flows in composition '
              '{} set to {}'.format(
                veh_composition.AttValue('Name'),
                speed_distribution.AttValue('Name')))

    def get_speed_distribution_by_name(self, speed_distribution_name: str):
        speed_distribution_container = self.vissim.Net.DesSpeedDistributions
        for speed_distribution in speed_distribution_container:
            if (speed_distribution.AttValue('Name').lower()
                    == speed_distribution_name.lower()):
                return speed_distribution
        raise ValueError('[Client] DesSpeedDistribution {} not found in {} '
                         'network.'.format(
                            speed_distribution_name,
                            self.file_handler.get_network_name()))

    def is_some_network_loaded(self):
        return self.vissim.AttValue('InputFile') != ''
        # if self.vissim.AttValue('InputFile') != '':
        #     # In case we loaded the simulation through VISSIM's interface:
        #     if self.file_handler is None:
        #         network_file = (
        #             self.vissim.AttValue('InputFile').split('.')[0])
        #         self.network_name = (
        #             file_handling.get_network_name_from_file_name(network_file)
        #         )
        #         # self.create_network_results_directory()
        #     return True
        # else:
        #     return False

    def is_correct_network_loaded(self):
        if (self.vissim.AttValue('InputFile')
                != self.file_handler.get_file_name() + self.vissim_net_ext):
            print('You must load the ', self.file_handler.get_network_name(),
                  ' network before running it')
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

        print('[Client] Resetting simulation count...')
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
        debug_log_folder = self.file_handler.get_vissim_test_folder()
        self.set_results_folder(debug_log_folder)

    def get_max_decel_data(self):
        """Checks the vehicle types used in simulation, creates a Panda
        dataframe with max deceleration values by vehicle type and speed, and
        saves the dataframe as csv

        :return: None
        """

        if not self.is_some_network_loaded():
            print("No network loaded")
            return
            # if self.network_name is None:
            #     print("No network file defined to load either")
            #     return
            # print("Loading network: ", self.network_name)
            # self.load_simulation(self.network_name)

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
        max_decel_df.to_csv(os.path.join(
            self.file_handler.get_network_file_folder(),
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
                            'LENGTH', 'ACCELERATION', 'LNCHG'}
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
        results_directory = self.file_handler.get_results_base_folder()
        if not os.path.isdir(results_directory):
            os.mkdir(results_directory)

    def close_vissim(self):
        self.vissim = None


def _print_run_time_with_unit(start_time, end_time, total_runs):
    """
    Computes the run time and returns it with the appropriate unit. Run times
    are given in seconds if below a minute, in minutes if below an hour,
    and in hours otherwise.
    """
    run_time = end_time - start_time
    if run_time <= 60:
        unit = 's'
    elif run_time <= 3600:
        run_time /= 60
        unit = 'min'
    else:
        run_time /= 3600
        unit = 'h'
    print('Total time: {:.1f}{} to run {} simulations.'.
          format(run_time, unit, total_runs))
