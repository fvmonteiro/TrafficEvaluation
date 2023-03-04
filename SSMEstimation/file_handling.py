import itertools
import warnings
from dataclasses import dataclass
from collections import defaultdict
import os
import shutil
from typing import Any, Dict, List, Tuple, Union

from vehicle import VehicleType, PlatoonLaneChangeStrategy, \
    vehicle_type_to_print_name_map, strategy_to_print_name_map


@dataclass
class _PCInfo:
    """Used to help run simulation in different computers"""
    easy_id: str
    networks_folder: str
    shared_folder: str
    moves_folder: str
    moves_database_port: int


@dataclass
class _NetworkInfo:
    """Contains information about different VISSIM network"""
    name: str
    file_name: str
    n_lanes: int
    main_links: List[int]


@dataclass
class ScenarioInfo:
    """Defines a VISSIM scenario. The scenario parameters must agree with
    the network being run
    vehicle_percentages: Describes the percentages of controlled
     vehicles in the simulations.
    vehicles_per_lane: Vehicle input per lane on VISSIM. Possible
     values depend on the controlled_vehicles_percentage: 500:500:2500
    accepted_risk: maximum lane changing risk in m/s
    platoon_lane_change_strategy: Coordination strategy used in platoon lane
     changing scenarios.
    orig_and_dest_lane_speeds: Mean desired speeds in the platoon lane
     changing scenario
    """
    vehicle_percentages: Dict[VehicleType, int]
    vehicles_per_lane: int
    accepted_risk: Union[int, None] = None
    platoon_lane_change_strategy: Union[PlatoonLaneChangeStrategy, None] = None
    orig_and_dest_lane_speeds: Union[Tuple[Union[str, int], Union[str, int]],
                                     None] = None
    special_case: Union[str, None] = None  # identifies test simulation runs


_folders_map = {
    'DESKTOP-P2O85S9': _PCInfo('personal_pc',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'TrafficSimulation\\VISSIM_networks',
                               'G:\\My Drive\\Safety in Mixed Traffic'
                               '\\data_exchange',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'EnvironmentalEvaluations',
                               3307
                               ),
    'DESKTOP-626HHGI': _PCInfo('usc-old',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'AV_TrafficSimulation\\VISSIM_networks',
                               'C:\\Users\\fvall\\Google Drive\\'
                               'Safety in Mixed Traffic\\data_exchange',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'EnvironmentalEvaluations',
                               3306
                               ),
    'DESKTOP-B1GECOE': _PCInfo('usc',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'TrafficSimulation\\VISSIM_networks',
                               'G:\\My Drive\\Safety in Mixed Traffic'
                               '\\data_exchange',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'EnvironmentalEvaluations',
                               3306
                               ),
}

# TODO: check main links of remaining scenarios
_network_info_all = {
    'in_and_out':
        _NetworkInfo('in_and_out', 'highway_in_and_out_lanes', 3, [3]),
    'in_and_merge':
        _NetworkInfo('in_and_merge', 'highway_in_and_merge', 3, [3]),
    'i710':
        _NetworkInfo('i710', 'I710-MultiSec-3mi', 3, []),
    'us101':
        _NetworkInfo('us101', 'US_101', 6, []),
    'traffic_lights':
        _NetworkInfo('traffic_lights', 'traffic_lights_study', 2, []),
    'platoon_mandatory_lane_change':
        _NetworkInfo('platoon_mandatory_lane_change',
                     'platoon_mandatory_lane_change', 2, [3]),
    'platoon_discretionary_lane_change':
        _NetworkInfo('platoon_discretionary_lane_change',
                     'platoon_discretionary_lane_change', 2, [1, 3])
}


def temp_name_editing():
    folder = ("C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation"
              "\\VISSIM_networks\\highway_in_and_out_lanes\\test")
    for file in os.listdir(folder):
        file_str = os.fsdecode(file)
        file_name, file_ext = file_str.split('.')
        base_name = file_name[:-3]
        num_str = int(file_name[-3:]) + 5
        new_name = base_name + str(num_str).rjust(3, '0') + '.' + file_ext
        old_file = os.path.join(folder, file_str)
        new_file = os.path.join(folder, new_name)
        os.rename(old_file, new_file)


def create_multiple_scenarios(
        vehicle_percentages: List[Dict[VehicleType, int]],
        vehicle_inputs: List[int],
        accepted_risks: List[int] = None,
        lane_change_strategies: List[PlatoonLaneChangeStrategy] = None,
        orig_and_dest_lane_speeds: List[Tuple[Union[str, int],
                                              Union[str, int]]] = None,
        special_case: str = None):
    if accepted_risks is None:
        accepted_risks = [None]
        if lane_change_strategies is None:  # not a platoon scenario
            print('[WARNING] Using empty list of accepted risks instead of '
                  '[None] might prevent the readers from working.')
    if lane_change_strategies is None:
        lane_change_strategies = [None]
    if orig_and_dest_lane_speeds is None:
        orig_and_dest_lane_speeds = [None]
    scenarios = []
    for vp, vi, ar, st, sp in itertools.product(
            vehicle_percentages, vehicle_inputs, accepted_risks,
            lane_change_strategies, orig_and_dest_lane_speeds):
        if sum(vp.values()) == 0 and ar is not None and ar > 0:
            continue
        scenarios.append(ScenarioInfo(vp, vi, ar, st, sp,
                                      special_case))
    return scenarios


def print_scenario(scenario: ScenarioInfo):
    str_list = []
    veh_percent_list = [str(p) + "% " + vt.name.lower()
                        for vt, p in scenario.vehicle_percentages.items()]
    str_list.append("Vehicles: " + ", ".join(veh_percent_list))
    str_list.append(str(scenario.vehicles_per_lane) + " vehs/lane/hour")
    if scenario.accepted_risk is not None:
        str_list.append("Accepted risk: " + str(scenario.accepted_risk))
    if scenario.platoon_lane_change_strategy is not None:
        str_list.append("Platoon LC strat.: "
                        + scenario.platoon_lane_change_strategy.name.lower())
    if scenario.orig_and_dest_lane_speeds is not None:
        str_list.append("Platoon speed "
                        + str(scenario.orig_and_dest_lane_speeds[0])
                        + " and dest lane speed: "
                        + str(scenario.orig_and_dest_lane_speeds[1]))
    if scenario.special_case is not None:
        str_list.append("Special case: " + scenario.special_case)
    return "\n".join(str_list)


def split_scenario_by(scenarios: List[ScenarioInfo], attribute: str) \
        -> Dict[Any, List[ScenarioInfo]]:
    """
    Splits a list of scenarios in subsets based on the value of the attribute.
    :returns: Dictionary where keys are unique values of the attribute and
     values are lists of scenarios.
    """
    def attribute_value_to_str(value):
        if value is None:
            return 'None'
        if attribute == 'vehicle_percentages':
            return vehicle_percentage_dict_to_string(value)
        elif attribute == 'platoon_lane_change_strategy':
            return strategy_to_print_name_map[value]
        else:
            return value

    subsets = defaultdict(list)
    for sc in scenarios:
        subsets[attribute_value_to_str(getattr(sc, attribute))].append(sc)
    return subsets


def vehicle_percentage_dict_to_string(vp_dict: Dict[VehicleType, int]) -> str:
    if sum(vp_dict.values()) == 0:
        return '100% HDV'
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + '% ' + vehicle_type_to_print_name_map[veh_type])
    return ' '.join(sorted(ret_str))


class FileHandler:
    """Class is the interface between scenario names and all their properties"""

    def __init__(self, scenario_name: str, is_data_in_cloud: bool = False):
        self.scenario_name = scenario_name
        if scenario_name.startswith('in_and_out'):
            self._network_info = _network_info_all['in_and_out']
            # Get the string after 'in_and_out'
            self.simulation_output_folder = scenario_name[len('in_and_out_'):]
        else:
            self._network_info = _network_info_all[scenario_name]
            self.simulation_output_folder = 'results'
        # self._scenario_info = _scenario_info_all[scenario_name]
        self._is_data_in_cloud = is_data_in_cloud

    def set_is_data_in_cloud(self, use_cloud_directory: bool = False):
        self._is_data_in_cloud = use_cloud_directory

    def get_network_name(self):
        return self._network_info.name

    def get_file_name(self):
        return self._network_info.file_name

    def get_network_file_relative_address(self):
        # The network files are in folder with the same name as the file
        return self._network_info.file_name

    def get_networks_folder(self):
        if self._is_data_in_cloud:
            return get_cloud_networks_folder()
        else:
            return get_local_networks_folder()

    def get_network_file_folder(self):
        return os.path.join(self.get_networks_folder(),
                            self.get_network_file_relative_address())

    def get_results_base_folder(self):
        return os.path.join(self.get_networks_folder(),
                            self.get_results_relative_address())

    def get_main_links(self) -> List[int]:
        return self._network_info.main_links
    
    def get_n_lanes(self) -> int:
        return self._network_info.n_lanes

    def get_results_relative_address(self):
        return os.path.join(self.get_network_file_relative_address(),
                            self.simulation_output_folder)

    def get_moves_default_data_folder(self):
        return os.path.join(get_moves_folder(),
                            self.get_network_file_relative_address())

    def get_vissim_test_folder(self):
        return os.path.join(self.get_results_base_folder(), 'test')

    def get_vissim_data_folder(self, scenario_info: ScenarioInfo) -> str:
        """
        Creates a string with the full path of the VISSIM simulation results
        data folder.
        """

        if (scenario_info.vehicle_percentages is not None
                and sum(scenario_info.vehicle_percentages.values()) == 0):
            scenario_info.accepted_risk = None
            base_folder = self.get_network_file_folder()
        else:
            base_folder = self.get_results_base_folder()

        return create_file_path(base_folder, scenario_info)

    def get_moves_data_folder(self, scenario_info: ScenarioInfo) -> str:
        """
        Creates a string with the full path of the MOVES data
        folder.
        """
        return create_file_path(
            self.get_moves_default_data_folder(), scenario_info)

    def find_min_max_file_number(
            self, data_identifier: str, file_format: str,
            scenario_info: ScenarioInfo) -> (int, int):
        """"
        Looks for the file with the highest simulation number. This is
        usually the file containing results from all simulations.

        :param data_identifier: last part of the file name
        :param file_format: file extension
        :param scenario_info: Simulation scenario parameters
        :return: (min, max) simulation number.
        """
        max_simulation_number = -1
        min_simulation_number = 10000

        results_folder = self.get_vissim_data_folder(scenario_info)
        network_file = self.get_file_name()
        for file in os.listdir(results_folder):
            file_str = os.fsdecode(file)
            if (file_str.startswith(network_file + data_identifier)
                    and file_str.endswith(file_format)):
                file_no_extension = file_str.split('.')[0]
                try:
                    sim_number = int(file_no_extension.split('_')[-1])
                except ValueError:
                    print('File {} is not being read because its name does not '
                          'end with a number.'.format(file_no_extension))
                    continue
                if sim_number > max_simulation_number:
                    max_simulation_number = sim_number
                if sim_number < min_simulation_number:
                    min_simulation_number = sim_number

        if max_simulation_number == -1:
            file_base_name = network_file + data_identifier + file_format
            raise FileNotFoundError('File {} not found in directory {}'
                                    .format(file_base_name, results_folder))

        return min_simulation_number, max_simulation_number

    def export_multiple_results_to_cloud(self, scenarios: List[ScenarioInfo]):
        self._move_multiple_results(True, scenarios)

    def import_multiple_results_from_cloud(
            self, scenarios: List[ScenarioInfo]):
        self._move_multiple_results(False, scenarios)

    def _move_multiple_results(
            self, is_exporting: bool, scenarios: List[ScenarioInfo]):
        for sc in scenarios:
            try:
                self.move_result_files(is_exporting, sc)
            except FileNotFoundError:
                destination = 'cloud' if is_exporting else 'local'
                print("Couldn't move scenario {} to {} "
                      "folder.".format(sc, destination))
                continue

    def move_result_files(self, is_exporting: bool,
                          scenario_info: ScenarioInfo):
        """
        Moves data collections, link segments, and all post-processed data
        from the local to cloud folder is is_exporting is true, or from cloud
        to local is is_exporting is false.
        """

        temp = self._is_data_in_cloud
        if is_exporting:
            self.set_is_data_in_cloud(False)
            target_base = get_cloud_networks_folder()
            source_base = get_local_networks_folder()
        else:
            self.set_is_data_in_cloud(True)
            target_base = get_local_networks_folder()
            source_base = get_cloud_networks_folder()

        source_dir = self.get_vissim_data_folder(scenario_info)
        target_dir = os.path.join(target_base,
                                  source_dir.split(source_base + "\\")[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        all_file_names = os.listdir(source_dir)
        all_csv_files = [file for file in all_file_names if
                         file.endswith('csv')]
        try:
            _, max_file_number = self.find_min_max_file_number(
                '', 'att', scenario_info)
        except FileNotFoundError:
            print("No data collection measurements or link evaluation results"
                  " files found.")
            max_file_number = 0
        max_att_files = [file for file in all_file_names if
                         file.endswith(str(max_file_number) + '.att')]
        files_to_copy = all_csv_files + max_att_files
        for file_name in files_to_copy:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)

        self.set_is_data_in_cloud(temp)

    def get_temp_results_folder(self):
        return os.path.join(self.get_networks_folder(), "temp_results")

    def copy_all_files_from_temp_folder(self, target_dir):
        source_dir = self.get_temp_results_folder()
        all_files = os.listdir(source_dir)
        for file_name in all_files:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)


def create_percent_folder_name(
        vehicle_percentages: Dict[VehicleType, int]) -> str:
    """Creates the name of the folder which contains results for the
    given percentage of controlled vehicles (not the full path)"""

    if sum(vehicle_percentages.values()) == 0:
        return '0_percent_'
    all_strings = []
    for vt, p in vehicle_percentages.items():
        if p > 0:
            all_strings += [str(int(p)), 'percent', vt.name.lower()]
    return '_'.join(all_strings)


def get_local_networks_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].networks_folder


def get_cloud_networks_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].shared_folder


def get_moves_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].moves_folder


def get_moves_database_port() -> int:
    return _folders_map[os.environ['COMPUTERNAME']].moves_database_port


def create_vehs_per_lane_folder_name(vehicles_per_lane: Union[int, str]) -> str:
    """Creates the name of the folder which contains results for the
    given vehicle per lane input (not the full path)"""

    if vehicles_per_lane is None:
        return ''
    if isinstance(vehicles_per_lane, str):
        vehs_per_lane_folder = vehicles_per_lane
    else:
        vehs_per_lane_folder = str(int(vehicles_per_lane)) + '_vehs_per_lane'
    return vehs_per_lane_folder


def create_accepted_risk_folder_name(accepted_risk: Union[int, None]) -> str:
    """
    Creates the name of the folder which contains the results for the given
    maximum accepted lane change risk
    :param accepted_risk: simulation's maximum accepted lane change risk
    :return: folder name as: [accepted_risk]_accepted_risk
    """
    return ('' if accepted_risk is None
            else str(accepted_risk) + '_accepted_risk')


def create_platoon_lc_strategy_folder_name(
        platoon_lc_strategy: PlatoonLaneChangeStrategy) -> str:
    return platoon_lc_strategy.name


def create_speeds_folder_name(orig_and_dest_lane_speeds: Tuple[int, str]
                              ) -> str:
    return '_'.join(['origin_lane', str(orig_and_dest_lane_speeds[0]),
                     'dest_lane', str(orig_and_dest_lane_speeds[1])])


def create_file_path(
        base_folder: str, scenario_info: ScenarioInfo) -> str:
    """
    Creates a string with the full path of the data
    folder.

    :param base_folder: full path of the base folder where the data is
    :param scenario_info: Simulation scenario parameters
    :return: string with the folder where the data is
    """
    if (scenario_info.vehicle_percentages is None
            and scenario_info.vehicles_per_lane is None):
        warnings.warn("Using create_file_path to obtain the test folder"
                      " is deprecated. Use get_vissim_test_folder instead.")
        return os.path.join(base_folder, 'test')

    folder_list = [base_folder]
    if scenario_info.platoon_lane_change_strategy is not None:
        folder_list.append(create_platoon_lc_strategy_folder_name(
            scenario_info.platoon_lane_change_strategy))
    if scenario_info.vehicles_per_lane > 0:
        folder_list.append(create_percent_folder_name(
            scenario_info.vehicle_percentages))
        folder_list.append(create_vehs_per_lane_folder_name(
                scenario_info.vehicles_per_lane))
        if scenario_info.orig_and_dest_lane_speeds is not None:
            folder_list.append(create_speeds_folder_name(
                scenario_info.orig_and_dest_lane_speeds))
    else:
        folder_list.append(create_vehs_per_lane_folder_name(
                scenario_info.vehicles_per_lane))
    if scenario_info.accepted_risk is not None:
        folder_list.append(create_accepted_risk_folder_name(
            scenario_info.accepted_risk))
    if scenario_info.special_case is not None:
        folder_list.append(scenario_info.special_case)

    return os.path.join(*folder_list)
