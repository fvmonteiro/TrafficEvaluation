import warnings
from dataclasses import dataclass
import os
import shutil
from typing import Dict, List, Tuple, Union

from vehicle import VehicleType, PlatoonLaneChangeStrategy


@dataclass
class _PCInfo:
    """Used to help run simulation in different computers"""
    easy_id: str
    networks_folder: str
    shared_folder: str
    moves_folder: str


@dataclass
class _NetworkInfo:
    """Contains information about different VISSIM network"""
    file_name: str
    total_lanes: int
    main_links: List[int]
    # on_ramp_link: List[int]
    # off_ramp_link: List[int]
    # merging_link: List[int]


@dataclass
class _ScenarioInfo:
    """Contains information about different simulation scenarios"""
    network_name: str
    network: _NetworkInfo
    results_folder: str

    def __init__(self, network_name, results_folder):
        self.network_name = network_name
        self.network_info = _network_info_all[network_name]
        self.results_folder = results_folder


_folders_map = {
    'DESKTOP-P2O85S9': _PCInfo('personal_pc',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'TrafficSimulation\\VISSIM_networks',
                               'G:\\My Drive\\Safety in Mixed Traffic'
                               '\\data_exchange',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'EnvironmentalEvaluations'
                               ),
    'DESKTOP-626HHGI': _PCInfo('usc-old',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'AV_TrafficSimulation\\VISSIM_networks',
                               'C:\\Users\\fvall\\Google Drive\\'
                               'Safety in Mixed Traffic\\data_exchange',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'EnvironmentalEvaluations'
                               ),
    'DESKTOP-B1GECOE': _PCInfo('usc',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'TrafficSimulation\\VISSIM_networks',
                               'G:\\My Drive\\Safety in Mixed Traffic'
                               '\\data_exchange',
                               'C:\\Users\\fvall\\Documents\\Research\\'
                               'EnvironmentalEvaluations'
                               ),
}

# TODO: check main links of remaining scenarios
_network_info_all = {
    'in_and_out':
        _NetworkInfo('highway_in_and_out_lanes', 3, [3]),
    'in_and_merge':
        _NetworkInfo('highway_in_and_merge', 3, [3]),
    'i710':
        _NetworkInfo('I710-MultiSec-3mi', 3, []),
    'us101':
        _NetworkInfo('US_101', 6, []),
    'traffic_lights':
        _NetworkInfo('traffic_lights_study', 2, []),
    'platoon_mandatory_lane_change':
        _NetworkInfo('platoon_mandatory_lane_change', 2, [3]),
    'platoon_discretionary_lane_change':
        _NetworkInfo('platoon_discretionary_lane_change', 2, [3])
}

_scenario_info_all = {
    'in_and_out':
        _ScenarioInfo('in_and_out', ''),
    'in_and_out_safe':
        _ScenarioInfo('in_and_out', 'safe'),
    'in_and_out_risk_in_headway':
        _ScenarioInfo('in_and_out', 'risk_in_headway'),
    'in_and_out_risk_in_gap':
        _ScenarioInfo('in_and_out', 'risk_in_gap'),
    'in_and_merge':
        _ScenarioInfo('in_and_merge', 'results'),
    'i710':
        _ScenarioInfo('i710', 'results'),
    'us101':
        _ScenarioInfo('us101', 'results'),
    'traffic_lights':
        _ScenarioInfo('traffic_lights', 'traffic_lights_study'),
    'platoon_mandatory_lane_change':
        _ScenarioInfo('platoon_mandatory_lane_change', 'results'),
    'platoon_discretionary_lane_change':
        _ScenarioInfo('platoon_discretionary_lane_change', 'results')
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


class FileHandler:
    """Class is the interface between scenario names and all their properties"""

    def __init__(self, scenario_name: str, is_data_in_cloud: bool = False):
        self.scenario_name = scenario_name
        self._scenario_info = _scenario_info_all[scenario_name]
        self._is_data_in_cloud = is_data_in_cloud
        # self.network_name = _scenario_info[scenario_name].network

    def set_is_data_in_cloud(self, use_cloud_directory: bool = False):
        self._is_data_in_cloud = use_cloud_directory

    def get_network_name(self):
        return self._scenario_info.network_name

    def get_file_name(self):
        return self._scenario_info.network_info.file_name

    def get_network_file_relative_address(self):
        # The network files are in folder with the same name as the file
        return self._scenario_info.network_info.file_name

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
        return self._scenario_info.network_info.main_links

    def get_results_relative_address(self):
        return os.path.join(self.get_network_file_relative_address(),
                            self._scenario_info.results_folder)

    def get_moves_default_data_folder(self):
        return os.path.join(get_moves_folder(),
                            self.get_network_file_relative_address())

    def get_vissim_test_folder(self):
        return os.path.join(self.get_results_base_folder(), 'test')

    def get_vissim_data_folder(
            self, vehicle_percentages: Dict[VehicleType, int],
            vehicle_input_per_lane: int, accepted_risk: int = None,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, str] = None) -> str:
        """
        Creates a string with the full path of the VISSIM simulation results
        data folder.
        """

        if (vehicle_percentages is not None
                and sum(vehicle_percentages.values()) == 0):
            accepted_risk = None
            base_folder = self.get_network_file_folder()
        else:
            base_folder = self.get_results_base_folder()

        return create_file_path(
            base_folder, vehicle_percentages, vehicle_input_per_lane,
            accepted_risk, platoon_lane_change_strategy,
            orig_and_dest_lane_speeds)

    def get_moves_data_folder(
            self, vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: int, accepted_risk: int,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, str] = None) -> str:
        """
        Creates a string with the full path of the MOVES data
        folder. If all parameters are None, returns the test data folder

        :param vehicle_percentages: Describes the percentages of controlled
         vehicles in the simulations.
        :param vehicles_per_lane: Vehicle input per lane on VISSIM. Possible
         values depend on the controlled_vehicles_percentage: 500:500:2500
        :param accepted_risk: maximum lane changing risk in m/s
        :return: string with the folder where the data is
        """
        return create_file_path(
            self.get_moves_default_data_folder(), vehicle_percentages,
            vehicles_per_lane, accepted_risk, platoon_lane_change_strategy,
            orig_and_dest_lane_speeds)

    def find_min_max_file_number(
            self, data_identifier: str, file_format: str,
            vehicle_percentages: Dict[VehicleType, int],
            vehicle_input_per_lane: int,
            accepted_risk: int = None,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, str] = None) -> (int, int):
        """"
        Looks for the file with the highest simulation number. This is
        usually the file containing results from all simulations.

        :param data_identifier: last part of the file name
        :param file_format: file extension
        :param vehicle_percentages: Describes the percentages of controlled
         vehicles in the simulations.
        :param vehicle_input_per_lane: Vehicle input per lane used in simulation
        :param platoon_lane_change_strategy: Coordination strategy used in
         platoon lane changing scenarios.
        :param accepted_risk: accepted lane change risk
        :param orig_and_dest_lane_speeds: Mean desired speeds in the platoon
         lane changing scenario
        :return: (min, max) simulation number.
        """
        max_simulation_number = -1
        min_simulation_number = 10000

        results_folder = self.get_vissim_data_folder(
            vehicle_percentages, vehicle_input_per_lane, accepted_risk,
            platoon_lane_change_strategy, orig_and_dest_lane_speeds)
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

    def export_multiple_results_to_cloud(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            accepted_risks: List[int]):
        for vp in vehicle_percentages:
            for vi in vehicles_per_lane:
                for ar in accepted_risks:
                    try:
                        self.move_result_files(True, vp, vi, accepted_risk=ar)
                    except FileNotFoundError:
                        print("Couldn't export {}, {}, {} to shared "
                              "folder.".format(vp, vi, ar))
                        continue

    def export_multiple_platoon_results_to_cloud(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            origin_and_dest_lane_speeds: List[Tuple[int, str]]):
        self._move_multiple_platoon_results(
            True, vehicle_percentages, vehicles_per_lane,
            lane_change_strategies, origin_and_dest_lane_speeds)

    def import_multiple_platoon_results_from_cloud(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            origin_and_dest_lane_speeds: List[Tuple[int, str]]):
        self._move_multiple_platoon_results(
            False, vehicle_percentages, vehicles_per_lane,
            lane_change_strategies, origin_and_dest_lane_speeds)

    def _move_multiple_platoon_results(
            self, is_exporting: bool,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            origin_and_dest_lane_speeds: List[Tuple[int, str]]):
        for st in lane_change_strategies:
            for vp in vehicle_percentages:
                for vi in vehicles_per_lane:
                    for speed_pair in origin_and_dest_lane_speeds:
                        try:
                            self.move_result_files(
                                is_exporting, vp, vi,
                                platoon_lane_change_strategy=st,
                                origin_and_dest_lane_speeds=speed_pair)
                        except FileNotFoundError:
                            destination = 'cloud' if is_exporting else 'local'
                            print("Couldn't move {}, {}, {}, {} to {} "
                                  "folder.".format(st.name, vp, vi,
                                                   speed_pair, destination))
                            continue

    def move_result_files(
            self, is_exporting: bool,
            vehicle_percentages: Dict[VehicleType, int],
            vehicle_input_per_lane: int,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            accepted_risk: int = None,
            origin_and_dest_lane_speeds: Tuple[int, str] = None):
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

        source_dir = self.get_vissim_data_folder(
            vehicle_percentages, vehicle_input_per_lane,
            accepted_risk, platoon_lane_change_strategy,
            origin_and_dest_lane_speeds)
        target_dir = os.path.join(target_base,
                                  source_dir.split(source_base + "\\")[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        all_file_names = os.listdir(source_dir)
        all_csv_files = [file for file in all_file_names if
                         file.endswith('csv')]
        try:
            _, max_file_number = self.find_min_max_file_number(
                '', 'att', vehicle_percentages, vehicle_input_per_lane,
                accepted_risk, platoon_lane_change_strategy,
                origin_and_dest_lane_speeds)
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
    # vehicle_type_names = [v.name.lower() for v in vehicle_type]
    # percentage_folder = ''
    # for v, p in sorted(zip(vehicle_type_names, percentage)):
    #     if p > 0:
    #         percentage_folder += str(int(p)) + '_percent_' + v + '_'
    #
    # return percentage_folder[:-1]  # remove last '_'


def get_local_networks_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].networks_folder


def get_cloud_networks_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].shared_folder


def get_moves_folder() -> str:
    return _folders_map[os.environ['COMPUTERNAME']].moves_folder


def get_scenario_number_of_lanes(scenario_name: str) -> int:
    return _scenario_info_all[scenario_name].network_info.total_lanes


def get_scenario_main_links(scneario_name: str) -> List[int]:
    return _scenario_info_all[scneario_name].network_info.main_links


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
        base_folder: str, vehicle_percentages: Dict[VehicleType, int],
        vehicle_input_per_lane: int, accepted_risk: Union[int, None],
        platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
        orig_and_dest_lane_speeds: Tuple[int, str] = None) -> str:
    """
    Creates a string with the full path of the data
    folder. If all parameters are None, returns the test data folder

    :param base_folder: full path of the base folder where the data is
    :param vehicle_percentages: Describes the percentages of controlled
     vehicles in the simulations.
    :param vehicle_input_per_lane: Vehicle input per lane on VISSIM. Possible
     values depend on the controlled_vehicles_percentage: 500:500:2500
    :param accepted_risk: maximum lane changing risk in m/s
    :return: string with the folder where the data is
    """
    if (vehicle_percentages is None
            and vehicle_input_per_lane is None):
        warnings.warn("Using create_file_path to obtain the test folder"
                      " is deprecated. Use get_vissim_test_folder instead.")
        return os.path.join(base_folder, 'test')

    folder_list = [base_folder]
    if platoon_lane_change_strategy is not None:
        folder_list.append(create_platoon_lc_strategy_folder_name(
            platoon_lane_change_strategy))
    if vehicle_input_per_lane > 0:
        folder_list.append(create_percent_folder_name(vehicle_percentages))
        folder_list.append(create_vehs_per_lane_folder_name(
                vehicle_input_per_lane))
        if orig_and_dest_lane_speeds is not None:
            folder_list.append(create_speeds_folder_name(
                orig_and_dest_lane_speeds))
    else:
        folder_list.append(create_vehs_per_lane_folder_name(
                vehicle_input_per_lane))
    if accepted_risk is not None:
        folder_list.append(create_accepted_risk_folder_name(
            accepted_risk))

    return os.path.join(*folder_list)
