from abc import ABC, abstractmethod
import os
from typing import Union
import warnings

import mariadb
import pandas as pd
import xml.etree.ElementTree as ET

from file_handling import FileHandler, get_moves_database_port
from scenario_handling import ScenarioInfo
from vehicle import PlatoonLaneChangeStrategy, VehicleType


def match_sim_number_to_random_seed(data) -> None:
    """Matches each simulation number to the used random seed. This is only
    possible if we know the initial random seed, the random seed increment,
    and the number of runs per vehicle input."""
    # TODO: where should this function and these constants be saved?
    # These variables are needed because we only save the simulation number,
    # which doesn"t mean much unless all percentages had the
    # exact same number of simulations.
    _first_simulation_number = 1
    _runs_per_input = 10
    _initial_random_seed = 7
    _random_seed_increment = 1
    if not data.empty:
        data["random_seed"] = _initial_random_seed + (
                (data["simulation_number"] - _first_simulation_number)
                % _runs_per_input) * _random_seed_increment


def _add_scenario_info_columns(data: pd.DataFrame,
                               scenario_info: ScenarioInfo) -> None:
    """
    Modifies data in place
    """
    # Vehicle percentages
    s = ""
    if sum(scenario_info.vehicle_percentages.values()) == 0:
        s = "100% HDV"
    for vt, p in scenario_info.vehicle_percentages.items():
        data[vt.name.lower() + "_percentage"] = p
        if p > 0:
            s += str(p) + "% " + vt.get_print_name()
    data["control percentages"] = s

    data["vehicles_per_lane"] = int(scenario_info.vehicles_per_lane)
    if scenario_info.accepted_risk is not None:
        data["accepted_risk"] = int(scenario_info.accepted_risk)
    if scenario_info.platoon_lane_change_strategy is not None:
        data["lane_change_strategy"] = (
            scenario_info.platoon_lane_change_strategy.get_print_name())
    else:
        data["lane_change_strategy"] = "None"
    if scenario_info.orig_and_dest_lane_speeds is not None:
        data["orig_lane_speed"] = scenario_info.orig_and_dest_lane_speeds[0]
        data["dest_lane_speed"] = scenario_info.orig_and_dest_lane_speeds[1]
    if scenario_info.computation_time is not None:
        data["computation_time"] = scenario_info.computation_time
    if scenario_info.platoon_size is not None:
        data["platoon_size"] = scenario_info.platoon_size
    # _add_special_case_columns(data, scenario_info.special_case)


def _add_vehicle_type_columns(data: pd.DataFrame,
                              vehicle_percentages: dict[VehicleType, int]
                              ) -> None:
    if vehicle_percentages is not None:
        s = ""
        if sum(vehicle_percentages.values()) == 0:
            s = "100% HDV"
        for vt, p in vehicle_percentages.items():
            data[vt.name.lower() + "_percentage"] = p
            if p > 0:
                s += str(p) + "% " + vt.get_print_name()
        data["control percentages"] = s


def _add_vehicle_input_column(data: pd.DataFrame,
                              vehicles_per_lane: int) -> None:
    if vehicles_per_lane is not None:
        data["vehicles_per_lane"] = int(vehicles_per_lane)


def _add_risk_column(data: pd.DataFrame,
                     accepted_risk: Union[int, None]) -> None:
    if accepted_risk is not None:
        data["accepted_risk"] = int(accepted_risk)


def _add_platoon_lane_change_strategy_column(
        data: pd.DataFrame, strategy: PlatoonLaneChangeStrategy) -> None:
    if strategy is not None:
        data["lane_change_strategy"] = strategy.get_print_name()
    else:
        data["lane_change_strategy"] = "None"


def _add_speeds_column(data: pd.DataFrame,
                       orig_and_dest_lane_speeds: tuple[int, str]) -> None:
    if orig_and_dest_lane_speeds is not None:
        data["orig_lane_speed"] = orig_and_dest_lane_speeds[0]
        data["dest_lane_speed"] = orig_and_dest_lane_speeds[1]


def _add_special_case_columns(data: pd.DataFrame, special_case: str) -> None:

    platoon_size = 4
    # platoon_desired_speed = 110
    first_platoon_time = 180
    # creation_period = 60
    if special_case is None:
        return
    elif special_case == "no_lane_change":
        simulation_period = 600
        first_platoon_time = simulation_period + 1
        creation_period = simulation_period
    elif special_case == "single_lane_change":
        simulation_period = 1200
        creation_period = simulation_period + 1
    elif special_case.endswith("lane_change_period"):
        creation_period = int(special_case.split("_")[0])
    elif special_case.endswith("platoon_vehicles"):
        simulation_period = 1200
        creation_period = simulation_period + 1  # single lane change
        platoon_size = int(special_case.split("_")[0])
    else:
        raise ValueError("Unknown special case: {}.".format(special_case))
    data["platoon_size"] = platoon_size
    data["creation_period"] = creation_period
    data["first_platoon_time"] = first_platoon_time


class DataReader(ABC):

    def __init__(self, scenario_name=None):
        self.scenario_name = scenario_name

    @abstractmethod
    def load_data(self, file_identifier) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_data_from_several_scenarios(
            self, scenarios: list[ScenarioInfo]) -> pd.DataFrame:
        """
        :param scenarios: List of simulation parameters for several scenarios
        """
        pass

    def load_test_data(
            self, scenario_info: ScenarioInfo) -> pd.DataFrame:
        pass


class VissimDataReader(DataReader):
    """Base class to read data generated by VISSIM"""

    def __init__(self, scenario_name: str,
                 file_format: str, separator: str,
                 data_identifier: str, header_identifier: str,
                 header_map: dict):

        self.file_handler = FileHandler(scenario_name)
        # network_data_dir = self.file_handler.get_results_base_folder()
        DataReader.__init__(self, scenario_name)
        # self.vehicle_type = vehicle_type.name.lower()
        self.file_format = file_format
        self.separator = separator
        self.data_identifier = data_identifier
        self.header_identifier = header_identifier
        self.header_map = header_map

    # @staticmethod
    # def load_max_deceleration_data():
    #     """ Loads data describing maximum deceleration distribution per
    #     vehicle
    #      type and velocity
    #
    #     :return: pandas dataframe with double index
    #     """
    #     max_deceleration_data = pd.read_csv(os.path.join(
    #         vissim_networks_folder, "max_decel_data.csv"))
    #     kph_to_mps = 1 / 3.6
    #     max_deceleration_data["vel"] = max_deceleration_data["vel"]
    #     * kph_to_mps
    #     max_deceleration_data.set_index(["veh_type", "vel"], inplace=True)
    #     return max_deceleration_data

    def load_data(self, file_identifier: str,
                  n_rows: int = None) -> pd.DataFrame:
        """ Loads data in the full path given by file_identifier
        Accepting None vehicle_type and vehicles_per_lane during testing phase

        :param file_identifier: File full path
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :return: pandas dataframe with the data
        """

        full_address = file_identifier
        try:
            with open(full_address, "r") as file:
                # Skip header lines
                for line in file:
                    # In all VISSIM files, the data starts after a line such as
                    # "$VEHICLE:". The variable names are listed after the ":".
                    if line.startswith(self.header_identifier):
                        header_no_split = line
                        if ":" in header_no_split:
                            header_no_split = line.partition(":")[-1]
                        file_header = header_no_split.rstrip("\n").split(
                            self.separator)
                        break

                column_names = []
                for variable_name in file_header:
                    extra_info = ""
                    if "(" in variable_name:
                        opening_idx = variable_name.find("(")
                        closing_idx = variable_name.find(")")
                        extra_info = (
                            variable_name[opening_idx:closing_idx + 1])
                        variable_name = variable_name[:opening_idx]
                    try:
                        column_names.append(self.header_map[
                                                variable_name.lstrip(" ")]
                                            + extra_info)
                    except KeyError:
                        column_names.append(variable_name.lower() + extra_info)
                data = pd.read_csv(file, sep=self.separator,
                                   dtype={"state": str},
                                   names=column_names, index_col=False,
                                   nrows=n_rows)
        except OSError:
            raise ValueError("No VISSIM file at {}".format(file_identifier))

        data.dropna(axis="columns", how="all", inplace=True)
        return data

    def load_data_from_scenario(
            self, scenario_info: ScenarioInfo,
            n_rows: int = None) -> pd.DataFrame:
        """
        Loads all the simulation data from the scenario described by the
        parameters.

        :param scenario_info: Simulation scenario parameters
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :return: pandas dataframe with the data
        """
        pass

    def load_test_data(
            self, scenario_info: ScenarioInfo) -> pd.DataFrame:
        file_number = 1
        file_name = self._create_file_name(file_number)
        data_folder = self.file_handler.get_vissim_test_folder()
        full_address = os.path.join(data_folder, file_name)
        data = self.load_data(full_address)
        if "simulation_number" not in data.columns:
            data["simulation_number"] = file_number
        _add_scenario_info_columns(data, scenario_info)
        return data

    def load_data_from_several_scenarios(
            self, scenarios: list[ScenarioInfo]) -> pd.DataFrame:
        """
        :param scenarios: List of simulation parameters for several scenarios
        """
        data_per_folder = []
        for sc in scenarios:
            new_data = self.load_data_from_scenario(sc)
            data_per_folder.append(new_data)
        data = pd.concat(data_per_folder, ignore_index=True)
        match_sim_number_to_random_seed(data)
        return data

    def load_single_file_from_scenario(
            self, file_identifier: int,
            scenario_info: ScenarioInfo,
            n_rows: int = None) -> pd.DataFrame:
        """
        Creates the file address based on the scenario parameters and loads
        the data.

        :param file_identifier: An integer indicating the simulation number
         Used for debugging purposes.
        :param scenario_info: Simulation scenario parameters
        :param n_rows: Defines how many rows to read. Default None reads the
         entire file
        :return: pandas dataframe with the data
        """

        full_address = self._create_full_file_address(
            file_identifier, scenario_info)
        data = self.load_data(full_address, n_rows=n_rows)
        if "simulation_number" not in data.columns:
            data["simulation_number"] = file_identifier
        _add_scenario_info_columns(data, scenario_info)
        return data

    def _create_full_file_address(
            self, file_identifier: Union[int, str],
            scenario_info: ScenarioInfo) -> str:
        """

        :param file_identifier: This can be either a integer indicating
         the simulation number or the file name directly
        :return: string with the full file address ready to be opened
        """
        if isinstance(file_identifier, str):
            file_name = file_identifier
        else:
            file_name = self._create_file_name(file_identifier)
        data_folder = self.file_handler.get_vissim_data_folder(
            scenario_info)
        return os.path.join(data_folder, file_name)

    def _create_file_name(self, file_identifier: int = None) -> str:
        network_file = self.file_handler.get_file_name()
        if file_identifier is not None:
            # Create a three-character string with trailing zeros and then
            # sim_nums (e.g.: _004, _015, _326)
            num_str = "_" + str(file_identifier).rjust(3, "0")
        else:
            num_str = ""
        file_name = (network_file + self.data_identifier
                     + num_str + self.file_format)
        return file_name

    @staticmethod
    def _keep_only_aggregated_data(data: pd.DataFrame) -> None:
        """
        Some files contains data aggregated for all vehicle categories and
        then detailed for each vehicle category. This method keeps only the
        aggregated data
        :param data: the data to be cleaned
        :return:
        """
        cols_to_be_dropped = []
        for name in data.columns:
            if "(" in name and name.split("(")[1][:-1] != "ALL":
                cols_to_be_dropped.append(name)
        data.drop(columns=cols_to_be_dropped, inplace=True)
        # Remove (ALL) from the column names
        column_names = [name.split("(")[0] for name in data.columns]
        data.columns = column_names


class AggregatedDataReader(VissimDataReader):
    """ Used to read aggregated simulation results, such as link evaluation
    and data collection results. """

    def load_data_from_scenario(
            self, scenario_info: ScenarioInfo,
            n_rows: int = None) -> pd.DataFrame:
        """
        Loads all the simulation data from the scenario described by the
        parameters.

        :param scenario_info: Simulation scenario parameters
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :return: pandas dataframe with the data
        """
        # In aggregated data, the file with the highest number contains the
        # results from all previous simulations
        _, max_file_number = (
            self.file_handler.find_min_max_file_number(
                self.data_identifier, self.file_format, scenario_info))
        return self.load_single_file_from_scenario(
            max_file_number, scenario_info, n_rows)


class VehicleRecordReader(VissimDataReader):
    """Reads vehicle records generated by VISSIM"""

    _file_format = ".fzp"
    _separator = ";"
    _data_identifier = ""
    _header_identifier = "$VEHICLE"
    _header_map = {
        "SIMRUN": "simulation_number", "SIMSEC": "time", "NO": "veh_id",
        "VEHTYPE": "veh_type", "LANE\\LINK\\NO": "link",
        "LANE\\INDEX": "lane", "POS": "x", "SPEED": "vx",
        "ACCELERATION": "ax", "POSLAT": "y", "LEADTARGNO": "leader_id",
        "FOLLOWDIST": "vissim_delta_x",
        "COORDFRONTX": "front_x", "COORDFRONTY": "front_y",
        "COORDREARX": "rear_x", "COORDREARY": "rear_y",
        "SPEEDDIFF": "vissim_delta_v", "LENGTH": "length",
        "LNCHG": "lane_change", "GIVECONTROLTOVISSIM": "vissim_control",
        "LEADTARGTYPE": "target_type", "CURRENTSTATE": "state",
        "PLATOONID": "platoon_id", "PLATOONLCSTRATEGY": "lc_strategy"
    }

    # Note: we don"t necessarily want all the variables listed in the map above

    def __init__(self, scenario_name):
        VissimDataReader.__init__(self, scenario_name,
                                  self._file_format, self._separator,
                                  self._data_identifier,
                                  self._header_identifier, self._header_map)

    def load_data_from_scenario(
            self, scenario_info: ScenarioInfo,
            n_rows: int = None, emit_warning: bool = True) -> pd.DataFrame:
        if emit_warning:
            print("[VehicleRecordReader] To load all data from this scenario, "
                  "use the generator methods.\n"
                  "Returning single file from the scenario")
        file_number = 1
        return self.load_single_file_from_scenario(
                file_number, scenario_info, n_rows)

    def load_sample_data_from_scenario(
            self, scenario_info: ScenarioInfo,
            n_rows: int = None) -> pd.DataFrame:
        """Loads only the first vehicle record data from the scenario."""
        return self.load_data_from_scenario(scenario_info, n_rows,
                                            emit_warning=False)

    def generate_all_data_from_scenario(
            self, scenario_info: ScenarioInfo,
            n_rows: int = None) -> tuple[pd.DataFrame, int]:
        """
        Yields all the vehicle record files for the chosen simulation scenario.

        :param scenario_info: Simulation scenario parameters
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :yields:
        """

        min_file_number, max_file_number = (
            self.file_handler.find_min_max_file_number(
                self.data_identifier, self.file_format, scenario_info))
        for file_number in range(min_file_number, max_file_number + 1):
            print("Loading file number {} / {}".format(
                file_number - min_file_number + 1,
                max_file_number - min_file_number + 1))
            yield (self.load_single_file_from_scenario(
                file_number, scenario_info, n_rows), file_number)

    def generate_data_from_several_scenarios(
            self, scenarios: list[ScenarioInfo],
            n_rows: int = None) -> pd.DataFrame:
        """

        :param scenarios: List of simulation parameters for several scenarios
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        """
        for sc in scenarios:
            print(sc)
            yield from self.generate_all_data_from_scenario(sc, n_rows)


class DataCollectionReader(AggregatedDataReader):
    """Reads data generated from data collection measurements in VISSIM"""

    _file_format = ".att"
    _separator = ";"
    _data_identifier = "_Data Collection Results"
    _header_identifier = "$DATACOLLECTIONMEASUREMENTEVALUATION"
    _header_map = {
        "SIMRUN": "simulation_number", "TIMEINT": "time_interval",
        "DATACOLLECTIONMEASUREMENT": "sensor_number",
        "DIST": "distance", "VEHS": "vehicle_count",
        "QUEUEDELAY": "queue_delay", "OCCUPRATE": "occupancy_rate",
        "ACCELERATION": "acceleration", "LENGTH": "length",
        "PERS": "people count", "SPEEDAVGARITH": "speed_avg",
        "SPEEDAVGHARM": "speed_harmonic_avg"
    }

    def __init__(self, scenario_name):
        VissimDataReader.__init__(self, scenario_name,
                                  self._file_format, self._separator,
                                  self._data_identifier,
                                  self._header_identifier, self._header_map)

    def load_data(self, file_identifier, n_rows: int = None) -> pd.DataFrame:
        """
        Loads data collection results from one file of a chosen network with
        given vehicle input  and controlled vehicle percentage.

        :param file_identifier: This can be either a integer indicating
         the simulation number or the file name directly
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :return: pandas dataframe with the data
        """

        data = super().load_data(file_identifier, n_rows)
        # We remove columns with information specific to a single vehicle
        # type. We only want the (ALL) columns
        VissimDataReader._keep_only_aggregated_data(data)
        # Include flow
        time_interval = data["time_interval"].iloc[0]
        data["flow"] = self._compute_flow(data["vehicle_count"], time_interval)
        return data

    @staticmethod
    def _compute_flow(vehicle_count: pd.Series, measurement_interval: str
                      ) -> pd.Series:
        interval_start, _, interval_end = measurement_interval.partition("-")
        measurement_period = int(interval_end) - int(interval_start)
        seconds_in_hour = 3600
        return seconds_in_hour / measurement_period * vehicle_count


class LinkEvaluationReader(AggregatedDataReader):
    """Reads data generated from link evaluation measurements in VISSIM"""

    _file_format = ".att"
    _separator = ";"
    _data_identifier = "_Link Segment Results"
    _header_identifier = "$LINKEVALSEGMENTEVALUATION"
    _header_map = {
        "SIMRUN": "simulation_number", "TIMEINT": "time_interval",
        "LINKEVALSEGMENT": "link_segment_number", "DENSITY": "density",
        "DELAYREL": "delay_relative", "SPEED": "average_speed",
        "VOLUME": "volume"
    }
    _data_to_ignore = "emissions"

    def __init__(self, scenario_name):
        VissimDataReader.__init__(self, scenario_name,
                                  self._file_format, self._separator,
                                  self._data_identifier,
                                  self._header_identifier, self._header_map)

    def load_data(self, file_identifier, n_rows: int = None) -> pd.DataFrame:
        """
        Loads link evaluation outputs from one file of a chosen network with
        given vehicle input  and controlled vehicle percentage.

        :param file_identifier: This can be either a integer indicating
         the simulation number or the file name directly
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :return: pandas dataframe with the data
        """

        data = super().load_data(file_identifier, n_rows)
        # Some column names contain (ALL). We can remove that information
        # We remove columns with information specific to a single vehicle
        # type. We only want the (ALL) columns
        VissimDataReader._keep_only_aggregated_data(data)
        # Drop all the emissions columns
        cols_to_be_dropped = [name for name in data.columns
                              if name.startswith(self._data_to_ignore)]
        data.drop(columns=cols_to_be_dropped, inplace=True)
        data["delay_relative"] = data["delay_relative"].str.rstrip(
            " %").astype(float)
        link_information = data["link_segment_number"].str.split(
            "-", expand=True).astype(int)
        data["link_number"] = link_information.iloc[:, 0]
        data["segment_length"] = (link_information.iloc[:, -1]
                                  - link_information.iloc[:, -2])
        data["link_segment"] = link_information.iloc[:, -2].rank(
            method="dense").astype(int)
        # If data was not exported per lane, we set all lanes to zero
        data["lane"] = (0 if link_information.shape[1] == 3 else
                        link_information.iloc[:, 1])

        return data


class VissimLaneChangeReader(VissimDataReader):
    """Reads lane change data generated by VISSIM"""

    _file_format = ".spw"
    _separator = ";"
    _data_identifier = ""
    _header_identifier = "t; VehNo;"
    _header_map = {
        "t": "time", "VehNo": "veh_id", "v [m/s]": "vx",
        "Link No.": "link", "Lane": "origin_lane", "New Lane": "dest_lane",
        "VF": "lo_id", "v VF [m/s]": "lo_vx",
        "dv VF [m/s]": "lo_delta_vx", "dx VF [m]": "lo_gap",
        "VB": "fo_id", "v VB": "fo_vx",
        "dv VB [m/s]": "fo_delta_vx", "dx VB": "fo_gap",
        "new VF": "ld_id", "v new VF [m/s]": "ld_vx",
        "dv new VF [m/s]": "ld_delta_vx", "dx new VF [m]": "ld_gap",
        "new VB": "fd_id", "v new VB [m/s]": "fd_vx",
        "dv new VB [m/s]": "fd_delta_vx", "dx new VB [m]": "fd_gap",
    }

    def __init__(self, scenario_name):
        VissimDataReader.__init__(self, scenario_name,
                                  self._file_format, self._separator,
                                  self._data_identifier,
                                  self._header_identifier, self._header_map)

    # def load_data(self, file_identifier, n_rows: int = None) -> pd.DataFrame:
    #     data = super().load_data(file_identifier, n_rows)
    #     data["simulation_number"] = file_identifier
    #     return data

    def load_data_from_scenario(
            self, scenario_info: ScenarioInfo, n_rows: int = None
    ) -> pd.DataFrame:
        """
        Loads all the simulation data from the scenario described by the
        parameters.

        :param scenario_info: Simulation scenario parameters
        :param n_rows: Number of rows going to be read from the file.
         Used for debugging purposes.
        :return: pandas dataframe with the data
        """

        min_file_number, max_file_number = (
            self.file_handler.find_min_max_file_number(
                self.data_identifier, self.file_format, scenario_info))
        sim_output = []
        for i in range(min_file_number, max_file_number + 1):
            try:
                new_data = self.load_single_file_from_scenario(
                    i, scenario_info)
                if "simulation_number" not in new_data.columns:
                    new_data["simulation_number"] = i
                sim_output.append(new_data)
            except FileNotFoundError:
                warnings.warn("Tried to load simulations from {} to {}, "
                              "but stopped at {}".
                              format(min_file_number, max_file_number, i))
                break

        return pd.concat(sim_output, ignore_index=True)


class LinkReader(VissimDataReader):
    _file_format = ".att"
    _separator = ";"
    _data_identifier = "_Links"
    _header_identifier = "$LINK"
    _header_map = {
        "NO": "number", "NAME": "name", "NUMLANES": "number_of_lanes",
        "LENGTH2D": "length", "ISCONN": "is_connector",
        "FROMLINK": "from_link", "TOLINK": "to_link"
    }

    def __init__(self, scenario_name):
        VissimDataReader.__init__(self, scenario_name,
                                  self._file_format, self._separator,
                                  self._data_identifier,
                                  self._header_identifier, self._header_map)

    def load_data(self, file_identifier=None, n_rows=None) -> pd.DataFrame:
        link_file_folder = self.file_handler.get_network_file_folder()
        file_name = self._create_file_name()
        full_address = os.path.join(link_file_folder, file_name)
        data = super().load_data(full_address)
        return data

    def load_data_from_scenario(
            self, scenario_info: ScenarioInfo = None,
            n_rows: int = None) -> pd.DataFrame:
        # All scenarios of the same network have the same links
        return self.load_data()


# class VehicleInputReader(VissimDataReader):
#     """Reads files containing simulation vehicle input """
#
#     _file_format = ".att"
#     _separator = ";"
#     _data_identifier = "_Vehicle Inputs"
#     _header_identifier = "$VEHICLEINPUT"
#     _header_map = {"NO": "number", "NAME": "name", "LINK": "link",
#                    "VOLUME": "vehicle_input",
#                    "VEHCOMP": "vehicle_composition"}
#
#     def __init__(self, scenario_name):
#         VissimDataReader.__init__(self, scenario_name,
#                                   self._file_format, self._separator,
#                                   self._data_identifier,
#                                   self._header_identifier, self._header_map)


# class ReducedSpeedAreaReader(VissimDataReader):
#     """Reads data from reduced speed areas used in a VISSIM simulation"""
#
#     _file_format = ".att"
#     _separator = ";"
#     _data_identifier = "_Reduced Speed Areas"
#     _header_identifier = "$REDUCEDSPEEDAREA"
#     _header_map = {
#         "NO": "number", "NAME": "name", "LANE": "lane", "POS": "position",
#         "LENGTH": "length", "TIMEFROM": "time_from", "TIMETO": "time_to",
#         "DESSPEEDDISTR": "speed_limit", "DECEL": "max_approach_deceleration"
#     }
#
#     def __init__(self, scenario_name):
#         VissimDataReader.__init__(self, scenario_name,
#                                   self._file_format, self._separator,
#                                   self._data_identifier,
#                                   self._header_identifier, self._header_map)


class PostProcessedDataReader(DataReader):
    """
    Base class to read safety data extracted from vissim simulations
    """
    file_format = ".csv"
    data_identifier = ""

    def __init__(self, scenario_name: str, data_identifier: str):
        self.file_handler = FileHandler(scenario_name)
        DataReader.__init__(self, scenario_name)
        self.data_identifier = data_identifier

    def load_data(self, file_identifier: str) -> pd.DataFrame:
        """
        Loads data from one file of a chosen network with given
        vehicle input and controlled vehicle percentage

        :param file_identifier: File full path
        :return: pandas dataframe with the data
        """

        # vehicle_type = file_identifier
        # data_folder = self.file_handler.get_vissim_data_folder(
        #     vehicle_type, controlled_vehicles_percentage,
        #     vehicles_per_lane, accepted_risk)
        # network_file = self.file_handler.get_file_name()
        # file_name = (network_file + self.data_identifier + self.file_format)
        # full_address = os.path.join(data_folder, file_name)
        full_address = file_identifier
        try:
            data = pd.read_csv(full_address)
        except OSError:
            # Old format files end with a three-digit number. Let's try to
            # read that before giving up
            network_file_name = self.file_handler.get_file_name()
            data_folder = os.path.dirname(full_address)
            file_name = self._load_file_starting_with_name(network_file_name,
                                                           data_folder)
            full_address = os.path.join(data_folder, file_name)
            data = pd.read_csv(full_address)

        # _add_vehicle_type_columns(data, vehicle_type,
        #                           controlled_vehicles_percentage)
        # _add_vehicle_input_column(data, vehicles_per_lane)
        # _add_risk_column(data, accepted_risk)
        return data

    def load_test_data(
            self, scenario_info: ScenarioInfo) -> pd.DataFrame:
        network_file_name = self.file_handler.get_file_name()
        file_name = (network_file_name + self.data_identifier
                     + self.file_format)
        data_folder = self.file_handler.get_vissim_test_folder()
        full_address = os.path.join(data_folder, file_name)
        data = self.load_data(full_address)
        _add_scenario_info_columns(data, scenario_info)
        return data

    def load_data_from_scenario(
            self, scenario_info: ScenarioInfo) -> pd.DataFrame:
        """
        :param scenario_info: Simulation scenario parameters
        """
        network_file_name = self.file_handler.get_file_name()
        file_name = (network_file_name + self.data_identifier
                     + self.file_format)
        data_folder = self.file_handler.get_vissim_data_folder(scenario_info)
        full_address = os.path.join(data_folder, file_name)
        data = self.load_data(full_address)
        _add_scenario_info_columns(data, scenario_info)
        return data

    def load_data_from_several_scenarios(
            self, scenarios: list[ScenarioInfo]) -> pd.DataFrame:
        """
        :param scenarios: List of simulation parameters for several scenarios
        """
        data_per_scenario = []
        for sc in scenarios:
            data_per_scenario.append(self.load_data_from_scenario(sc))
        data = pd.concat(data_per_scenario, ignore_index=True)
        match_sim_number_to_random_seed(data)
        return data

        # if lane_change_strategies is None
        # or lane_change_strategies[0] is None:
        #     return self.load_data_with_controlled_percentage(
        #         vehicle_percentages, vehicle_input_per_lane, accepted_risks)
        # else:
        #     return self.load_platoon_scenario_data(
        #         vehicle_percentages, vehicle_input_per_lane,
        #         lane_change_strategies, orig_and_dest_lane_speeds)

    def _load_file_starting_with_name(self, network_file, data_folder
                                      ) -> str:
        base_name = network_file + self.data_identifier
        file_with_longer_name = []
        for file in os.listdir(data_folder):
            file_str = os.fsdecode(file)
            if file_str.startswith(base_name):
                file_with_longer_name.append(file_str)
        if len(file_with_longer_name) > 1:
            raise OSError("Too many possible files starting with {} "
                          "at {} found".format(self.data_identifier,
                                               data_folder))
        if len(file_with_longer_name) == 0:
            raise FileNotFoundError("No {} file at {}".format(
                self.data_identifier, data_folder))
        return file_with_longer_name[0]


class SSMDataReader(PostProcessedDataReader):
    """Reads aggregated SSM data obtained after processing vehicle record
    data"""

    _data_identifier = "_SSM Results"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)

    def load_data(self, file_identifier) -> pd.DataFrame:
        """

        :param file_identifier: file full path
        :return: SSM data for the requested simulation
        """
        data = super().load_data(file_identifier)
        # Ensure compatibility with previous naming convention
        data.rename(columns={"exact_risk": "risk"}, inplace=True)
        data.rename(columns={
            "exact_risk_no_lane_change": "risk_no_lane_change"}, inplace=True)
        return data


class RiskyManeuverReader(PostProcessedDataReader):
    _data_identifier = "_Risky Maneuvers"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)


class LaneChangeReader(PostProcessedDataReader):
    _data_identifier = "_Lane Changes"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)

    def load_data(self, file_identifier: str) -> pd.DataFrame:
        data = super().load_data(file_identifier)
        y = "total_risk"
        data["total_lane_change_risk"] = (data[y + "_lo"] + data[y + "_ld"]
                                          + data[y + "_fd"])
        y = "initial_risk"
        data[y] = data[y + "_to_lo"] + data[y + "_to_ld"] + data[y + "_to_fd"]

        # TODO: temporary [Oct 27, 22]. I saved the file path instead of the
        #  simulation number
        if isinstance(data["simulation_number"].iloc[0], str):
            data["simulation_number"] = (
                data["simulation_number"].str.split(".").str[0].str[-3:].
                astype(int))

        return data


class LaneChangeIssuesReader(PostProcessedDataReader):
    _data_identifier = "_Lane Change Issues"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)


class ViolationsReader(PostProcessedDataReader):
    _data_identifier = "_Traffic Light Violations"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)


class DiscomfortReader(PostProcessedDataReader):
    _data_identifier = "_Discomfort"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)


class PlatoonLaneChangeEfficiencyReader(PostProcessedDataReader):
    _data_identifier = "_Platoon Lane Change Efficiency"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)


class PlatoonLaneChangeImpactsReader(PostProcessedDataReader):
    _data_identifier = "_Platoon Lane Change Impacts"

    def __init__(self, scenario_name):
        PostProcessedDataReader.__init__(self, scenario_name,
                                         self._data_identifier)


class NGSIMDataReader:
    """Reads raw vehicle trajectory data from NGSIM scenarios on the US-101"""

    file_extension = ".csv"
    ngsim_dir = ("C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation"
                 "\\NGSIM_original\\")
    location_switch = {"us-101": "US-101-LosAngeles-CA\\us-101-vehicle"
                                 "-trajectory-data"}
    interval_switch = {1: "0750am-0805am", 2: "0805am-0820am",
                       3: "0820am-0835am"}
    ngsim_to_reader_naming = {"Global_Time": "time", "Vehicle_ID": "veh_id",
                              "v_Class": "veh_type", "Local_Y": "x",
                              "v_Vel": "vx", "Local_X": "y",
                              "Preceding": "leader_id", "Lane_ID": "lane",
                              "Space_Hdwy": "delta_x", "v_Length": "length"}
    relevant_columns = {"time", "veh_id", "veh_type", "link", "lane",
                        "x", "vx", "y", "leader_id", "delta_x", "leader_type",
                        "front_x", "front_y", "rear_x", "rear_y", "length",
                        "delta_v", "lane_change"}

    def __init__(self, location):
        # self.interval = 0
        try:
            self.data_dir = os.path.join(self.ngsim_dir,
                                         self.location_switch[location])
            file_name = "trajectories-"
        except KeyError:
            print("{}: KeyError: location {} not defined".
                  format(self.__class__.__name__, location))
            self.data_dir = None
            file_name = None
        self.file_base_name = file_name
        # DataReader.__init__(self, file_name)
        self.data_source = "NGSIM"

    def load_data(self, file_identifier=1) -> pd.DataFrame:

        if file_identifier not in self.interval_switch:
            print("Requested interval not available")
            return pd.DataFrame()

        # self.interval = interval
        file_name = self.file_base_name + self.interval_switch[file_identifier]
        full_address = os.path.join(self.data_dir,
                                    file_name + self.file_extension)
        try:
            with open(full_address, "r") as file:
                data = pd.read_csv(file)
                data.rename(columns=self.ngsim_to_reader_naming, inplace=True)
        except OSError:
            raise ValueError("No NGSIM file with name {}".format(file_name))

        self.select_relevant_columns(data)
        return data

    # def get_simulation_identifier(self):
    #     """Returns the time of the day of the data which was loaded last"""
    #     return self.interval_switch[self.interval]

    @staticmethod
    def select_relevant_columns(data) -> None:
        columns_to_drop = []
        for col in data.columns:
            if col not in NGSIMDataReader.relevant_columns:
                columns_to_drop.append(col)
        data.drop(columns=columns_to_drop, inplace=True)


class SyntheticDataReader:
    file_extension = ".csv"
    data_dir = ("C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation"
                "\\synthetic_data\\")
    synthetic_sim_name = "synthetic_data"

    def __init__(self):
        self.sim_number = 0
        self.column_names = ["time", "veh_id", "veh_type", "link", "lane", "x",
                             "vx", "y", "leader_id", "delta_x"]
        self.data_source = "synthetic"

    def load_data(self) -> pd.DataFrame:
        file_name = "synthetic"
        full_address = os.path.join(self.data_dir,
                                    file_name + self.file_extension)
        with open(full_address, "r") as file:
            data = pd.read_csv(file)
        NGSIMDataReader.select_relevant_columns(data)
        return data

    def load_test_data(self) -> pd.DataFrame:
        return self.load_data()


class TrafficLightSourceReader:
    _file_extension = ".csv"
    _data_identifier = "_source_times"

    def __init__(self, scenario_name: str):
        file_handler = FileHandler(scenario_name)
        file_name = (file_handler.get_file_name()
                     + self._data_identifier + self._file_extension)
        self._file_address = os.path.join(
            file_handler.get_network_file_folder(), file_name)

    def load_data(self) -> pd.DataFrame:
        tl_data = pd.read_csv(self._file_address)
        if "starts_red" not in tl_data.columns:
            tl_data["starts_red"] = True
        tl_data["cycle_time"] = (
                tl_data["red duration"]
                + tl_data["green duration"]
                + tl_data["amber duration"])
        return tl_data


class SignalControllerFileReader:
    _file_extension = ".sig"

    def __init__(self, scenario_name):
        self.scenario_name = scenario_name
        self.file_handler = FileHandler(scenario_name)

    def load_data(self, file_identifier: int) -> ET.ElementTree:
        """

        :param file_identifier: Indicates the id of the signal controller
        :return:
        """
        file_address = self.file_handler.get_network_file_folder()
        file_name = self.file_handler.get_file_name()
        full_address = os.path.join(file_address,
                                    file_name + str(file_identifier)
                                    + self._file_extension)
        try:
            with open(full_address, "r") as file:
                return ET.parse(file)
        except OSError:
            raise ValueError("File {} not found".format(full_address))


class MovesDataReader(DataReader):
    """ Class to read from Excel files generated by MOVES """

    _file_extension = ".xls"

    def __init__(self, scenario_name: str, data_identifier: str,
                 sheet_name: str):
        DataReader.__init__(self, scenario_name)
        self.data_identifier = data_identifier
        self.sheet_name = sheet_name
        self.file_handler = FileHandler(scenario_name)

    def load_data(self, file_identifier=None) -> pd.DataFrame:
        folder = self.file_handler.get_moves_default_data_folder()
        # file_name = (self.file_handler.get_file_name()
        #              + "_MOVES_" + self.data_identifier
        #              + self._file_extension)
        file_name = self.data_identifier + self._file_extension
        full_address = os.path.join(folder, file_name)
        if file_identifier is None:
            data = pd.read_excel(full_address, sheet_name=self.sheet_name)
        else:
            data = pd.read_excel(full_address, sheet_name=file_identifier)
        return data

    def load_data_from_several_scenarios(
            self, scenarios: list[ScenarioInfo]) -> pd.DataFrame:
        # TODO
        return pd.DataFrame()
    # def get_data_from_all_sheets(self) -> dict[str, pd.DataFrame]:
    #     folder = self.file_handler.get_moves_data_folder()
    #     file_name = (self.file_handler.get_file_name()
    #                  + "_MOVES_" +
    #                  self.data_identifier
    #                  + self._file_extension)
    #     full_address = os.path.join(folder, file_name)
    #     data = pd.read_excel(full_address, sheet_name=None)
    #     return data


class MovesLinkReader(MovesDataReader):

    _data_identifier = "links"
    _sheet_name = "link"

    def __init__(self, scenario_name: str):
        MovesDataReader.__init__(self, scenario_name, self._data_identifier,
                                 self._sheet_name)

    def get_count_id(self) -> int:
        data = self.load_data("County")
        return data["countyID"].iloc[0]

    def get_zone_id(self) -> int:
        data = self.load_data("Zone")
        return data["zoneID"].iloc[0]

    def get_road_id(self) -> int:
        data = self.load_data("RoadType")
        return data.loc[data["roadDesc"] == "Urban Restricted Access",
                        "roadTypeID"].iloc[0]

    def get_off_road_id(self) -> int:
        data = self.load_data("RoadType")
        return data.loc[data["roadDesc"] == "Off-Network",
                        "roadTypeID"].iloc[0]


class MovesLinkSourceReader(MovesDataReader):

    _data_identifier = "linksource"
    _sheet_name = "linkSourceTypeHour"

    def __init__(self, scenario_name: str):
        MovesDataReader.__init__(self, scenario_name, self._data_identifier,
                                 self._sheet_name)

    def get_passenger_vehicle_id(self) -> int:
        data = self.load_data("SourceUseType")
        return data.loc[data["sourceTypeName"] == "Passenger Car",
                        "sourceTypeID"].iloc[0]


class MOVESDatabaseReader (DataReader):
    user = "moves"
    port = get_moves_database_port()
    hostname = "127.0.0.1"
    _vehicle_type_str_map = {
        VehicleType.HDV: "hdv",
        VehicleType.ACC: "acc",
        VehicleType.AUTONOMOUS: "av",
        VehicleType.CONNECTED: "cav",
        VehicleType.CONNECTED_NO_LANE_CHANGE: "cav",
        VehicleType.PLATOON: "platoon",
        VehicleType.VIRDI: "virdi"
    }

    def __init__(self, scenario_name: str):
        DataReader.__init__(self, scenario_name)
        with open("db_password.txt", "r") as f:
            self.password = f.readline()
        self.file_handler = FileHandler(scenario_name)

    def load_data(self, scenario: ScenarioInfo) -> pd.DataFrame:
        database_name = self._get_database_name(scenario)
        output_database = "_".join([database_name, "out"])
        input_database = "_".join([database_name, "in"])
        data = self._load_pollutants(output_database)
        self._add_volume_data(input_database, data)
        _add_scenario_info_columns(data, scenario)
        data["emission_per_volume"] = data["emission"] / data["volume"]
        return data

    def _get_database_name(self, scenario: ScenarioInfo) -> str:
        if scenario.platoon_lane_change_strategy is None:
            # Single vehicle *safe* lane change maneuvers
            # Sample name: highway_in_and_out_hdv_6000
            vehicle_percentages = scenario.vehicle_percentages
            vehicles_per_lane = scenario.vehicles_per_lane
            temp = []
            for vt, p in vehicle_percentages.items():
                p_str = (str(p) + "_") if p < 100 else ""
                if p > 0:
                    temp.append(p_str + self._vehicle_type_str_map[vt])
            if not temp:
                temp.append(self._vehicle_type_str_map[VehicleType.HDV])
            vt_str = "_".join(sorted(temp))
            return "_".join([self.file_handler.get_file_name(), vt_str,
                             str(3 * vehicles_per_lane)])
        else:
            # Platoon lane changes
            # Name format:
            # platoon_discretionary_lane_change_[hdv or cav]
            # + [strategy]_[x]_dest_speed
            hdv = VehicleType.HDV
            veh_percentages = scenario.vehicle_percentages
            if len(veh_percentages) > 1:
                raise ValueError("MOVES reader not ready for platoon scenarios "
                                 "with mixed vehicles.")
            veh_type_str = self._vehicle_type_str_map[
                list(veh_percentages.keys())[0]]
            strategy_str = (
                scenario.platoon_lane_change_strategy.get_print_name())
            speed_str = str(scenario.orig_and_dest_lane_speeds[1])
            return "_".join([self.file_handler.get_file_name(), veh_type_str,
                             strategy_str, speed_str, "dest_speed"])

    def load_data_from_several_scenarios(
            self, scenarios: list[ScenarioInfo]
    ) -> pd.DataFrame:
        data_per_folder = []
        for sc in scenarios:
            data_per_folder.append(self.load_data(sc))
        data = pd.concat(data_per_folder, ignore_index=True)
        return data

    def _load_pollutants(self, output_database: str) -> pd.DataFrame:
        conn = mariadb.connect(
            user=self.user,
            password=self.password,
            host=self.hostname,
            port=self.port,
            database=output_database
        )
        cur = conn.cursor()
        # We assume the latest run contains the correct results
        max_run_id_query = ("(SELECT MAX(MOVESRunID) FROM "
                            + output_database + ".movesoutput)")
        sql_query = ("SELECT roadTypeID, pollutantID, sum(emissionQuant) "
                     "FROM " + output_database + ".movesoutput "
                     "WHERE MOVESRunID = " + max_run_id_query + " "
                     "GROUP BY roadTypeID, pollutantID")
        cur.execute(sql_query)

        data = {"pollutant_id": [], "emission": [], "road_type": []}
        for roadTypeID, pollutantID, emmissionQuant in cur:
            data["pollutant_id"].append(pollutantID)
            data["emission"].append(emmissionQuant)
            data["road_type"].append(roadTypeID)
        conn.close()
        return pd.DataFrame(data=data)

    def _add_volume_data(self, input_database: str, data: pd.DataFrame) -> None:
        conn = mariadb.connect(
            user=self.user,
            password=self.password,
            host=self.hostname,
            port=self.port,
            database=input_database
        )
        cur = conn.cursor()
        volume_query = ("SELECT roadTypeID, sum(linkVolume) "
                        "FROM " + input_database + ".link "
                                                   "GROUP BY roadTypeID")
        cur.execute(volume_query)
        data["volume"] = 0
        for roadTypeID, linkVolume in cur:
            data.loc[data["road_type"] == roadTypeID, "volume"] = linkVolume
        conn.close()
        data.drop(data[data["volume"] == 0].index, inplace=True)

