import itertools
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Union
from collections import defaultdict
import vehicle


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
    vehicle_percentages: dict[vehicle.VehicleType, int]
    vehicles_per_lane: int
    accepted_risk: Union[int, None] = None
    platoon_lane_change_strategy: vehicle.PlatoonLaneChangeStrategy = None
    orig_and_dest_lane_speeds: tuple[Union[str, int], Union[str, int]] = None
    platoon_size: int = None
    special_case: str = None

    def __str__(self):
        str_list = []
        veh_percent_list = [str(p) + "% " + vt.name.lower()
                            for vt, p in self.vehicle_percentages.items()]
        str_list.append("Vehicles: " + ", ".join(veh_percent_list))
        str_list.append("Input: " + str(self.vehicles_per_lane)
                        + " vehs/lane/hour")
        if self.platoon_lane_change_strategy is not None:
            str_list.append("Platoon LC strat.: "
                            + self.platoon_lane_change_strategy.name.lower())
        if self.orig_and_dest_lane_speeds is not None:
            str_list.append("Orig lane speed "
                            + str(self.orig_and_dest_lane_speeds[0])
                            + ". Dest lane speed: "
                            + str(self.orig_and_dest_lane_speeds[1]))
        if self.platoon_size is not None:
            str_list.append("n_platoon=" + str(self.platoon_size))
        if self.special_case is not None:
            str_list.append("Special case: " + self.special_case)
        return "\n".join(str_list)


def is_all_human(scenario: ScenarioInfo) -> bool:
    return (sum(scenario.vehicle_percentages.values()) == 0
            or (vehicle.VehicleType.HDV in scenario.vehicle_percentages
                and scenario.vehicle_percentages[vehicle.VehicleType.HDV]
                == 100))


def create_vehicle_percentages_dictionary(
        vehicle_types: list[vehicle.VehicleType], percentages: list[int],
        n_vehicle_types: int) -> list[dict[vehicle.VehicleType, int]]:
    """
    :param vehicle_types:
    :param percentages:
    :param n_vehicle_types: Must be equal to 1 or 2
    :return: List of dictionaries describing the percentage of each vehicle
     type in the simulation
    """
    percentages_list = []
    percentages_ = percentages.copy()
    if n_vehicle_types == 1:
        for vt in vehicle_types:
            for p in percentages_:
                percentages_list.append({vt: p})
            if 0 in percentages_:
                percentages_.remove(0)
    if n_vehicle_types == 2:
        for p1 in percentages_:
            for p2 in percentages_:
                if p1 > 0 and p2 > 0 and p1 + p2 <= 100:
                    percentages_list.append({vehicle_types[0]: p1,
                                             vehicle_types[1]: p2})
    return percentages_list


def create_multiple_scenarios(
        vehicle_percentages: Iterable[dict[vehicle.VehicleType, int]],
        vehicle_inputs: Iterable[int],
        accepted_risks: Iterable[int] = None,
        lane_change_strategies:
        Iterable[vehicle.PlatoonLaneChangeStrategy] = None,
        orig_and_dest_lane_speeds: Iterable[tuple[Union[str, int],
                                            Union[str, int]]] = None,
        platoon_size: Iterable[int] = None,
        special_cases: Iterable[str] = None) -> list[ScenarioInfo]:
    if accepted_risks is None:
        accepted_risks = [None]
        if lane_change_strategies is None:  # not a platoon scenario
            print("[WARNING] Using empty list of accepted risks instead of "
                  "[None] might prevent the readers from working.")
    if lane_change_strategies is None:
        lane_change_strategies = [None]
    if orig_and_dest_lane_speeds is None:
        orig_and_dest_lane_speeds = [None]
    if platoon_size is None:
        platoon_size = [None]
    if special_cases is None:
        special_cases = [None]
    scenarios = []
    for vp, vi, ar, st, speeds, sizes, case in itertools.product(
            vehicle_percentages, vehicle_inputs, accepted_risks,
            lane_change_strategies, orig_and_dest_lane_speeds, platoon_size,
            special_cases):
        if sum(vp.values()) == 0 and ar is not None and ar > 0:
            continue
        scenarios.append(ScenarioInfo(vp, vi, ar, st, speeds, sizes, case))
    return scenarios


def filter_scenarios(
        scenarios: Iterable[ScenarioInfo],
        desired_vehicle_percentages: Iterable[dict[vehicle.VehicleType, int]] = None,
        vehicles_per_lane: tuple[Union[int, None],
                                 Union[int, None]] = None,
        accepted_risk: tuple[Union[int, None], Union[int, None]] = None,
        dest_lane_speeds: tuple[Union[int, None], Union[int, None]] = None,
        special_cases: Iterable[str] = None) -> list[ScenarioInfo]:
    """

    Returns a new scenario list containing only scenarios that respect the
    filter conditions
    """
    def filter_min_max(value, min_max_values,):
        if min_max_values is not None:
            min_value, max_value = vehicles_per_lane
            if min_value is not None and value < min_value:
                will_include = False
            if max_vpl is not None and scenario_vpl > max_vpl:
                will_include = False

    new_list = []
    for sc in scenarios:
        will_include = True
        if not (desired_vehicle_percentages is None
                or sc.vehicle_percentages in desired_vehicle_percentages):
            will_include = False
        if vehicles_per_lane is not None:
            min_vpl, max_vpl = vehicles_per_lane
            scenario_vpl = sc.vehicles_per_lane
            if min_vpl is not None and scenario_vpl < min_vpl:
                will_include = False
            if max_vpl is not None and scenario_vpl > max_vpl:
                will_include = False
        if accepted_risk is not None:
            min_ar, max_ar = accepted_risk
            scenario_ar = sc.accepted_risk
            if min_ar is not None and scenario_ar < min_ar:
                will_include = False
            if max_ar is not None and scenario_ar > max_ar:
                will_include = False

    return new_list


def split_scenario_by(scenarios: Iterable[ScenarioInfo], attribute: str
                      ) -> dict[Any, list[ScenarioInfo]]:
    """
    Splits a list of scenarios in subsets based on the value of the attribute.
    :returns: Dictionary where keys are unique values of the attribute and
     values are lists of scenarios.
    """
    def attribute_value_to_str(value):
        if value is None:
            return "None"
        if attribute == "vehicle_percentages":
            return vehicle_percentage_dict_to_string(value)
        elif attribute == "platoon_lane_change_strategy":
            return value.get_print_name()
        else:
            return value

    subsets = defaultdict(list)
    for sc in scenarios:
        subsets[attribute_value_to_str(getattr(sc, attribute))].append(sc)
    return subsets


def vehicle_percentage_dict_to_string(vp_dict: dict[vehicle.VehicleType, int]
                                      ) -> str:
    if sum(vp_dict.values()) == 0:
        return "100% HDV"
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + "% "
                       + veh_type.get_print_name())
    return " ".join(sorted(ret_str))


all_platoon_simulation_configurations: dict[str, Iterable] = {
    "strategies": [
        vehicle.PlatoonLaneChangeStrategy.single_body_platoon,
        vehicle.PlatoonLaneChangeStrategy.last_vehicle_first,
        vehicle.PlatoonLaneChangeStrategy.leader_first_and_reverse,
        vehicle.PlatoonLaneChangeStrategy.graph_min_accel,
        vehicle.PlatoonLaneChangeStrategy.graph_min_time
    ],
    "orig_and_dest_lane_speeds": [("70", "50"), ("70", "70"), ("70", "90")],
    "platoon_size": [2, 3, 4, 5],
    "vehicles_per_lane": [500, 1000, 1500]
}


def get_platoon_lane_change_scenarios(
        select: str = None, with_hdv: bool = False,
        include_no_lane_change: bool = False) -> list[ScenarioInfo]:
    """

    :param select: "all" returns all the lane change scenarios;
     "dest_lane_speed" returns the 15 scenarios with varying relative speed;
     "vehicles_per_lane" returns the 60 scenarios with varying vehicle input;
     "platoon_size" returns the 60 scenarios with varying number of vehicles
     in the platoon
    :param with_hdv: If true, non-platoon vehicles are human driven
    :param include_no_lane_change: if True also includes the no lane change
     scenario in the list
    :returns: list with the requested scenarios
    """
    if with_hdv:
        other_vehicles = [{vehicle.VehicleType.HDV: 100}]
    else:
        other_vehicles = [{vehicle.VehicleType.CONNECTED: 100}]
    strategies = all_platoon_simulation_configurations["strategies"]
    orig_and_dest_lane_speeds = all_platoon_simulation_configurations[
        "orig_and_dest_lane_speeds"]
    platoon_size = all_platoon_simulation_configurations["platoon_size"]
    vehicles_per_lane = all_platoon_simulation_configurations[
        "vehicles_per_lane"]
    scenarios = []
    if select == "all" or select == "dest_lane_speed":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [2700],
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            platoon_size=[4]))
    if select == "all" or select == "vehicles_per_lane":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, vehicles_per_lane,
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            platoon_size=[4]))
    if select == "all" or select == "platoon_size":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [2700],
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            platoon_size=platoon_size
        ))
    if include_no_lane_change:
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [2700],
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            special_cases=["no_lane_change"]))
    if len(scenarios) == 0:
        raise ValueError("No scenarios selected. Probably parameter 'select' "
                         "is incorrect.")
    return scenarios


def get_lane_change_scenarios_graph_paper(
        strategies: Iterable[vehicle.PlatoonLaneChangeStrategy] = None,
        orig_and_dest_lane_speeds: Iterable[tuple[int, int]] = None,
        platoon_size: Iterable[int] = None,
        vehicles_per_lane: Iterable[int] = None):
    other_vehicles = [{vehicle.VehicleType.HDV: 100}]
    if strategies is None:
        strategies = all_platoon_simulation_configurations["strategies"]
    if orig_and_dest_lane_speeds is None:
        orig_and_dest_lane_speeds = all_platoon_simulation_configurations[
            "orig_and_dest_lane_speeds"]
    if platoon_size is None:
        platoon_size = all_platoon_simulation_configurations["platoon_size"]
    if vehicles_per_lane is None:
        vehicles_per_lane = all_platoon_simulation_configurations[
            "vehicles_per_lane"]
    scenarios = create_multiple_scenarios(
        other_vehicles, vehicles_per_lane, lane_change_strategies=strategies,
        orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
        platoon_size=platoon_size)
    return scenarios
