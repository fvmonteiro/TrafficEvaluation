import itertools
from dataclasses import dataclass
from typing import Any, Union
from collections import defaultdict

from vehicle import VehicleType, PlatoonLaneChangeStrategy, \
    vehicle_type_to_print_name_map, strategy_to_print_name_map


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
    vehicle_percentages: dict[VehicleType, int]
    vehicles_per_lane: int
    accepted_risk: Union[int, None] = None
    platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None
    orig_and_dest_lane_speeds: tuple[Union[str, int], Union[str, int]] = None
    platoon_size: int = None
    special_case: str = None


def is_all_human(scenario: ScenarioInfo) -> bool:
    return (sum(scenario.vehicle_percentages.values()) == 0
            or (VehicleType.HDV in scenario.vehicle_percentages
                and scenario.vehicle_percentages[VehicleType.HDV] == 100))


def create_vehicle_percentages_dictionary(
        vehicle_types: list[VehicleType], percentages: list[int],
        n_vehicle_types: int) -> list[dict[VehicleType, int]]:
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
        vehicle_percentages: list[dict[VehicleType, int]],
        vehicle_inputs: list[int],
        accepted_risks: list[int] = None,
        lane_change_strategies: list[PlatoonLaneChangeStrategy] = None,
        orig_and_dest_lane_speeds: list[tuple[Union[str, int],
                                              Union[str, int]]] = None,
        special_cases: list[str] = None) -> list[ScenarioInfo]:
    if accepted_risks is None:
        accepted_risks = [None]
        if lane_change_strategies is None:  # not a platoon scenario
            print("[WARNING] Using empty list of accepted risks instead of "
                  "[None] might prevent the readers from working.")
    if lane_change_strategies is None:
        lane_change_strategies = [None]
    if orig_and_dest_lane_speeds is None:
        orig_and_dest_lane_speeds = [None]
    if special_cases is None:
        special_cases = [None]
    scenarios = []
    for vp, vi, ar, st, sp, case in itertools.product(
            vehicle_percentages, vehicle_inputs, accepted_risks,
            lane_change_strategies, orig_and_dest_lane_speeds, special_cases):
        if sum(vp.values()) == 0 and ar is not None and ar > 0:
            continue
        scenarios.append(ScenarioInfo(vp, vi, ar, st, sp, case))
    return scenarios


def print_scenario(scenario: ScenarioInfo) -> str:
    str_list = []
    veh_percent_list = [str(p) + "% " + vt.name.lower()
                        for vt, p in scenario.vehicle_percentages.items()]
    str_list.append("Vehicles: " + ", ".join(veh_percent_list))
    str_list.append("Input: " + str(scenario.vehicles_per_lane)
                    + " vehs/lane/hour")
    if scenario.accepted_risk is not None:
        str_list.append("Accepted risk: " + str(scenario.accepted_risk))
    if scenario.platoon_lane_change_strategy is not None:
        str_list.append("Platoon LC strat.: "
                        + scenario.platoon_lane_change_strategy.name.lower())
    if scenario.orig_and_dest_lane_speeds is not None:
        str_list.append("Orig lane speed "
                        + str(scenario.orig_and_dest_lane_speeds[0])
                        + ". Dest lane speed: "
                        + str(scenario.orig_and_dest_lane_speeds[1]))
    if scenario.special_case is not None:
        str_list.append("Special case: " + scenario.special_case)
    return "\n".join(str_list)


def filter_scenarios(
        scenarios: list[ScenarioInfo],
        desired_vehicle_percentages: list[dict[VehicleType, int]] = None,
        vehicles_per_lane: tuple[Union[int, None],
                                 Union[int, None]] = None,
        accepted_risk: tuple[Union[int, None], Union[int, None]] = None,
        dest_lane_speeds: tuple[Union[int, None], Union[int, None]] = None,
        special_cases: list[str] = None) -> list[ScenarioInfo]:
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


def split_scenario_by(scenarios: list[ScenarioInfo], attribute: str) \
        -> dict[Any, list[ScenarioInfo]]:
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
            return strategy_to_print_name_map[value]
        else:
            return value

    subsets = defaultdict(list)
    for sc in scenarios:
        subsets[attribute_value_to_str(getattr(sc, attribute))].append(sc)
    return subsets


def vehicle_percentage_dict_to_string(vp_dict: dict[VehicleType, int]) -> str:
    if sum(vp_dict.values()) == 0:
        return "100% HDV"
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + "% " + vehicle_type_to_print_name_map[veh_type])
    return " ".join(sorted(ret_str))


def get_platoon_lane_change_scenarios(
        select: str = None, with_hdv: bool = False,
        include_no_lane_change: bool = False) -> list[ScenarioInfo]:
    """

    :param select: "all" returns all the 56 lane change scenarios;
     "dest lane speed" returns the 12 scenarios with varying relative speed;
     "vehicles per hour" returns the 24 scenarios with varying vehicle input;
     "platoon_size" returns the 20 scenarios with varying number of vehicles
     in the platoon
    :param with_hdv: If true, non-platoon vehicles are human driven
    :param include_no_lane_change: if True also includes the no lane change
     scenario in the list
    :returns: list with the requested scenarios
    """
    if with_hdv:
        other_vehicles = [{VehicleType.HDV: 100}]
    else:
        other_vehicles = [{VehicleType.CONNECTED_NO_LANE_CHANGE: 100}]
    inputs_per_lane = [2700]
    strategies = [
        PlatoonLaneChangeStrategy.single_body_platoon,
        PlatoonLaneChangeStrategy.leader_first,
        PlatoonLaneChangeStrategy.last_vehicle_first,
        PlatoonLaneChangeStrategy.leader_first_and_reverse
    ]
    scenarios = []
    if select == "all" or select == "dest_lane_speed":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, inputs_per_lane,
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=[("70", "50"),
                                       ("70", "70"),
                                       ("70", "90")],
            special_cases=["single_lane_change"]))
    if select == "all" or select == "vehicles_per_lane":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [i for i in range(500, 3001, 500)],
            # other_vehicles, [1000, 1500, 2000, 3000],
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=[("70", "50"),
                                       ("70", "70"),
                                       ("70", "90")],
            special_cases=["single_lane_change"]))
    if select == "all" or select == "platoon_size":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles,
            vehicle_inputs=inputs_per_lane,
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=[("70", "50"),
                                       ("70", "70"),
                                       ("70", "90")],
            special_cases=[str(i) + "_platoon_vehicles" for i in
                           [2, 3, 7, 9]]
        ))
    if include_no_lane_change:
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, inputs_per_lane,
            orig_and_dest_lane_speeds=[("70", "50"),
                                       ("70", "70"),
                                       ("70", "90")],
            special_cases=["no_lane_change"]))
    if len(scenarios) == 0:
        raise ValueError("No scenarios selected. Probably parameter 'select' "
                         "is incorrect.")
    return scenarios
