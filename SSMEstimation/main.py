from typing import List, Dict

import matplotlib.pyplot as plt

import data_writer
import file_handling
import moves_file_handling
import post_processing
import result_analysis
import vehicle
from vehicle import VehicleType, Vehicle, PlatoonLaneChangeStrategy
from vissim_interface import VissimInterface


def run_i170_scenario(save_results=False):
    network_file = "I710-MultiSec-3mi"
    sim_resolution = 5  # No. of simulation steps per second
    simulation_time = 3600  # 5400  # 4000
    vi = VissimInterface()
    if not vi.load_simulation(network_file):
        return
    vi.set_evaluation_options(save_vehicle_record=save_results,
                              save_ssam_file=False,
                              activate_data_collections=save_results,
                              activate_link_evaluation=save_results,
                              save_lane_changes=save_results)
    idx_scenario = 2  # 0: No block, 1: All time block, 2: Temporary block
    demands = [5500, ]
    # Vehicle Composition ID
    # 1: 10% Trucks
    # demandComposition = 2

    for demand in demands:
        random_seed = 1
        sim_params = {'SimPeriod': simulation_time, 'SimRes': sim_resolution,
                      'UseMaxSimSpeed': True, 'RandSeed': random_seed}
        vi.set_simulation_parameters(sim_params)
        vi.run_i710_simulation(idx_scenario, demand)


def test_risk_computation():
    data_source = 'synthetic'
    follower_v0 = 20
    leader_v0 = 10
    follower_type = Vehicle.VISSIM_CAR_ID
    leader_type = Vehicle.VISSIM_CAR_ID
    is_lane_changing = True
    veh_data = data_writer.SyntheticDataWriter.create_single_veh_data(
        follower_v0, leader_v0, is_lane_changing, follower_type, leader_type)
    post_processing.post_process_data(data_source, veh_data)
    ssm_estimator = post_processing.SSMEstimator(veh_data)
    ssm_estimator.include_ssms(['risk'])
    veh1 = veh_data[veh_data['veh_id'] == 1]

    # Numerical risk computation
    g0 = veh1['delta_x'].to_numpy()
    tc, severity = vehicle.find_collision_time_and_severity(
        g0, follower_v0, leader_v0, is_lane_changing,
        follower_type, leader_type)

    plt.plot(g0, veh1['risk'], 'b-', label='risk')
    plt.plot(g0, severity, 'r--', label='numerical')

    plt.legend()
    plt.show()


def run_all_safe_lane_change_scenarios():
    # Scenario definition
    scenario_name = 'in_and_out_safe'
    vehicle_type = [
        VehicleType.VIRDI,
        VehicleType.CONNECTED,
        # VehicleType.AUTONOMOUS,
        # VehicleType.ACC
    ]

    percentages = [100]  # [i for i in range(0, 101, 100)]
    full_penetration = create_vehicle_percentages_dictionary(
        vehicle_type, percentages, 1)
    # varied_cav_penetration = create_vehicle_percentages_dictionary(
    #     [VehicleType.CONNECTED], [i for i in range(0, 76, 25)], 1)
    inputs_per_lane = [1000, 2000]

    full_penetration_scenarios = file_handling.create_multiple_scenarios(
        full_penetration, inputs_per_lane
    )
    # varied_penetration_scenarios = file_handling.create_multiple_scenarios(
    #     varied_cav_penetration, inputs_per_lane
    # )
    # all_scenarios = [full_penetration_scenarios, varied_penetration_scenarios]

    # Running
    # vi = VissimInterface()
    # vi.load_simulation(scenario_name)
    # vi.run_multiple_scenarios(full_penetration_scenarios)
    # vi.close_vissim()

    # Post-processing
    post_processing.create_summary_with_risks(
        scenario_name, full_penetration_scenarios)
    # for ipl in inputs_per_lane:
    #     post_processing.get_individual_vehicle_trajectories_to_moves(
    #         scenario_name, ipl, sp, 0)

    # Transfer files to the cloud
    file_handler = file_handling.FileHandler(scenario_name)
    try:
        file_handler.export_multiple_results_to_cloud(
            full_penetration_scenarios)
    except FileNotFoundError:
        print("Couldn't copy files to shared folder.")
        # continue


def run_comparison_method():
    scenario_name = 'in_and_out_safe'
    inputs_per_lane = [1000, 2000]

    # Running
    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    for vt in [VehicleType.VIRDI,
               # VehicleType.AUTONOMOUS, VehicleType.ACC
               ]:
        scenarios = file_handling.create_multiple_scenarios(
            [{vt: 100}], inputs_per_lane)
        vi.run_multiple_scenarios(scenarios)
        post_processing.create_summary_with_risks(
            scenario_name, scenarios)
    vi.close_vissim()

    # Post-processing

    # for ipl in inputs_per_lane:
    #     post_processing.get_individual_vehicle_trajectories_to_moves(
    #         scenario_name, ipl, sp, 0)

    # Transfer files to the cloud
    file_handler = file_handling.FileHandler(scenario_name)
    try:
        file_handler.export_multiple_results_to_cloud(
            scenarios)
    except FileNotFoundError:
        print("Couldn't copy files to shared folder.")


def run_platoon_scenarios():
    scenario_name = 'platoon_discretionary_lane_change'
    strategies = [
        PlatoonLaneChangeStrategy.single_body_platoon,
        PlatoonLaneChangeStrategy.leader_first,
        PlatoonLaneChangeStrategy.last_vehicle_first,
        PlatoonLaneChangeStrategy.leader_first_and_reverse
    ]
    vehicle_percentages = [{VehicleType.CONNECTED_NO_LANE_CHANGE: 100}]
    platoon_speed = 90
    main_road_speeds = ['same']
    orig_and_dest_lane_speeds = [(platoon_speed, s) for s in main_road_speeds]
    vehicle_inputs = [i for i in range(500, 2501, 1000)]

    scenarios = file_handling.create_multiple_scenarios(
        vehicle_percentages, vehicle_inputs, lane_change_strategies=strategies,
        orig_and_dest_lane_speeds=orig_and_dest_lane_speeds)

    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    vi.run_multiple_platoon_lane_change_scenarios(scenarios,
                                                  runs_per_scenario=3)
    vi.close_vissim()


def plot_acc_av_and_cav_results(save_results=False):
    scenario_name = 'in_and_out_safe'
    vehicle_types = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
        # VehicleType.VIRDI,
    ]
    percentage = [0, 100]
    veh_inputs = [1000, 2000]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_types, percentage, 1)
    # Our own scenarios
    scenarios = file_handling.create_multiple_scenarios(
        simulation_percentages, veh_inputs, accepted_risks=[0])
    # The comparison scenarios
    scenarios.extend(file_handling.create_multiple_scenarios(
        [{VehicleType.VIRDI: 100}], veh_inputs))
    result_analyzer = result_analysis.ResultAnalyzer(scenario_name,
                                                     save_results)
    result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
        scenarios, warmup_time=10)
    # result_analyzer.plot_link_evaluation_box_plot_vs_controlled_percentage(
    #     'volume', scenarios, warmup_time=10)
    # result_analyzer.plot_risk_histograms('total_risk', scenarios, min_risk=1)
    # result_analyzer.plot_risk_histograms('total_lane_change_risk', scenarios,
    #                                      min_risk=1)
    # result_analyzer.plot_fd_discomfort(scenarios)
    # result_analyzer.plot_emission_heatmap(scenarios)


def plot_comparison_scenario(save_results=False):
    scenario_name = 'in_and_out_safe'
    vehicle_types = [
        VehicleType.VIRDI,
        VehicleType.CONNECTED
    ]
    percentage = [0, 100]
    veh_inputs = [1000, 2000]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_types, percentage, 1)
    scenarios = file_handling.create_multiple_scenarios(
        simulation_percentages, veh_inputs)
    result_analyzer = result_analysis.ResultAnalyzer(scenario_name,
                                                     save_results)
    result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
        scenarios, warmup_time=10)
    result_analyzer.plot_risk_histograms('total_risk', scenarios, min_risk=1)
    result_analyzer.plot_risk_histograms('total_lane_change_risk', scenarios,
                                         min_risk=1)
    result_analyzer.plot_fd_discomfort(scenarios)


def plot_cav_varying_percentage_results(save_results=False):
    network_name = 'in_and_out'
    vehicle_types = [VehicleType.CONNECTED]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [1000, 2000]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_types, percentages, 1)
    scenarios = file_handling.create_multiple_scenarios(simulation_percentages,
                                                        veh_inputs)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.get_flow_and_risk_plots(scenarios)


def plot_traffic_lights_results(save_results=False):
    network_name = 'traffic_lights'
    vehicle_types = [VehicleType.TRAFFIC_LIGHT_ACC,
                     VehicleType.TRAFFIC_LIGHT_CACC]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [500, 1000]
    percentages_per_vehicle_type = create_vehicle_percentages_dictionary(
        vehicle_types, percentages, 1)
    scenarios = file_handling.create_multiple_scenarios(
        percentages_per_vehicle_type, veh_inputs)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
        scenarios, warmup_time=10)

    result_analyzer.accel_vs_time_for_different_vehicle_pairs()
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        'vehicle_count', scenarios, 10)
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        'barrier_function_risk', scenarios, 10)
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        'discomfort', scenarios, 10)
    result_analyzer.plot_violations_heatmap(scenarios, 10)


def all_plots_for_scenarios_with_risk(
        network_name: str, scenarios: List[file_handling.ScenarioInfo],
        save_fig=False):
    ra = result_analysis.ResultAnalyzer(network_name,
                                        should_save_fig=save_fig)
    ra.plot_grid_of_lane_change_risk_histograms('total_lane_change_risk',
                                                scenarios)
    ra.plot_grid_of_lane_change_risk_histograms('initial_risk',
                                                scenarios)
    all_vehicle_inputs = set(sc.vehicles_per_lane for sc in scenarios)
    for veh_input in all_vehicle_inputs:
        scenario_subset = [sc for sc in scenarios
                           if sc.vehicles_per_lane == veh_input]
        ra.box_plot_y_vs_vehicle_type('total_lane_change_risk',
                                      'accepted_risk', scenario_subset)
        ra.box_plot_y_vs_vehicle_type('initial_risk',
                                      'accepted_risk', scenario_subset)
        ra.box_plot_y_vs_vehicle_type('flow', 'accepted_risk', scenario_subset)
        ra.box_plot_y_vs_vehicle_type('risk', 'accepted_risk', scenario_subset)
        ra.plot_lane_change_count_heatmap(scenarios)
        ra.plot_risk_heatmap('initial_risk', scenarios)
        ra.plot_risk_heatmap('total_lane_change_risk', scenarios)
        ra.plot_total_output_heatmap(scenarios)


def all_plots_for_scenarios_with_risk_and_varying_penetration(
        network_name: str, scenarios: List[file_handling.ScenarioInfo],
        save_fig=False):
    ra = result_analysis.ResultAnalyzer(network_name,
                                        should_save_fig=save_fig)

    ra.box_plot_y_vs_vehicle_type('flow', 'accepted_risk', scenarios)
    ra.plot_heatmap_risk_vs_control('lane_change_count', scenarios)


# TODO: move this to some other file
def create_vehicle_percentages_dictionary(
        vehicle_types: List[VehicleType], percentages: List[int],
        n_vehicle_types: int) -> List[Dict[VehicleType, int]]:
    """
    :param vehicle_types:
    :param percentages:
    :param n_vehicle_types: Must be equal to 1 or 2
    :return: List of dictionaries describing the percentage of each vehicle
     type in the simulation
    """
    percentages_list = []
    if n_vehicle_types == 1:
        for vt in vehicle_types:
            for p in percentages:
                percentages_list.append({vt: p})
            if 0 in percentages:
                percentages.remove(0)
    if n_vehicle_types == 2:
        for p1 in percentages:
            for p2 in percentages:
                if p1 > 0 and p2 > 0 and p1 + p2 <= 100:
                    percentages_list.append({vehicle_types[0]: p1,
                                             vehicle_types[1]: p2})
    return percentages_list


def main():
    # =============== Scenario Definition =============== #
    scenario_name = 'platoon_discretionary_lane_change'
    # scenario_name = 'in_and_out_safe'
    vehicle_type = [
        # VehicleType.ACC,
        # VehicleType.AUTONOMOUS,
        # VehicleType.CONNECTED,
        # VehicleType.VIRDI
    ]

    strategies = [
        PlatoonLaneChangeStrategy.single_body_platoon,
        PlatoonLaneChangeStrategy.leader_first,
        PlatoonLaneChangeStrategy.last_vehicle_first,
        PlatoonLaneChangeStrategy.leader_first_and_reverse
    ]

    # full_penetration = create_vehicle_percentages_dictionary(
    #     vehicle_type, percentages, 1)
    inputs_per_lane = 2500

    # =============== Running =============== #
    other_vehicles = {VehicleType.CONNECTED_NO_LANE_CHANGE: 100}
    lc_scenarios = file_handling.create_multiple_scenarios(
        [other_vehicles], [inputs_per_lane],
        lane_change_strategies=strategies[:1],
        orig_and_dest_lane_speeds=[('90', '90'), ('70', '110'),
                                   ('same', 'faster')],
        # special_case='single_lane_change'
    )
    no_lc_scenario = file_handling.ScenarioInfo(
        other_vehicles, inputs_per_lane,
        orig_and_dest_lane_speeds=(90, 'same'),
        special_case='no_lane_change')

    all_scenarios = [no_lc_scenario]
    all_scenarios.extend(lc_scenarios)

    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    vi.run_platoon_lane_change_scenario(4, 110, 60, 300, 50, 50)
    # vi.run_platoon_scenario_sample(4, scenarios[0], simulation_period=1200,
    #                                number_of_runs=1, first_platoon_time=300,
    #                                platoon_creation_period=120,
    #                                is_fast_mode=True)
    # vi.run_multiple_platoon_lane_change_scenarios(
    #     [no_lc_scenario], runs_per_scenario=3)
    # vi.run_multiple_platoon_lane_change_scenarios(
    #     lc_scenarios, runs_per_scenario=3, is_debugging=True)
    # vi.close_vissim()

    # =============== Post processing =============== #
    # post_processing.create_platoon_lane_change_summary(
    #     scenario_name, lc_scenarios)

    # file_handler = file_handling.FileHandler(scenario_name)
    # file_handler.export_multiple_platoon_results_to_cloud(
    #         vehicle_percentages, vehicle_inputs, strategies,
    #         orig_and_dest_lane_speeds)
    # file_handler.import_multiple_platoon_results_from_cloud(
    #     vehicle_percentages, vehicle_inputs, strategies,
    #     orig_and_dest_lane_speeds)

    # =============== To MOVES =============== #
    # scenarios = file_handling.create_multiple_scenarios(
    #     full_penetration, inputs_per_lane)
    # for sc in scenarios:
    #     moves_file_handling.get_individual_vehicle_trajectories_to_moves(
    #         scenario_name, sc, vehs_per_simulation=3)

    # =============== Check results graphically =============== #
    ra = result_analysis.ResultAnalyzer(scenario_name, should_save_fig=False,
                                        is_debugging=False)

    # platoon_scenarios = file_handling.create_multiple_scenarios(
    #     [other_vehicles], [veh_input], lane_change_strategies=strategies,
    #     orig_and_dest_lane_speeds=[(90, 'same')])

    # scenarios.append(no_lc_scenario)
    # scenarios.extend(platoon_scenarios)

    before_or_after_lc_point = 'before'
    lanes = 'both'
    segment = 1 if before_or_after_lc_point == 'after' else 2
    all_sims = False
    # ra.plot_flow_box_plot_vs_strategy(scenarios, before_or_after_lc_point,
    #                                   lanes, hue='sensor position',
    #                                   warmup_time=5, aggregation_period=30)
    # ra.plot_flows_vs_time_per_strategy(scenarios, before_or_after_lc_point,
    #                                    lanes, warmup_time=0,
    #                                    aggregation_period=30,
    #                                    use_single_simulation=~all_sims)
    ys = [
        'volume',
        'average_speed',
        # 'density'
          ]
    for y in ys:
        ra.plot_link_data_box_plot_vs_strategy(
            y,  lc_scenarios, before_or_after_lc_point, lanes, segment,
            warmup_time=5, aggregation_period=30)
    #     ra.plot_link_data_vs_time_per_strategy(
    #         y, all_scenarios, before_or_after_lc_point, lanes, segment,
    #         warmup_time=0, aggregation_period=5, use_all_simulations=all_sims)
    # ra.plot_y_vs_platoon_lc_strategy('platoon_maneuver_time', scenarios)

    # for sc in lc_scenarios:
    #     ra.speed_color_map(sc, link=1)


if __name__ == '__main__':
    main()
