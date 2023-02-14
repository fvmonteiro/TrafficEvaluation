import time
from typing import List, Dict
from math import pi

import matplotlib.pyplot as plt
import numpy as np

import data_writer
import file_handling
import post_processing
import result_analysis
import vehicle
from data_writer import SyntheticDataWriter
import readers
from vehicle import VehicleType, Vehicle, PlatoonLaneChangeStrategy
from vissim_interface import VissimInterface


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    sim_params = {'SimPeriod': 200, 'RandSeed': 1}
    vi = VissimInterface()
    if not vi.load_simulation(network_file):
        return
    vi.set_evaluation_options()  # save nothing
    vi.set_simulation_parameters(sim_params)
    vi.run_in_and_out_scenario()
    vi.close_vissim()


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


# def test_safe_gap_computation():
#     # create_and_save_synthetic_data()
#     data_source = 'synthetic'
#     # post_process_and_save('synthetic_data', network_file)
#     data_reader = readers.PostProcessedDataReader_OLD(data_source)
#     data = data_reader.load_data()
#
#     gamma = 1 / 0.8
#     rho = 0.2
#     ssm_estimator = post_processing.SSMEstimator(data)
#     ssm_estimator.include_risk(same_type_gamma=gamma)
#     ssm_estimator.include_estimated_risk(rho=rho, same_type_gamma=gamma)
#     risk_gap = data.loc[data['delta_x'] <= (data['safe_gap'] + 0.02),
#                         'delta_x']
#     risk = data.loc[data['delta_x']
#                     <= (data['safe_gap'] + 0.02), 'exact_risk']
#     estimated_risk_gap = data.loc[data['delta_x'] <= (data['vf_gap'] + 0.02),
#                                   'delta_x']
#     estimated_risk = data.loc[data['delta_x'] <= (data['vf_gap'] + 0.02),
#                               'estimated_risk']
#
#     # Plot
#     fig, ax = plt.subplots()
#     ax.plot(risk_gap, risk)
#     ax.plot(estimated_risk_gap, estimated_risk)
#     plt.show()
#     # save_post_processed_data(ssm_estimator.veh_data,
#     #                          data_reader.data_source, data_reader.file_name)


def run_all_safe_lane_change_scenarios():
    # Scenario definition
    scenario_name = 'in_and_out_safe'
    vehicle_type = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
    ]

    percentages = [i for i in range(0, 101, 100)]
    full_penetration = create_vehicle_percentages_dictionary(
        vehicle_type, percentages, 1)
    varied_cav_penetration = create_vehicle_percentages_dictionary(
        [VehicleType.CONNECTED], [i for i in range(0, 76, 25)], 1)
    inputs_per_lane = [1000, 2000]
    accepted_risks = [0]

    # Running
    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    for simulation_percentages in [full_penetration, varied_cav_penetration]:
        for ipl in inputs_per_lane:
            vi.run_multiple_scenarios(simulation_percentages, [ipl],
                                      accepted_risks)
        vi.close_vissim()

        # Post processing
        for sp in full_penetration:
            print(sp)
            post_processing.create_summary_with_risks(
                scenario_name, [sp], inputs_per_lane, accepted_risks)
            # for ipl in inputs_per_lane:
            #     post_processing.get_individual_vehicle_trajectories_to_moves(
            #         scenario_name, ipl, sp, 0)

        # Transfer files to the cloud
        file_handler = file_handling.FileHandler(scenario_name)
        try:
            file_handler.export_multiple_results_to_cloud(
                simulation_percentages, inputs_per_lane, accepted_risks)
        except FileNotFoundError:
            print("Couldn't copy files to shared folder.")
            continue


def plot_acc_av_and_cav_results(save_results=False):
    scenario_name = 'in_and_out_safe'
    vehicle_types = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED
    ]
    percentage = [0, 100]
    veh_inputs = [1000]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_types, percentage, 1)
    result_analyzer = result_analysis.ResultAnalyzer(scenario_name,
                                                     save_results)
    result_analyzer.get_flow_and_risk_plots(veh_inputs,
                                            simulation_percentages,
                                            )


def plot_cav_varying_percentage_results(save_results=False):
    network_name = 'in_and_out'
    vehicle_types = [VehicleType.CONNECTED]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [1000, 2000]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_types, percentages, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.get_flow_and_risk_plots(veh_inputs, simulation_percentages)


def plot_traffic_lights_results(save_results=False):
    network_name = 'traffic_lights'
    vehicle_types = [VehicleType.TRAFFIC_LIGHT_ACC,
                     VehicleType.TRAFFIC_LIGHT_CACC]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [500, 1000]
    percentages_per_vehicle_type = create_vehicle_percentages_dictionary(
        vehicle_types, percentages, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
        veh_inputs, percentages_per_vehicle_type, warmup_time=10)

    result_analyzer.accel_vs_time_for_different_vehicle_pairs()
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        'vehicle_count', percentages_per_vehicle_type, veh_inputs, 10)
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        'barrier_function_risk', percentages_per_vehicle_type, veh_inputs, 10)
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        'discomfort', percentages_per_vehicle_type, veh_inputs, 10)
    result_analyzer.plot_violations_heatmap(percentages_per_vehicle_type,
                                            veh_inputs, 10)


def all_plots_for_scenarios_with_risk(
        network_name: str, simulation_percentages: List[Dict[VehicleType, int]],
        inputs_per_lane: List[int], accepted_risks: List[int],
        save_fig=False):
    ra = result_analysis.ResultAnalyzer(network_name,
                                        should_save_fig=save_fig)
    # ra.plot_grid_of_risk_histograms('total_risk', simulation_percentages,
    #                                 inputs_per_lane, accepted_risks,
    #                                 min_risk=1)
    # ra.plot_grid_of_lane_change_risk_histograms('total_lane_change_risk',
    #                                             simulation_percentages,
    #                                             inputs_per_lane, accepted_risks)
    # ra.plot_grid_of_lane_change_risk_histograms('initial_risk',
    #                                             simulation_percentages,
    #                                             inputs_per_lane, accepted_risks)
    # for veh_name in ['lo', 'ld', 'fd']:
    #     ra.plot_grid_of_risk_histograms('initial_risk_to_' + veh_name,
    #                                     simulation_percentages,
    #                                     inputs_per_lane, accepted_risks)
    for ipl in inputs_per_lane:
        # ra.box_plot_y_vs_vehicle_type('total_lane_change_risk',
        #                               'accepted_risk', ipl,
        #                               simulation_percentages, accepted_risks)
        # ra.box_plot_y_vs_vehicle_type('initial_risk',
        #                               'accepted_risk', ipl,
        #                               simulation_percentages, accepted_risks)
        ra.box_plot_y_vs_vehicle_type('flow', 'accepted_risk', [ipl],
                                      simulation_percentages, accepted_risks)
        # ra.box_plot_y_vs_vehicle_type('risk', 'accepted_risk', ipl,
        #                               simulation_percentages, accepted_risks)
        # ra.plot_heatmap('lane_change_count', simulation_percentages,
        #                 [ipl], accepted_risks)
        # ra.plot_heatmap('initial_risk', simulation_percentages,
        #                 [ipl], accepted_risks)
        # ra.plot_heatmap('total_lane_change_risk', simulation_percentages,
        #                 [ipl], accepted_risks)
        # ra.plot_heatmap('vehicle_count', simulation_percentages,
        #                 [ipl], accepted_risks, normalize=False)


def all_plots_for_scenarios_with_risk_and_varying_penetration(
        network_name: str, simulation_percentages: List[Dict[VehicleType, int]],
        inputs_per_lane: int, accepted_risks: List[int],
        save_fig=False):
    ra = result_analysis.ResultAnalyzer(network_name,
                                        should_save_fig=save_fig)

    ra.box_plot_y_vs_vehicle_type('flow', 'accepted_risk', [inputs_per_lane],
                                  simulation_percentages, accepted_risks)
    ra.plot_heatmap_risk_vs_control('lane_change_count', simulation_percentages,
                                    [inputs_per_lane], accepted_risks)


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
    # image_folder = "G:\\My Drive\\Safety in Mixed Traffic\\images"

    # =============== Scenario Definition =============== #
    scenario_name = 'platoon_discretionary_lane_change'
    strategies = [
        # PlatoonLaneChangeStrategy.human_driven,
        # PlatoonLaneChangeStrategy.no_strategy,
        PlatoonLaneChangeStrategy.single_body_platoon,
        PlatoonLaneChangeStrategy.leader_first,
        PlatoonLaneChangeStrategy.last_vehicle_first,
        PlatoonLaneChangeStrategy.leader_first_and_reverse
    ]
    vehicle_types = [
        # VehicleType.HUMAN_DRIVEN,
        VehicleType.CONNECTED_NO_LANE_CHANGE
    ]
    # percentages = [100]
    # vehicle_percentages = create_vehicle_percentages_dictionary(
    #     vehicle_types, percentages, 1)
    platoon_speed = 90
    main_road_speeds = ['same']
    orig_and_dest_lane_speeds = [(platoon_speed, s) for s in main_road_speeds]
    vehicle_inputs = [i for i in range(3000, 3001, 500)]

    # =============== Running =============== #
    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    # vi.run_platoon_scenario_sample(
    #     4, PlatoonLaneChangeStrategy.leader_first, 1500,
    #     orig_and_dest_lane_speeds[0], 360, 1, 10, 30)
    vi.run_multiple_platoon_lane_change_scenarios(
        strategies, vehicle_types, orig_and_dest_lane_speeds, vehicle_inputs,
        runs_per_scenario=3)
    vi.close_vissim()

    # =============== Post processing =============== #
    vehicle_percentages = [{vt: 100} for vt in vehicle_types]

    # post_processing.create_platoon_lane_change_summary(
    #     scenario_name, vehicle_percentages, vehicle_inputs, strategies,
    #     orig_and_dest_lane_speeds)

    file_handler = file_handling.FileHandler(scenario_name)
    file_handler.export_multiple_platoon_results_to_cloud(
            vehicle_percentages, vehicle_inputs, strategies,
            orig_and_dest_lane_speeds)
    # file_handler.import_multiple_platoon_results_from_cloud(
    #     vehicle_percentages, vehicle_inputs, strategies,
    #     orig_and_dest_lane_speeds)

    # =============== Check results graphically =============== #
    vehicle_inputs = [i for i in range(0, 3001, 500)]
    ra = result_analysis.ResultAnalyzer(scenario_name, False)
    ra.plot_fundamental_diagram_per_strategy(
            vehicle_percentages[0], vehicle_inputs, warmup_time=5,
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds[0],
            link_segment_number=1, lanes=[1, 2], aggregation_period=30)

    # ra.plot_volume_box_plot_vs_strategy(
    #     vehicle_inputs, vehicle_percentages, strategies,
    #     orig_and_dest_lane_speeds[0], 3, [1, 2], aggregation_period=60)
    # ra.plot_flow_box_plot_vs_strategy(
    #     vehicle_inputs, vehicle_percentages, strategies,
    #     orig_and_dest_lane_speeds[0], [5, 6], aggregation_period=60)
    # ys = ['was_lane_change_completed', 'platoon_maneuver_time',
    #       'travel_time', 'accel_cost', 'stayed_in_platoon']
    # for y in ys:
    #     ra.plot_y_vs_platoon_lc_strategy(y, vehicle_percentages[0],
    #                                      vehicle_inputs, strategies,
    #                                      orig_and_dest_lane_speeds[0])
    #     ra.plot_y_vs_vehicle_input(y, vehicle_percentages[0], vehicle_inputs,
    #                                strategies, orig_and_dest_lane_speeds[0])
    # vehicle_inputs = [i for i in range(500, 2501, 500)]
    # for speed_pair in orig_and_dest_lane_speeds:
    #     ra.plot_fundamental_diagram({VehicleType.HUMAN_DRIVEN: 100},
    #                                 vehicle_inputs,
    #                                 warmup_time=5,
    #                                 platoon_lane_change_strategy=strategies[0],
    #                                 orig_and_dest_lane_speeds=speed_pair
    #                                 )

    # all_plots_for_scenarios_with_risk(scenario_name, simulation_percentages,
    #                                   inputs_per_lane, accepted_risks,
    #                                   save_fig=False)

    # simulation_percentages = varied_cav_penetration
    #
    # ra.plot_risk_histograms('total_risk', simulation_percentages,
    #                         inputs_per_lane, [0], min_risk=1)
    # ra.plot_risk_histograms('total_lane_change_risk', simulation_percentages,
    #                         inputs_per_lane, [0], min_risk=1)
    # ra.plot_flow_box_plot_vs_controlled_percentage(
    #     inputs_per_lane, simulation_percentages, aggregation_period=60)
    # ra.plot_fd_discomfort(simulation_percentages, inputs_per_lane, [0])
    # ra.plot_emission_heatmap(simulation_percentages, inputs_per_lane,
    #                          accepted_risks)
    # ra.plot_risk_heatmap('total_lane_change_risk', simulation_percentages,
    #                      inputs_per_lane, [0])
    # ra.plot_risk_heatmap('total_lane_change_risk', simulation_percentages,
    #                      inputs_per_lane, [0])

    # ra.accel_vs_time_for_different_vehicle_pairs()
    # ra.risk_vs_time_example()
    # ra.find_unfinished_simulations(simulation_percentages, inputs_per_lane)
    # ra.plot_lane_change_count_heatmap(full_penetration, inputs_per_lane,
    #                                   [0])
    # for b in range(4, 8):
    #     ra.plot_discomfort_heatmap(simulation_percentages, inputs_per_lane,
    #                                accepted_risks, max_brake=b)
    # ra.print_summary_of_issues(simulation_percentages, inputs_per_lane,
    #                            accepted_risks)

    # plot_acc_av_and_cav_results(False)
    # plot_cav_varying_percentage_results(False)
    # plot_traffic_lights_results(False)


if __name__ == '__main__':
    main()
