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
from vehicle import VehicleType, Vehicle
from vissim_interface import VissimInterface


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    sim_params = {'SimPeriod': 200, 'RandSeed': 1}
    vi = VissimInterface()
    if not vi.load_simulation(network_file):
        return
    vi.set_evaluation_outputs()  # save nothing
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
    vi.set_evaluation_outputs(save_vehicle_record=save_results,
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


def plot_acc_av_and_cav_results(save_results=False):
    network_name = 'in_and_out'
    vehicle_types = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED
    ]
    percentage = [0, 100]
    veh_inputs = [1000, 2000]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_types, percentage, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.get_flow_and_risk_plots(veh_inputs, simulation_percentages)


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
    result_analyzer.box_plot_y_vs_controlled_percentage(
        'flow', veh_inputs, percentages_per_vehicle_type, 10)

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
    ra.plot_heatmap('lane_change_count', simulation_percentages,
                    [inputs_per_lane], accepted_risks)


# TODO: move this to some other file
def create_vehicle_percentages_dictionary(
        vehicle_types: List[VehicleType], percentages: List[int],
        n_vehicle_types: int) -> List[Dict[VehicleType, int]]:
    """

    :param vehicle_types:
    :param percentages:
    :param n_vehicle_types:
    :return: Dictionary with tuple of VehicleType as key and list of
     percentages as value
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
    # Options: i710, us101, in_and_out, in_and_merge,
    # platoon_lane_change, traffic_lights
    scenario_name = 'in_and_out_safe'
    vehicle_type = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
        # VehicleType.TRAFFIC_LIGHT_ACC,
        # VehicleType.TRAFFIC_LIGHT_CACC
    ]

    percentages = [i for i in range(0, 101, 100)]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_type, percentages, 1)
    inputs_per_lane = [1000, 2000]
    accepted_risks = [0]  # [i for i in range(0, 31, 10)]

    # =============== Running =============== #
    # vi = VissimInterface()
    # vi.load_simulation(scenario_name)
    # for ipl in inputs_per_lane:
    #     vi.run_multiple_scenarios(simulation_percentages,
    #                               [ipl],
    #                               accepted_risks)
    # vi.close_vissim()

    # =============== Post processing =============== #
    # for item in simulation_percentages:
    #     vehicle_types = list(item.keys())
    #     percentages = list(item.values())
    #     post_processing.create_summary_with_risks(
    #         scenario_name, vehicle_types, percentages, inputs_per_lane,
    #         accepted_risks)

    # MOVES
    # for sp in simulation_percentages:
    #     for ipl in inputs_per_lane:
    #         post_processing.translate_links_from_vissim_to_moves(
    #             scenario_name, ipl, sp)

    # r = readers.MOVESDatabaseReader(scenario_name)
    # r.load_data_with_controlled_percentage(simulation_percentages, [1000, 2000])

    # file_handler = file_handling.FileHandler(scenario_name)
    # try:
    #     file_handler.copy_results_from_multiple_scenarios(
    #         simulation_percentages, inputs_per_lane, accepted_risks)
    # except FileNotFoundError:
    #     print("Couldn't copy files to shared folder.")
    #     # continue

    # =============== Check results graphically =============== #
    all_inputs = [1000, 1200, 1500, 2000]
    # all_plots_for_scenarios_with_risk(scenario_name, simulation_percentages,
    #                                   inputs_per_lane, accepted_risks,
    #                                   save_fig=False)
    ra = result_analysis.ResultAnalyzer(scenario_name, False)
    ra.plot_fundamental_diagram([1000, 2000], simulation_percentages,
                                accepted_risks=[0],
                                flow_sensor_name=['in'])
    ra.box_plot_y_vs_vehicle_type('flow', 'vehicles_per_lane', all_inputs,
                                  simulation_percentages, [0])
    # ra.plot_lane_change_risk_histograms_risk_as_hue('total_lane_change_risk',
    #                                                 simulation_percentages,
    #                                                 inputs_per_lane,
    #                                                 accepted_risks,
    #                                                 )
    # ra.plot_heatmap('lane_change_count', simulation_percentages,
    #                 inputs_per_lane, accepted_risks)

    # plot_acc_av_and_cav_results(False)
    # plot_cav_varying_percentage_results(False)
    # plot_traffic_lights_results(False)

    # save_fig = False
    # ra = result_analysis.ResultAnalyzer(scenario_name, save_fig)
    # ra.plot_lane_change_risks(None, None, 0)


if __name__ == '__main__':
    main()
