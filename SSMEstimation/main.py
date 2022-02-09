import os
from typing import List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_writer
import file_handling
import post_processing
import result_analysis
from data_writer import SyntheticDataWriter
import readers
from vehicle import VehicleType
from vissim_interface import VissimInterface


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    sim_params = {'SimPeriod': 200, 'RandSeed': 1}
    vi = VissimInterface()
    if not vi.load_simulation(network_file):
        return
    vi.set_evaluation_outputs(False, False, False, False)
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
    vi.set_evaluation_outputs(save_results, save_results,
                              save_results, save_results)
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


def create_and_save_synthetic_data():
    vx = 20
    delta_v = 0
    writer = SyntheticDataWriter()
    data = writer.create_data(vx, delta_v)
    writer.write_data(data)


def test_safe_gap_computation():
    # create_and_save_synthetic_data()
    data_source = 'synthetic'
    # post_process_and_save('synthetic_data', network_file)
    data_reader = readers.PostProcessedDataReader(data_source)
    data = data_reader.load_data()

    gamma = 1 / 0.8
    rho = 0.2
    ssm_estimator = post_processing.SSMEstimator(data)
    ssm_estimator.include_risk(same_type_gamma=gamma)
    ssm_estimator.include_estimated_risk(rho=rho, same_type_gamma=gamma)
    risk_gap = data.loc[data['delta_x'] <= (data['safe_gap'] + 0.02),
                        'delta_x']
    risk = data.loc[data['delta_x'] <= (data['safe_gap'] + 0.02), 'exact_risk']
    estimated_risk_gap = data.loc[data['delta_x'] <= (data['vf_gap'] + 0.02),
                                  'delta_x']
    estimated_risk = data.loc[data['delta_x'] <= (data['vf_gap'] + 0.02),
                              'estimated_risk']

    # Plot
    fig, ax = plt.subplots()
    ax.plot(risk_gap, risk)
    ax.plot(estimated_risk_gap, estimated_risk)
    plt.show()
    # save_post_processed_data(ssm_estimator.veh_data,
    #                          data_reader.data_source, data_reader.file_name)


def run_simulations(network_name: str,
                    vehicle_types: Union[VehicleType, List[VehicleType]],
                    percentage: int,
                    input_per_lane: int,
                    percentage_increase: int = None,
                    final_percentage: int = None,
                    input_per_lane_increase: int = None,
                    final_input_per_lane: int = None,
                    debugging: bool = False):
    """
    Runs sets of simulations in VISSIM.

    :param network_name: Options are i710, us-101, in_and_out, in_and_merge
    :param vehicle_types: Type of controlled vehicle in the simulation
    :param percentage: Percentage of controlled vehicles in the simulation
    :param input_per_lane: Vehicles per hour entering the simulation on each
     lane
    :param percentage_increase: Percentage increase of controlled vehicles
     between sets of runs. Optional; must be set together with
     percentage_final
    :param final_percentage: Percentage of controlled vehicles in the last
     set of simulation runs. Optional; must be set together with
     percentage_increase
    :param input_per_lane_increase: Vehicle input per lane increase between
     sets of simulation runs. Optional; must be set together with
     final_input_per_lane
    :param final_input_per_lane: Vehicle input per lane of the last set of
     simulation runs. Optional; must be set together with
     input_per_lane_increase
    :param debugging: If true, only runs 2 simulations per set
    :return: Nothing. Results are saved to files
    """
    # Set all parameters
    if (percentage_increase is None) != (final_percentage is None):
        raise ValueError("Either set both percentage_increase and "
                         "percentage_final values or leave both as None.")
    if (input_per_lane_increase is None) != (final_input_per_lane is None):
        raise ValueError("Either set both input_per_lane_increase and "
                         "final_input_per_lane values or leave both as None.")

    if not isinstance(vehicle_types, list):
        vehicle_types = [vehicle_types]

    initial_percentage = percentage
    initial_input_per_lane = input_per_lane
    if percentage_increase is None:
        # Choice that guarantees only the initial percentage is tested
        percentage_increase = 150
        final_percentage = initial_percentage
    if input_per_lane_increase is None:
        # Choice that guarantees only the initial input per lane is tested
        input_per_lane_increase = 1000
        final_input_per_lane = initial_input_per_lane
    runs_per_scenario = 2 if debugging else 10

    # Running
    network_file = file_handling.network_names_map[network_name]
    vi = VissimInterface()
    vi.load_simulation(network_file)
    for vt in vehicle_types:
        vi.run_with_increasing_controlled_vehicle_percentage(
            vt,
            percentage_increase=percentage_increase,
            initial_percentage=initial_percentage,
            final_percentage=final_percentage,
            input_increase_per_lane=input_per_lane_increase,
            initial_input_per_lane=initial_input_per_lane,
            max_input_per_lane=final_input_per_lane,
            runs_per_input=runs_per_scenario)
    vi.close_vissim()


def main():
    # image_folder = "G:\\My Drive\\Safety in Mixed Traffic\\images"

    # ============ Playing with Traffic Lights =========== #
    # sc_reader = readers.SignalControllerFileReader('traffic_lights')
    # file_id = 1
    # sc_tree = sc_reader.load_data(file_id)
    # sc_writer = data_writer.SignalControllerTreeEditor()
    # sc_writer.set_times(sc_tree, 20, 30)
    # sc_writer.save_file(sc_tree, sc_reader.data_dir,
    #                     sc_reader.network_name + str(file_id))
    vi = VissimInterface()
    vi.load_simulation('traffic_lights')
    vi.set_traffic_lights()

    # =============== Define data source =============== #
    # Options: i710, us-101, in_and_out, in_and_merge
    network_file = file_handling.network_names_map['in_and_out']
    vehicle_type = [# VehicleType.ACC,
                    # VehicleType.AUTONOMOUS,
                    VehicleType.CONNECTED
                    ]

    # =============== Temporary tests  =============== #
    # ra = result_analysis.ResultAnalyzer(network_file, vehicle_type)
    # ra.find_unfinished_simulations(100)

    # =============== Running =============== #
    # run_simulations(network_name='in_and_out', vehicle_types=vehicle_type,
    #                 percentage=25, percentage_increase=25,
    #                 final_percentage=75,
    #                 input_per_lane=2000)

    # =============== Post processing =============== #

    # post_processor = post_processing.DataPostProcessor()
    # for vt in vehicle_type:
    #     for percentage in range(25, 75+1, 25):
    #         post_processor.create_ssm_summary(network_file,
    #                                           vt,
    #                                           percentage,
    #                                           vehicle_inputs=[2000],
    #                                           # debugging=True
    #                                           )
    #         post_processor.merge_data(network_file, vt, percentage)

    # =============== Check results numbers =============== #
    # for percentage in range(100, 100+1, 25):
    #     print('Percentage: ', percentage)
    #     post_processor.check_human_take_over(network_file,
    #                                          VehicleType.CONNECTED,
    #                                          percentage, [2000])

    # =============== Check results graphically =============== #

    vehicle_types = [
        # VehicleType.ACC,
        # VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED
        ]
    result_analyzer = result_analysis.ResultAnalyzer('in_and_out',
                                                     vehicle_types)
    # result_analyzer.find_removed_vehicles(VehicleType.CONNECTED, 100, [1000,
    #                                                                   2000])

    save_results = False

    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [i for i in range(1000, 2001, 1000)]
    # result_analyzer.plot_risky_maneuver_histogram_per_vehicle_type(
    #     percentages, [2000], min_total_risk=1,
    #     should_save_fig=save_results)
    # result_analyzer.plot_risky_maneuver_histogram_per_percentage(
    #     percentages, 2000, min_total_risk=1, should_save_fig=save_results
    # )
    # result_analyzer.box_plot_y_vs_controlled_percentage(
    #     'flow', [2000], percentages, warmup_time=10,
    #     should_save_fig=save_results
    # )
    # result_analyzer.box_plot_y_vs_controlled_percentage(
    #     'risk', veh_inputs, percentages, warmup_time=10,
    #     should_save_fig=save_results
    # )

    # for veh_input in [2000]:
    #     result_analyzer.plot_y_vs_time('flow', veh_input, percentages,
    #                                    warmup_time=5)
    #     result_analyzer.plot_y_vs_time('risk', veh_input, percentages,
    #                                    warmup_time=5)
    #     # result_analyzer.plot_y_vs_time('risk_no_lane_change', veh_input,
    #     #                                [100], warmup_time=5)

    # result_analyzer.box_plot_y_vs_controlled_percentage(
    #     'risk_no_lane_change', veh_inputs, [100], warmup_time=10)

    # result_analyzer.box_plot_y_vs_controlled_percentage(
    #     'risk', veh_inputs, [100], warmup_time=10,
    #     should_save_fig=save_results
    # )
    # result_analyzer = result_analysis.ResultAnalyzer(
    #     'in_and_out', [VehicleType.AUTONOMOUS, VehicleType.CONNECTED])
    # result_analyzer.box_plot_y_vs_controlled_percentage(
    #     'risk', veh_inputs, [100], warmup_time=10,
    #     should_save_fig=save_results
    # )

    # percentages = [i for i in range(0, 101, 100)]
    # vehicle_types = [
    #     VehicleType.ACC,
    #     VehicleType.AUTONOMOUS,
    #     VehicleType.CONNECTED
    # ]
    # result_analyzer = result_analysis.ResultAnalyzer('in_and_out',
    #                                                  vehicle_types)
    # result_analyzer.box_plot_y_vs_vehicle_type('flow', 2000, percentages,
    #                                            warmup_time=10,
    #                                            should_save_fig=save_results)
    # result_analyzer.box_plot_y_vs_vehicle_type('risk', 2000, percentages,
    #                                            warmup_time=10,
    #                                            should_save_fig=save_results)

    # =============== SSM computation check =============== #
    # veh_rec_reader = readers.VehicleRecordReader('toy')
    # veh_rec = veh_rec_reader.load_data(51, 100)
    # pp = result_analysis.VehicleRecordPostProcessor('vissim', veh_rec)
    # pp.post_process_data()
    #
    # ssm_estimator = result_analysis.SSMEstimator(veh_rec)
    # ssm_estimator.include_collision_free_gap()

    # Save all SSMs #
    # ssm_names = ['TTC', 'DRAC', 'collision_free_gap',
    #              'vehicle_following_gap', 'CPI',
    #              'exact_risk', 'estimated_risk']
    # ssm_names = ['CPI']
    # create_ssm(data_source, network_file, ssm_names)

    # =============== SSM tests =============== #
    # reader = readers.VehicleRecordReader('toy')
    # veh_record = reader.load_data(1, 'test')
    # pp = result_analysis.VehicleRecordPostProcessor('vissim', veh_record)
    # pp.post_process_data()
    # ssm_estimator = result_analysis.SSMEstimator(veh_record)
    # ssm_estimator.include_collision_free_gap()
    # ssm_estimator.include_exact_risk()
    # print('done')
    # # ssm_estimator.plot_ssm('low_TTC')
    # # ssm_estimator.plot_ssm('exact_risk')
    # # ssm_estimator.plot_ssm('estimated_risk')
    # # ssm_estimator.plot_ssm('vx')
    # image_path = os.path.join(image_folder, network_file)
    # ssm_estimator.plot_ssm_moving_average('low_TTC', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('high_DRAC', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('CPI', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('exact_risk', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('estimated_risk')
    # ssm_estimator.plot_ssm_moving_average('vx')


if __name__ == '__main__':
    main()
