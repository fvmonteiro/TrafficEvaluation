from collections import defaultdict
from typing import List, Dict, Tuple
from typing import Union

import matplotlib.pyplot as plt

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
                    percentage_per_vehicle_types: List[Dict[VehicleType, int]],
                    inputs_per_lane: List[int],
                    debugging: bool = False):
    """
    Runs sets of simulations in VISSIM.

    :param network_name: Options are i710, us-101, in_and_out, in_and_merge,
     and traffic_lights
    :param percentage_per_vehicle_types: Dictionary describing the desired
     percentages for each tuple of vehicle types
    :param inputs_per_lane: Vehicles per hour entering the simulation on each
     lane
    :param debugging: If true, only runs 2 simulations per set
    :return: Nothing. Results are saved to files
    """
    # Set all parameters
    runs_per_scenario = 2 if debugging else 10

    # Running
    vi = VissimInterface()
    vi.load_simulation(network_name)
    if network_name == 'traffic_lights':
        vi.set_traffic_lights()

    # for vt in vehicle_types:
    vi.run_with_varying_controlled_percentage(percentage_per_vehicle_types,
                                              inputs_per_lane,
                                              runs_per_input=runs_per_scenario)
    # if initial_percentage == 0:  # we don't want to run zero percent twice
    #     initial_percentage += percentage_increase
    vi.close_vissim()


def post_process_results(network_name: str,
                         vehicle_types: Union[VehicleType, List[VehicleType]],
                         input_per_lane: Union[int, List[int]],
                         initial_percentage: int,
                         percentage_increment: int = None,
                         final_percentage: int = None,
                         debugging: bool = False):
    """
    Creates files with safety and flow measurements for the highway
    scenario in_and_out
    :param network_name: Options are i710, us-101, in_and_out, in_and_merge,
     and traffic_lights
    :param vehicle_types: Type of controlled vehicle in the simulation
    :param input_per_lane: Vehicles per hour entering the simulation on each
     lane
    :param initial_percentage: Percentage of controlled vehicles in the
     simulation
    :param percentage_increment: (optional) Percentage increase of controlled
     vehicles between sets of runs. Must be set together with
     percentage_final
    :param final_percentage: (optional) Percentage of controlled vehicles in
     the last set of simulation runs. Must be set together with
     percentage_increment
    :param debugging: If true, loads fewer samples from vehicle records and
     does not save results.
    :return:
    """
    if (percentage_increment is None) != (final_percentage is None):
        raise ValueError("Either set both percentage_increment and "
                         "percentage_final values or leave both as None.")
    if not percentage_increment:
        final_percentage = initial_percentage
        percentage_increment = 100
    if not isinstance(input_per_lane, list):
        input_per_lane = [input_per_lane]

    post_processor = post_processing.DataPostProcessor()
    for vt in vehicle_types:
        for percentage in range(initial_percentage,
                                final_percentage + 1, percentage_increment):
            post_processor.create_safety_summary(network_name, vt, percentage,
                                                 vehicle_inputs=input_per_lane,
                                                 debugging=debugging)
            # post_processor.merge_data(network_name, vt, percentage)
        if initial_percentage == 0:
            initial_percentage += percentage_increment


def plot_acc_av_and_cav_results(save_results=False):
    network_name = 'in_and_out'
    vehicle_types = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED
    ]
    percentage = [0, 100]
    veh_inputs = [1000, 2000]
    d_list = create_vehicle_percentages_dictionary(vehicle_types, percentage, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name)
    result_analyzer.get_flow_and_risk_plots(veh_inputs, d_list, save_results)


def plot_cav_varying_percentage_results(save_results=False):
    network_name = 'in_and_out'
    vehicle_types = [VehicleType.CONNECTED]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [1000, 2000]
    d_list = create_vehicle_percentages_dictionary(vehicle_types,
                                                   percentages, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name)
    result_analyzer.get_flow_and_risk_plots(veh_inputs, d_list, save_results)


def plot_traffic_lights_results(save_results=False):
    network_name = 'traffic_lights'
    vehicle_types = [VehicleType.TRAFFIC_LIGHT_ACC,
                     VehicleType.TRAFFIC_LIGHT_CACC]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [500, 1000]
    percentages_per_vehicle_type = create_vehicle_percentages_dictionary(
        vehicle_types, percentages, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name)
    result_analyzer.box_plot_y_vs_controlled_percentage(
        'flow', veh_inputs, percentages_per_vehicle_type, 10, False)


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

    # =============== Define data source =============== #
    # Options: i710, us101, in_and_out, in_and_merge,
    # platoon_lane_change, traffic_lights
    network_name = 'traffic_lights'
    vehicle_type = [
        # VehicleType.ACC,
        # VehicleType.AUTONOMOUS,
        # VehicleType.CONNECTED,
        VehicleType.TRAFFIC_LIGHT_ACC,
        VehicleType.TRAFFIC_LIGHT_CACC
    ]

    percentages = [0, 25, 50, 75, 100]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_type, percentages, 1)
    # run_simulations(network_name, simulation_percentages, [500, 1000])

    # =============== Running =============== #
    # run_simulations(network_name=network_name, vehicle_types=vehicle_type,
    #                 percentage=0,
    #                 percentage_increase=100,
    #                 final_percentage=100,
    #                 input_per_lane=500,
    #                 input_per_lane_increase=500,
    #                 final_input_per_lane=1000,
    #                 )
    # vi = vissim_interface.VissimInterface()
    # vi.load_simulation(network_name)
    # vi.reset_saved_simulations(warning_active=False)
    # vi.run_platoon_scenario()

    # =============== Post processing =============== #
    # post_process_results(network_name, vehicle_type,
    #                      input_per_lane=[500, 1000],
    #                      initial_percentage=100,
    #                      percentage_increment=100,
    #                      final_percentage=100
    #                      )

    # post_processor = post_processing.DataPostProcessor()
    # post_processor.create_ssm_summary(network_name,
    #                                   vehicle_type[0],
    #                                   0, [1000],
    #                                   debugging)
    # post_processor.merge_data(network_name, vehicle_type[0], 0)
    # post_processing.DataPostProcessor.find_traffic_light_violations_all(
    #     network_name, vehicle_type[0], 0, [1000], debugging=True
    # )

    # =============== Check results numbers =============== #

    # =============== Check results graphically =============== #
    # plot_acc_av_and_cav_results(False)
    # plot_cav_varying_percentage_results(False)
    # plot_traffic_lights_results(False)
    # percentage = [0, 100]
    ra = result_analysis.ResultAnalyzer('traffic_lights')
    ra.plot_color_map('vehicle_count', simulation_percentages, [500], 10)
    # veh_inputs = [1000, 2000]
    # result_analyzer = result_analysis.ResultAnalyzer(network_name)
    # result_analyzer.box_plot_y_vs_controlled_percentage('flow', veh_inputs,
    #                                                     percentage, 10)
    # result_analyzer.box_plot_y_vs_controlled_percentage(
    #     'barrier_function_risk', veh_inputs, percentage, 10)
    # result_analyzer.plot_risky_maneuver_histogram_per_vehicle_type(
    #     percentage, veh_inputs)
    # result_analyzer.plot_violations_per_control_percentage(
    #     percentage, veh_inputs, 10)


if __name__ == '__main__':
    main()
