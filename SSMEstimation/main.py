import time
from typing import List, Dict
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns

import data_writer
import file_handling
import post_processing
import result_analysis
import vehicle
import vissim_interface
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


def test_safe_gap_computation():
    # create_and_save_synthetic_data()
    data_source = 'synthetic'
    # post_process_and_save('synthetic_data', network_file)
    data_reader = readers.PostProcessedDataReader_OLD(data_source)
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
    :param percentage_per_vehicle_types: TODO
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
                         percentage_per_vehicle_types: List[Dict[VehicleType,
                                                                 int]],
                         input_per_lane: Union[int, List[int]],
                         debugging: bool = False):
    """
    Creates files with safety and flow measurements for the highway
    scenario in_and_out
    :param network_name: Options are i710, us-101, in_and_out, in_and_merge,
     and traffic_lights
    :param percentage_per_vehicle_types: TODO
    :param input_per_lane: Vehicles per hour entering the simulation on each
     lane
    :param debugging: If true, loads fewer samples from vehicle records and
     does not save results.
    :return:
    """
    if not isinstance(input_per_lane, list):
        input_per_lane = [input_per_lane]

    for item in percentage_per_vehicle_types:
        vehicle_types = list(item.keys())
        percentages = list(item.values())
        post_processing.create_simulation_summary(network_name, vehicle_types,
                                                  percentages,
                                                  vehicle_inputs=input_per_lane,
                                                  debugging=debugging)


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
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.get_flow_and_risk_plots(veh_inputs, d_list)


def plot_cav_varying_percentage_results(save_results=False):
    network_name = 'in_and_out'
    vehicle_types = [VehicleType.CONNECTED]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [1000, 2000]
    d_list = create_vehicle_percentages_dictionary(vehicle_types,
                                                   percentages, 1)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.get_flow_and_risk_plots(veh_inputs, d_list)


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

    ra = result_analysis.ResultAnalyzer(network_name, save_results)
    ra.accel_vs_time_for_different_vehicle_pairs()
    ra.plot_heatmap('vehicle_count', percentages_per_vehicle_type,
                    veh_inputs, 10)
    ra.plot_heatmap('barrier_function_risk', percentages_per_vehicle_type,
                    veh_inputs, 10)
    ra.plot_heatmap('discomfort', percentages_per_vehicle_type, veh_inputs, 10)
    ra.plot_violations_heatmap(percentages_per_vehicle_type, veh_inputs, 10)


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
    network_name = 'in_and_out'
    vehicle_type = [
        # VehicleType.ACC,
        # VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
        # VehicleType.TRAFFIC_LIGHT_ACC,
        # VehicleType.TRAFFIC_LIGHT_CACC
    ]

    percentages = [0, 100]
    simulation_percentages = create_vehicle_percentages_dictionary(
        vehicle_type, percentages, 1)
    # vi = vissim_interface.VissimInterface()
    # vi.load_simulation(network_name)
    # vi.run_multiple_scenarios(simulation_percentages, [500, 1000], [0, 10],
    #                           runs_per_scenario=2, simulation_period=360)
    # vehicle_types = simulation_percentages
    # file_handling.copy_results_from_multiple_scenarios(network_name, )
    # =============== Tests ================= #
    # post_processing.create_simulation_summary_test(network_name)

    # lc_reader = readers.LaneChangeReader(network_name)
    # veh_reader = readers.VehicleRecordReader(network_name)
    # risky_maneuver_reader = readers.RiskyManeuverReader(network_name)
    # lc_data = lc_reader.load_test_data()
    # veh_data = veh_reader.load_test_data()
    # risky_maneuver_data = risky_maneuver_reader.load_test_data()
    #
    # # start_time = time.perf_counter()
    # post_processing.complement_lane_change_data(veh_data, lc_data,
    #                                             risky_maneuver_data)
    # # end_time = time.perf_counter()
    # # print('time: ', end_time - start_time)
    # post_processing.label_lane_changes(network_name, veh_data, lc_data)
    # writer = data_writer.LaneChangeWriter(network_name, None)
    # writer.save_as_csv(lc_data, None, None)

    # =============== Running =============== #
    # run_simulations(network_name, simulation_percentages, [500, 1000])
    # vi = vissim_interface.VissimInterface()
    # vi.load_simulation(network_name)
    # vi.reset_saved_simulations(warning_active=False)
    # vi.run_platoon_scenario()

    # =============== Post processing =============== #
    # post_process_results(network_name,
    #                      simulation_percentages,
    #                      input_per_lane=[500, 1000])
    # pp = post_processing
    # pp.create_simulation_summary('in_and_out', [VehicleType.CONNECTED],
    #                              [25], [1000], True)

    # =============== Checking risk computations =============== #
    # test_risk_computation()

    # =============== Check results graphically =============== #
    # plot_acc_av_and_cav_results(False)
    # plot_cav_varying_percentage_results(False)
    # plot_traffic_lights_results(False)
    # percentages = [0, 25, 50, 75, 100]
    # simulation_percentages = create_vehicle_percentages_dictionary(
    #     vehicle_type, percentages, 1)
    # simulation_percentages += create_vehicle_percentages_dictionary(
    #     vehicle_type, percentages, 2)

    # save_fig = False
    # ra = result_analysis.ResultAnalyzer(network_name, save_fig)
    # ra.plot_lane_change_risks(None, None, 0)


if __name__ == '__main__':
    main()
