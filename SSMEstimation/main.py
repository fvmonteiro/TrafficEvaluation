import matplotlib.pyplot as plt

import data_writer
from file_handling import FileHandler
import moves_file_handling
import post_processing
import result_analysis
import scenario_handling
import vehicle
import vissim_interface
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
        sim_params = {"SimPeriod": simulation_time, "SimRes": sim_resolution,
                      "UseMaxSimSpeed": True, "RandSeed": random_seed}
        vi.set_simulation_parameters(sim_params)
        vi.run_i710_simulation(idx_scenario, demand)


def test_risk_computation():
    data_source = "synthetic"
    follower_v0 = 20
    leader_v0 = 10
    follower_type = Vehicle.VISSIM_CAR_ID
    leader_type = Vehicle.VISSIM_CAR_ID
    is_lane_changing = True
    veh_data = data_writer.SyntheticDataWriter.create_single_veh_data(
        follower_v0, leader_v0, is_lane_changing, follower_type, leader_type)
    post_processing.post_process_data(data_source, veh_data)
    ssm_estimator = post_processing.SSMEstimator(veh_data)
    ssm_estimator.include_ssms(["risk"])
    veh1 = veh_data[veh_data["veh_id"] == 1]

    # Numerical risk computation
    g0 = veh1["delta_x"].to_numpy()
    tc, severity = vehicle.find_collision_time_and_severity(
        g0, follower_v0, leader_v0, is_lane_changing,
        follower_type, leader_type)

    plt.plot(g0, veh1["risk"], "b-", label="risk")
    plt.plot(g0, severity, "r--", label="numerical")

    plt.legend()
    plt.show()


def run_all_safe_lane_change_scenarios():
    # Scenario definition
    scenario_name = "in_and_out_safe"
    vehicle_type = [
        VehicleType.VIRDI,
        VehicleType.CONNECTED,
        # VehicleType.AUTONOMOUS,
        # VehicleType.ACC
    ]

    percentages = [100]  # [i for i in range(0, 101, 100)]
    full_penetration = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_type, percentages, 1))
    # varied_cav_penetration = create_vehicle_percentages_dictionary(
    #     [VehicleType.CONNECTED], [i for i in range(0, 76, 25)], 1)
    inputs_per_lane = [1000, 2000]

    full_penetration_scenarios = scenario_handling.create_multiple_scenarios(
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
    file_handler = FileHandler(scenario_name)
    try:
        file_handler.export_multiple_results_to_cloud(
            full_penetration_scenarios)
    except FileNotFoundError:
        print("Couldn't copy files to shared folder.")
        # continue


def run_comparison_method():
    scenario_name = "in_and_out_safe"
    inputs_per_lane = [1000, 2000]

    # Running
    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    for vt in [VehicleType.VIRDI,
               # VehicleType.AUTONOMOUS, VehicleType.ACC
               ]:
        scenarios = scenario_handling.create_multiple_scenarios(
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
    file_handler = FileHandler(scenario_name)
    try:
        file_handler.export_multiple_results_to_cloud(
            scenarios)
    except FileNotFoundError:
        print("Couldn't copy files to shared folder.")


def run_all_risky_lane_change_scenarios():
    # Scenario definition
    scenario_name = "risky_lane_changes"
    vehicle_type = [
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
    ]

    percentages = [100]  # [i for i in range(0, 101, 100)]
    full_penetration = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_type, percentages, 1))
    # varied_cav_penetration = create_vehicle_percentages_dictionary(
    #     [VehicleType.CONNECTED], [i for i in range(0, 76, 25)], 1)
    orig_lane_input = [500, 1000, 1500, 2000]
    accepted_risks = [10, 20, 30]

    full_penetration_scenarios = scenario_handling.create_multiple_scenarios(
        full_penetration, orig_lane_input, accepted_risks
    )

    # Running
    # vi = VissimInterface()
    # vi.load_simulation(scenario_name)
    # vi.run_multiple_scenarios(full_penetration_scenarios, runs_per_scenario=3)
    # vi.close_vissim()

    # Post-processing
    post_processing.create_summary_with_risks(
        scenario_name, full_penetration_scenarios, analyze_lane_change=True)
    # for ipl in inputs_per_lane:
    #     post_processing.get_individual_vehicle_trajectories_to_moves(
    #         scenario_name, ipl, sp, 0)

    # Transfer files to the cloud
    file_handler = FileHandler(scenario_name)
    try:
        file_handler.export_multiple_results_to_cloud(
            full_penetration_scenarios)
    except FileNotFoundError:
        print("Couldn't copy files to shared folder.")


def run_platoon_scenarios():
    scenario_name = "platoon_discretionary_lane_change"
    lc_scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "dest_lane_speed", False, False)
    # vi = VissimInterface()
    # vi.load_simulation(scenario_name)
    # vi.run_multiple_platoon_lane_change_scenarios(lc_scenarios,
    #                                               runs_per_scenario=1)
    # vi.close_vissim()
    post_processing.create_platoon_lane_change_summary(
        scenario_name, lc_scenarios)

    # hdv_scenarios = file_handling.get_platoon_lane_change_scenarios(
    #     "vehicles per hour", True, False)
    # vi.run_multiple_platoon_lane_change_scenarios(hdv_scenarios,
    #                                               runs_per_scenario=10)
    # vi.close_vissim()
    # post_processing.create_platoon_lane_change_summary(
    #     scenario_name, hdv_scenarios)


def plot_acc_av_and_cav_results(save_results=False):
    scenario_name = "in_and_out_safe"
    vehicle_types = [
        VehicleType.ACC,
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
        # VehicleType.VIRDI,
    ]
    percentage = [0, 100]
    veh_inputs = [1000, 2000]
    simulation_percentages = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_types, percentage, 1))
    # Our own scenarios
    scenarios = scenario_handling.create_multiple_scenarios(
        simulation_percentages, veh_inputs, accepted_risks=[0])
    # The comparison scenarios
    scenarios.extend(scenario_handling.create_multiple_scenarios(
        [{VehicleType.VIRDI: 100}], veh_inputs))
    result_analyzer = result_analysis.ResultAnalyzer(scenario_name,
                                                     save_results)
    result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
        scenarios, warmup_time=10)
    # result_analyzer.plot_link_evaluation_box_plot_vs_controlled_percentage(
    #     "volume", scenarios, warmup_time=10)
    # result_analyzer.plot_risk_histograms("total_risk", scenarios, min_risk=1)
    # result_analyzer.plot_risk_histograms("total_lane_change_risk", scenarios,
    #                                      min_risk=1)
    # result_analyzer.plot_fd_discomfort(scenarios)
    # result_analyzer.plot_emission_heatmap(scenarios)


def plot_comparison_scenario(save_results=False):
    scenario_name = "in_and_out_safe"
    vehicle_types = [
        VehicleType.VIRDI,
        VehicleType.CONNECTED
    ]
    percentage = [0, 100]
    veh_inputs = [1000, 2000]
    simulation_percentages = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_types, percentage, 1))
    scenarios = scenario_handling.create_multiple_scenarios(
        simulation_percentages, veh_inputs)
    result_analyzer = result_analysis.ResultAnalyzer(scenario_name,
                                                     save_results)
    # result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
    #     scenarios, warmup_time=10)
    result_analyzer.plot_link_data_box_plot_vs_controlled_percentage(
        "volume", scenarios, warmup_time=10, aggregation_period=5)
    result_analyzer.plot_link_data_box_plot_vs_controlled_percentage(
        "average_speed", scenarios, warmup_time=10, aggregation_period=5)
    result_analyzer.plot_link_data_box_plot_vs_controlled_percentage(
        "density", scenarios, warmup_time=10, aggregation_period=5)
    # result_analyzer.plot_risk_histograms("total_risk", scenarios, min_risk=1)
    # result_analyzer.plot_risk_histograms("total_lane_change_risk", scenarios,
    #                                      min_risk=1)
    # result_analyzer.plot_emission_heatmap(scenarios)
    result_analyzer.plot_fd_discomfort(scenarios)


def plot_cav_varying_percentage_results(save_results=False):
    network_name = "in_and_out"
    vehicle_types = [VehicleType.CONNECTED]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [1000, 2000]
    simulation_percentages = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_types, percentages, 1))
    scenarios = scenario_handling.create_multiple_scenarios(
        simulation_percentages, veh_inputs)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.get_flow_and_risk_plots(scenarios)


def plot_risky_lane_changes_results(save_results=False):
    scenario_name = "risky_lane_changes"
    vehicle_types = [
        VehicleType.AUTONOMOUS,
        VehicleType.CONNECTED,
    ]
    percentages = [0, 100]  # [i for i in range(0, 101, 100)]
    full_penetration = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_types, percentages, 1))
    # varied_cav_penetration = create_vehicle_percentages_dictionary(
    #     [VehicleType.CONNECTED], [i for i in range(0, 76, 25)], 1)
    orig_lane_input = [500, 1000, 1500, 2000]
    accepted_risks = [0]

    scenarios = scenario_handling.create_multiple_scenarios(
        full_penetration, orig_lane_input, accepted_risks
    )

    result_analyzer = result_analysis.ResultAnalyzer(scenario_name,
                                                     save_results)
    before_or_after_lc_point = "before"
    lanes = "both"
    segment = 1 if before_or_after_lc_point == "after" else 2

    result_analyzer.plot_link_data_box_plot_vs_controlled_percentage(
        "volume", scenarios, before_or_after_lc_point, lanes, segment,
        warmup_time=10, aggregation_period=5)
    result_analyzer.plot_link_data_box_plot_vs_controlled_percentage(
        "average_speed", scenarios, before_or_after_lc_point, lanes, segment,
        warmup_time=10, aggregation_period=5)
    result_analyzer.print_unfinished_lane_changes_for_risky_scenario(
        scenarios)


def plot_traffic_lights_results(save_results=False):
    network_name = "traffic_lights"
    vehicle_types = [VehicleType.TRAFFIC_LIGHT_ACC,
                     VehicleType.TRAFFIC_LIGHT_CACC]
    percentages = [i for i in range(0, 101, 25)]
    veh_inputs = [500, 1000]
    percentages_per_vehicle_type = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_types, percentages, 1))
    scenarios = scenario_handling.create_multiple_scenarios(
        percentages_per_vehicle_type, veh_inputs)
    result_analyzer = result_analysis.ResultAnalyzer(network_name, save_results)
    result_analyzer.plot_flow_box_plot_vs_controlled_percentage(
        scenarios, warmup_time=10)

    result_analyzer.accel_vs_time_for_different_vehicle_pairs()
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        "vehicle_count", scenarios, 10)
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        "barrier_function_risk", scenarios, 10)
    result_analyzer.plot_heatmap_for_traffic_light_scenario(
        "discomfort", scenarios, 10)
    result_analyzer.plot_violations_heatmap(scenarios, 10)


def all_plots_for_scenarios_with_risk(network_name: str, save_fig=False):
    vehicle_types = [VehicleType.AUTONOMOUS, VehicleType.CONNECTED]
    percentages = [0, 100]
    vehicle_percentages = (
        scenario_handling.create_vehicle_percentages_dictionary(
            vehicle_types, percentages, 1))
    inputs_per_lane = [1000, 2000]
    risks = [i for i in range(0, 31, 10)]
    scenarios = scenario_handling.create_multiple_scenarios(
        vehicle_percentages, inputs_per_lane, risks)

    ra = result_analysis.ResultAnalyzer(network_name,
                                        should_save_fig=save_fig)
    scenario_subsets = scenario_handling.split_scenario_by(
        scenarios, "vehicles_per_lane")
    for key, sc in scenario_subsets.items():
        ra.plot_pointplot_vs_accepted_risk("volume", sc)
        ra.plot_pointplot_vs_accepted_risk("average_speed", sc)
        ra.plot_pointplot_vs_accepted_risk("total_lane_change_risk", sc)
        # ra.box_plot_y_vs_vehicle_type("total_lane_change_risk",
        #                               "accepted_risk", sc)
        # ra.box_plot_y_vs_vehicle_type("volume", "accepted_risk", sc)
        # ra.plot_lane_change_count_heatmap_per_risk(sc)
        # ra.plot_risk_heatmap("total_lane_change_risk", sc)
        # ra.plot_total_output_heatmap_vs_risk(sc)


def all_plots_for_scenarios_with_risk_and_varying_penetration(
        network_name: str, save_fig=False):
    vehicle_types = [VehicleType.AUTONOMOUS, VehicleType.CONNECTED]
    percentages = [i for i in range(0, 101, 25)]
    inputs_per_lane = [1000]
    risks = [i for i in range(0, 21, 10)]
    ra = result_analysis.ResultAnalyzer(network_name,
                                        should_save_fig=save_fig)
    for vt in vehicle_types:
        vehicle_percentages = (
            scenario_handling.create_vehicle_percentages_dictionary(
                [vt], percentages, 1))
        scenarios = scenario_handling.create_multiple_scenarios(
            vehicle_percentages, inputs_per_lane, risks)
        # ra.plot_barplot_vs_accepted_risk("volume", scenarios)
        ra.plot_barplot_vs_accepted_risk("total_lane_change_risk", scenarios)
        # ra.box_plot_y_vs_vehicle_type("total_lane_change_risk",
        #                               "accepted_risk", sc)
        # ra.box_plot_y_vs_vehicle_type("volume", "accepted_risk", sc)
        # ra.plot_lane_change_count_heatmap_per_risk(sc)
        # ra.plot_risk_heatmap("total_lane_change_risk", sc)
        # ra.plot_total_output_heatmap_vs_risk(sc)


def plots_for_platoon_scenarios(should_save_fig: bool = False):
    scenario_name = "platoon_discretionary_lane_change"
    ra = result_analysis.ResultAnalyzer(scenario_name, should_save_fig,
                                        is_debugging=False)

    cav_scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "dest_lane_speed")
    # for i in [1, 3, 7, 10]:  # chosen after checking all figures
    #     ra.plot_platoon_states(cav_scenarios[i])
    ra.illustrate_travel_time_delay(cav_scenarios[0], lane="Origin",
                                    warmup_time=2, sim_time=5)

    selector = ["dest_lane_speed", "vehicles_per_lane", "platoon_size"]
    # for s in selector[1:]:
    #     cav_scenarios = scenario_handling.get_platoon_lane_change_scenarios(s)
    #     ra.compare_travel_times(cav_scenarios, x="delta_v",
    #                             plot_cols=s, warmup_time=1, sim_time=11)

    hdv_scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "vehicles_per_lane", with_hdv=True)
    # ra.plot_successful_maneuvers(hdv_scenarios)
    # ra.compare_travel_times(hdv_scenarios, x="delta_v",
    #                         plot_cols="vehicles_per_lane",
    #                         warmup_time=1, sim_time=11)

    scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "dest_lane_speed")
    # ra.compare_emissions_for_platoon_scenarios(scenarios)

    before_or_after_lc_point = "after"
    lanes = "both"
    segment = 1 if before_or_after_lc_point == "after" else 2
    all_sims = True
    scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "dest_lane_speed")
    scenarios_by_speed = scenario_handling.split_scenario_by(
        scenarios, "orig_and_dest_lane_speeds")
    for key, scenario_subset in scenarios_by_speed.items():
        print(key)
        warmup_time = 2.5
        sim_time = (12 if key[1] == "90" else 7) - warmup_time
    #     # ra.plot_travel_times_vs_entrance_times(scenario_subset, 1, 7)
        ys = [
            # "volume",
            # "average_speed",
            # "density",
            # "delay_relative"
        ]
        for y in ys:
            # ra.plot_link_data_box_plot_vs_strategy(
            #     y,  scenario_subset, before_or_after_lc_point, lanes, segment,
            #     warmup_time=warmup_time, sim_time=sim_time,
            #     aggregation_period=5)

            ra.plot_link_data_vs_time_per_strategy(
                y, scenario_subset, before_or_after_lc_point, lanes, segment,
                warmup_time=warmup_time, sim_time=sim_time,
                aggregation_period=5, use_all_simulations=all_sims)
    #     # ra.plot_y_vs_platoon_lc_strategy("platoon_maneuver_time", scenarios)
    #
    #     # for sc in scenario_subset:
    #     #     ra.speed_color_map(sc, link=3, warmup_time=2, sim_time=7)


def run_a_platoon_simulation():
    scenario_name = "platoon_discretionary_lane_change"
    other_vehicles = {VehicleType.HDV: 100}
    strategy = PlatoonLaneChangeStrategy.single_body_platoon
    scenario = scenario_handling.ScenarioInfo(
        other_vehicles, 0, platoon_lane_change_strategy=strategy,
        orig_and_dest_lane_speeds=("70", "90"),
        special_case="single_lane_change")
    vi = vissim_interface.VissimInterface()
    vi.load_simulation(scenario_name)
    vi.set_random_seed(8)
    vi.set_logged_vehicle_id(280)
    vi.run_platoon_scenario_sample(scenario, number_of_runs=1)
    vi.close_vissim()


def main():
    # plots_for_platoon_scenarios(False)
    # plot_comparison_scenario(False)
    # =============== Scenario Definition =============== #
    scenario_name = "risky_lane_changes"

    # =============== Running =============== #

    # run_all_risky_lane_change_scenarios()
    # plot_risky_lane_changes_results(False)
    run_a_platoon_simulation()
    # run_platoon_scenarios()
    # vi = VissimInterface()
    # vi.load_simulation(scenario_name)
    # # # vi.run_multiple_platoon_lane_change_scenarios(
    # # #     lc_scenarios, runs_per_scenario=1, is_debugging=False)
    # vi.run_multiple_platoon_lane_change_scenarios(
    #     all_scenarios, runs_per_scenario=1)
    # vi.close_vissim()

    # =============== Post processing =============== #
    # vehicle_types = [VehicleType.AUTONOMOUS, VehicleType.CONNECTED]
    # percentages = [0, 100]
    # vehicle_percentages = create_vehicle_percentages_dictionary(
    #     vehicle_types, percentages, 1)
    # inputs_per_lane = [2000]
    # risks = [i for i in range(0, 31, 10)]
    # scenarios = file_handling.create_multiple_scenarios(
    #     vehicle_percentages, inputs_per_lane, risks)
    # post_processing.create_summary_with_risks(scenario_name, scenarios)
    # post_processing.create_platoon_lane_change_summary(
    #     scenario_name, scenarios)

    # file_handler = file_handling.FileHandler(scenario_name)
    # file_handler.export_multiple_platoon_results_to_cloud(
    #         vehicle_percentages, vehicle_inputs, strategies,
    #         orig_and_dest_lane_speeds)
    # file_handler.import_multiple_results_from_cloud(lc_scenarios)
    # file_handler.import_multiple_results_from_cloud(no_lc_scenarios)

    # =============== To MOVES =============== #
    # scenarios = file_handling.create_multiple_scenarios(
    #     full_penetration, inputs_per_lane)
    # for sc in scenarios:
    #     moves_file_handling.get_individual_vehicle_trajectories_to_moves(
    #         scenario_name, sc, vehs_per_simulation=3)
    # scenarios = scenario_handling.get_platoon_lane_change_scenarios(
    #     "dest_lane_speed")
    # moves_file_handling.platoon_scenario_to_moves(scenarios)

    # =============== Check results graphically =============== #

    # all_plots_for_scenarios_with_risk(scenario_name, save_fig=False)
    # all_plots_for_scenarios_with_risk_and_varying_penetration(scenario_name,
    #                                                           True)


if __name__ == "__main__":
    main()
