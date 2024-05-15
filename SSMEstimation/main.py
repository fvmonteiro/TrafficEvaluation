import matplotlib.pyplot as plt

import data_writer
import file_handling
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
    file_handler = file_handling.FileHandler(scenario_name)
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
    file_handler = file_handling.FileHandler(scenario_name)
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
    file_handler = file_handling.FileHandler(scenario_name)
    try:
        file_handler.export_multiple_results_to_cloud(
            full_penetration_scenarios)
    except FileNotFoundError:
        print("Couldn't copy files to shared folder.")


def run_platoon_scenarios():
    scenario_name = "platoon_discretionary_lane_change"
    lc_scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "dest_lane_speed", True, False)
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


def run_platoon_warm_up():
    scenario_name = "platoon_discretionary_lane_change"
    other_vehicles = [{VehicleType.HDV: 100}]
    strategy = [PlatoonLaneChangeStrategy.graph_min_time]
    # vehicles_per_lane = [1000]
    vehicles_per_lane = (
        scenario_handling.all_platoon_simulation_configurations[
            "vehicles_per_lane"]
    )
    # orig_and_dest_lane_speeds = [("70", "50")]
    orig_and_dest_lane_speeds = (
        scenario_handling.all_platoon_simulation_configurations[
            "orig_and_dest_lane_speeds"])
    platoon_size = [2]
    # platoon_size = (
    #     scenario_handling.all_platoon_simulation_configurations["platoon_size"])
    scenarios = scenario_handling.create_multiple_scenarios(
        other_vehicles, vehicles_per_lane, lane_change_strategies=strategy,
        orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
        platoon_size=platoon_size, special_cases=["warmup"])
    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    vi.run_multiple_platoon_lane_change_scenarios(scenarios)


def run_a_platoon_simulation():
    scenario_name = "platoon_discretionary_lane_change"
    other_vehicles = {VehicleType.HDV: 100}
    strategy = PlatoonLaneChangeStrategy.graph_min_time
    scenario = scenario_handling.ScenarioInfo(
        other_vehicles, 500, platoon_lane_change_strategy=strategy,
        orig_and_dest_lane_speeds=("70", "50"), platoon_size=2,
        special_case="warmup")
    vi = vissim_interface.VissimInterface()
    vi.load_simulation(scenario_name)
    # vi.set_random_seed(8)
    # vi.set_logged_vehicle_id(280)
    vi.run_platoon_scenario_sample(scenario, number_of_runs=1,
                                   is_simulation_verbose=True)
    vi.close_vissim()


def main():
    # plots_for_platoon_scenarios(False)
    # plot_comparison_scenario(False)
    # =============== Scenario Definition =============== #
    scenario_name = "platoon_discretionary_lane_change"

    # =============== Running =============== #

    # run_a_platoon_simulation()
    # run_platoon_scenarios()
    # run_a_platoon_simulation()
    # run_platoon_warm_up()

    scenarios = scenario_handling.get_lane_change_scenarios_graph_paper()
    vi = VissimInterface()
    vi.load_simulation(scenario_name)
    vi.run_multiple_platoon_lane_change_scenarios(
        scenarios, runs_per_scenario=3, is_debugging=False)
    vi.close_vissim()

    # =============== Post processing =============== #
    # vehicle_types = [VehicleType.AUTONOMOUS, VehicleType.CONNECTED]
    # percentages = [0, 100]
    # vehicle_percentages = (
    #     scenario_handling.create_vehicle_percentages_dictionary(
    #     vehicle_types, percentages, 1))
    # inputs_per_lane = [2000]
    # risks = [i for i in range(0, 31, 10)]
    # scenarios = file_handling.create_multiple_scenarios(
    #     vehicle_percentages, inputs_per_lane, risks)
    # post_processing.create_summary_with_risks(scenario_name, scenarios)
    # post_processing.create_platoon_lane_change_summary(
    #     scenario_name, scenarios)

    post_processing.create_platoon_lane_change_summary(
        scenario_name, scenarios)

    # file_handler = file_handling.FileHandler(scenario_name)
    # file_handler.export_multiple_results_to_cloud(scenarios)
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
    result_analysis.plots_for_graph_paper(False)
    # all_plots_for_scenarios_with_risk(scenario_name, save_fig=False)
    # all_plots_for_scenarios_with_risk_and_varying_penetration(scenario_name,
    #                                                           True)


if __name__ == "__main__":
    main()
