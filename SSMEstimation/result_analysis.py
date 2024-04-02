import os
from collections.abc import Iterable
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from file_handling import FileHandler
import post_processing
import readers
import scenario_handling
from vehicle import VehicleType, Vehicle


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
    result_analyzer = ResultAnalyzer(scenario_name, save_results)
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
    result_analyzer = ResultAnalyzer(scenario_name, save_results)
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
    result_analyzer = ResultAnalyzer(network_name, save_results)
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

    result_analyzer = ResultAnalyzer(scenario_name, save_results)
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
    result_analyzer = ResultAnalyzer(network_name, save_results)
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

    ra = ResultAnalyzer(network_name,
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
    ra = ResultAnalyzer(network_name,
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
    ra = ResultAnalyzer(scenario_name, should_save_fig,
                        is_debugging=False)

    cav_scenarios = scenario_handling.get_platoon_lane_change_scenarios(
        "dest_lane_speed", with_hdv=True)
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


def plots_for_graph_paper(should_save_fig: bool = False):
    scenario_name = "platoon_discretionary_lane_change"
    ra = ResultAnalyzer(scenario_name, should_save_fig,
                        is_debugging=False)

    scenarios = scenario_handling.get_lane_change_scenarios_graph_paper()
    # ra.print_comparative_costs(scenarios)
    ra.plot_comparison_to_LVF(scenarios)
    # ra.plot_successful_maneuvers(scenarios)
    # ra.compare_travel_times(scenarios, x="delta_v",
    #                         plot_cols="vehicles_per_lane",
    #                         warmup_time=1, sim_time=11)


class ResultAnalyzer:
    _data_reader_map = {
        "flow": readers.DataCollectionReader,
        "vehicle_count": readers.DataCollectionReader,
        "density": readers.LinkEvaluationReader,
        "volume": readers.LinkEvaluationReader,
        "delay_relative": readers.LinkEvaluationReader,
        "average_speed": readers.LinkEvaluationReader,
        "risk": readers.SSMDataReader,
        "barrier_function_risk": readers.SSMDataReader,
        "total_risk": readers.RiskyManeuverReader,
        "discomfort": readers.DiscomfortReader,
        "violations": readers.ViolationsReader,
        "lane_change_count": readers.LaneChangeReader,
        "total_lane_change_risk": readers.LaneChangeReader,
        "initial_risk": readers.LaneChangeReader,
        "initial_risk_to_lo": readers.LaneChangeReader,
        "initial_risk_to_ld": readers.LaneChangeReader,
        "initial_risk_to_fd": readers.LaneChangeReader,
        "fd_discomfort": readers.LaneChangeReader,
        "lane_change_issues": readers.LaneChangeIssuesReader,
        "emission": readers.MOVESDatabaseReader,
        "emission_per_volume": readers.MOVESDatabaseReader,
        "lane_change_completed": readers.PlatoonLaneChangeEfficiencyReader,
        "platoon_maneuver_time": readers.PlatoonLaneChangeEfficiencyReader,
        "platoon_travel_time": readers.PlatoonLaneChangeEfficiencyReader,
        "accel_cost": readers.PlatoonLaneChangeEfficiencyReader,
        "dist_cost": readers.PlatoonLaneChangeEfficiencyReader,
        "stayed_in_platoon": readers.PlatoonLaneChangeEfficiencyReader,
        "travel_time": readers.PlatoonLaneChangeImpactsReader
    }

    _ssm_pretty_name_map = {"low_TTC": "Low TTC",
                            "high_DRAC": "High DRAC",
                            "CPI": "CPI",
                            "risk": "CRI"}

    _pollutant_id_to_string = {
        1: "Gaseous Hydrocarbons", 5: "Methane (CH4)",
        6: "Nitrous Oxide (N2O)", 90: "CO2",
        91: "Energy Consumption", 98: "CO2 Equivalent",
    }

    def __init__(self, scenario_name: str, should_save_fig: bool = False,
                 is_debugging: bool = False):
        if os.environ["COMPUTERNAME"] == "DESKTOP-626HHGI":
            self._figure_folder = ("C:\\Users\\fvall\\Google Drive\\"
                                   "PhD Research\\Lane Change\\images")
        else:
            self._figure_folder = ("G:\\My Drive\\PhD Research\\Lane Change"
                                   "\\images")
        self.file_handler = FileHandler(scenario_name)
        self.scenario_name = scenario_name
        self.should_save_fig = should_save_fig
        self.is_debugging = is_debugging
        # Default params
        sns.set_style("whitegrid")
        plt.rc("font", size=20)

    # Plots aggregating results from multiple simulations ==================== #
    def plot_xy(self, x: str, y: str, hue: str,
                scenarios: list[scenario_handling.ScenarioInfo],
                warmup_time: float = 0) -> plt.Axes:
        """

        """
        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        # Plot
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue, ci="sd")
        plt.show()
        return ax

    def plot_y_vs_time(self, y: str,
                       scenarios: list[scenario_handling.ScenarioInfo],
                       warmup_time: float = 10) -> plt.Axes:
        """Plots averaged y over several runs with the same vehicle input
        versus time.fi

        :param y: name of the variable being plotted.
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: must be given in minutes. Samples before start_time
         are ignored.
         """

        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        # ResultAnalyzer._check_if_data_is_uniform(data)
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        ax = sns.lineplot(data=data, x="time", y=y,
                          hue="simulation_number", errorbar="sd")
        if self.should_save_fig:
            self.save_fig(plt.gcf(), "time_plot", y, scenarios)
        plt.show()
        return ax

    def plot_fundamental_diagram(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            link_number: int,
            link_segment_number: int = None, lanes: list[int] = None,
            hue: str = None, col: str = None,
            aggregation_period: int = 30, warmup_time: float = 10,
            will_plot: bool = False) -> list[plt.Axes]:
        """
        Computes flow from link evaluation data.
        """
        plt.rc("font", size=15)
        aggregate_lanes = True
        if hue == "lane" or col == "lane":
            aggregate_lanes = False

        link_eval_data = self._load_data(
            "density", scenarios)
        link_eval_data = self._prepare_link_evaluation_data(
            link_eval_data, link_number, link_segment_number, lanes,
            aggregate_lanes=aggregate_lanes, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # ResultAnalyzer._check_if_data_is_uniform(link_eval_data)
        if col is None:
            ax = sns.scatterplot(data=link_eval_data, x="density", y="volume",
                                 hue=hue, palette="tab10")
            ax.set_xlabel("density (veh/km)")
            ax.set_ylabel("flow (veh/h)")
            axes = [ax]
        else:
            rp = sns.relplot(data=link_eval_data, x="density", y="volume",
                             hue=hue, col=col, col_wrap=2,
                             palette="tab10")
            axes = []
            for ax in rp.axes:
                ax.set_xlabel("density (veh/km)")
                ax.set_ylabel("flow (veh/h)")
                axes.append(ax)

        if will_plot:
            plt.tight_layout()
            plt.show()

        return axes

    def plot_fundamental_diagram_from_flow(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            flow_sensor_identifier: Union[str, int] = None,
            link_segment_number: int = None,
            aggregation_period: int = 30, warmup_time: float = 10) -> None:
        """
        Computes flow from data collection sensors and uses density from link
        evaluation.
        """

        density_data = self._load_data("density", scenarios)
        flow_data = self._load_data("flow", scenarios)

        flow_data = self._prepare_data_collection_data(
            flow_data, flow_sensor_identifier, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # We"re assuming scenarios with a single main link
        link = self.file_handler.get_main_links()[0]
        density_data = self._prepare_link_evaluation_data(
            density_data, link, link_segment_number,
            warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        intersection_columns = ["vehicles_per_lane", "control percentages",
                                "simulation_number", "time_interval",
                                "random_seed"]
        data = flow_data.merge(density_data, on=intersection_columns)

        # ResultAnalyzer._check_if_data_is_uniform(data)
        sns.scatterplot(data=data, x="density", y="flow",
                        hue="control percentages")
        plt.show()

    def plot_fundamental_diagram_per_control_percentage(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, aggregation_period: int = 30) -> None:
        main_link = self.file_handler.get_main_links()[0]
        self.plot_fundamental_diagram(
            scenarios, main_link, hue="control percentages",
            aggregation_period=aggregation_period,
            warmup_time=warmup_time)
        plt.tight_layout()
        plt.show()

    def plot_flow_box_plot_vs_controlled_percentage(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10,
            flow_sensor_name: str = "in",
            aggregation_period: int = 30) -> None:

        # for sc in scenarios:
        #     if sc.accepted_risk is None:
        #         sc.accepted_risk = 0
        data = self._load_data("flow", scenarios)
        relevant_data = self._prepare_data_collection_data(
            data, flow_sensor_name, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # ResultAnalyzer._check_if_data_is_uniform(relevant_data)
        fig, ax = _my_boxplot(relevant_data, "control percentages", "flow",
                              "vehicles per hour", will_show=False)
        ax.set_ylabel("Flow (vehs/h)")
        fig.set_size_inches(15, 6)
        fig.tight_layout()
        fig.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name("box_plot", "flow", scenarios)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(relevant_data, "flow", scenarios, np.median,
                                show_variation=True)

    def plot_link_data_box_plot_vs_controlled_percentage(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str = None, lanes: str = None,
            link_segment_number: int = None,
            warmup_time: float = 10, aggregation_period: int = 30) -> None:

        link, lane_numbers = self._select_link_and_lanes(
            before_or_after_lc_point, lanes)
        data = self._load_data(y, scenarios)
        relevant_data = self._prepare_link_evaluation_data(
            data, link, link_segment_number, lane_numbers,
            warmup_time=warmup_time, aggregation_period=aggregation_period)
        fig, ax = _my_boxplot(relevant_data, "control percentages", y,
                              "vehicles_per_lane", will_show=False)
        ax.legend(loc="upper center", bbox_to_anchor=(1.1, 1.1),
                  title="orig lane vehs/hour", ncols=1)
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name("box_plot", y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(relevant_data, y, scenarios, np.median,
                                show_variation=True)

    def box_plot_y_vs_controlled_percentage(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        """Plots averaged y over several runs with the same vehicle input
        versus controlled vehicles percentage as a box plot.

        :param y: name of the variable being plotted.
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        """

        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        # ResultAnalyzer._check_if_data_is_uniform(data)
        fig, ax = _my_boxplot(data, "control percentages", y,
                              "vehicles per hour")
        if self.should_save_fig:
            print("Must double check whether fig is being saved")
            self.save_fig(fig, "box_plot", y, scenarios)

    def box_plot_y_vs_vehicle_type(
            self, y: str, hue: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, sensor_name: str = None) -> None:
        """
        Plots averaged y over several runs with the same vehicle input
        versus vehicles type as a box plot. Parameter hue controls how to
        split data for each vehicle type.

        :param y: name of the variable being plotted.
        :param hue: must be "percentage" or "accepted_risk"
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param sensor_name: if plotting flow, we can determine choose
         which data collection measurement will be shown
        """
        raw_data = self._load_data(y, scenarios)

        # We do it per group because some set of simulations might have
        # different parameters. The processing operations make them uniform
        data_list = []
        for _, group in raw_data.groupby("control percentages"):
            data_list.append(self._prepare_data_for_plotting(
                y, group, warmup_time, sensor_name))
        data = pd.concat(data_list).reset_index()
        # ResultAnalyzer._check_if_data_is_uniform(data)

        no_control_idx = (data["control percentages"]
                          == "100% HDV")
        data[["percentage", "control_type"]] = data[
            "control percentages"].str.split(" ", expand=True)
        data.loc[no_control_idx, "control_type"] = "HDV"
        data.loc[no_control_idx, "percentage"] = "0%"
        data.loc[no_control_idx, "accepted_risk"] = 0
        data["Accepted Risk"] = data["accepted_risk"].map(
            {0: "safe", 10: "low", 20: "medium", 30: "high"}
        )
        if hue == "accepted_risk":
            hue = "Accepted Risk"

        # Plot
        plt.rc("font", size=25)
        sns.boxplot(data=data, x="control_type", y=y, hue=hue)

        # Direct output
        _produce_console_output(data, y, scenarios, np.median)

        plt.legend(title=hue, ncol=1,
                   bbox_to_anchor=(1.01, 1))
        # plt.title(str(vehicles_per_lane) + " vehs/h/lane", fontsize=22)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        if self.should_save_fig:
            self.save_fig(fig, "box_plot", y, scenarios)
        # self.widen_fig(plt.gcf(), len(percentages_per_vehicle_types))
        plt.tight_layout()
        plt.show()

    def plot_risk_histograms(
            self, risk_type: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, min_risk: float = 0.1
    ) -> None:
        """
        Plot one histogram of lane change risks for each vehicle percentage
        each single risk value

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type, scenarios)
        # ResultAnalyzer._check_if_data_is_uniform(data)
        data = self._prepare_risk_data(data, risk_type, warmup_time, min_risk)
        plt.rc("font", size=28)
        if ("accepted_risk" not in data.columns
                or data["accepted_risk"].nunique() <= 1):
            grouped_data = data.groupby("control percentages")
        else:
            grouped_data = data.groupby(["control percentages",
                                         "accepted_risk"])

        scenarios_per_vp = scenario_handling.split_scenario_by(
            scenarios, "vehicle_percentages")
        for group_name, data_to_plot in grouped_data:
            if data_to_plot.empty:
                continue
            ax = sns.histplot(data=data_to_plot, x=risk_type,
                              stat="count", hue="vehicles per hour",
                              palette="tab10")
            ax.set_xlabel(risk_type.replace("_", " "))
            ax.set_title(group_name)
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.tight_layout()
            plt.show()
            if self.should_save_fig:
                fig_name = self.create_figure_name(
                    "histogram", risk_type, scenarios_per_vp[group_name])
                self.save_fig(fig, fig_name=fig_name)

        normalizer = data.loc[
            (data["control percentages"] == "100% HDV")
            & (data["vehicles per hour"] == data[
                "vehicles per hour"].min()),
            risk_type].sum()
        _produce_console_output(data, risk_type, scenarios, [np.size, np.sum],
                                show_variation=True,
                                normalizer=normalizer)

    def plot_lane_change_risk_histograms_risk_as_hue(
            self, risk_type: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, min_risk: int = 0.1
    ) -> None:
        """
        Plot one histogram of lane change risks for each vehicle percentage.
        All risk values on the same plot

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type, scenarios)
        data = self._prepare_risk_data(data, risk_type, warmup_time, min_risk)
        data["risk"] = data["accepted_risk"].map({0: "safe", 10: "low",
                                                  20: "medium", 30: "high"})
        # ResultAnalyzer._check_if_data_is_uniform(data)

        plt.rc("font", size=30)
        # sns.set(font_scale=2)
        # for cp in control_percentages:
        for sc in scenarios:
            cp = vehicle_percentage_dict_to_string(sc.vehicle_percentages)
            if cp == "100% HDV":
                continue
            data_to_plot = data[
                (data["control percentages"] == cp)
            ]
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat="count", hue="risk",
                         palette="tab10")
            # plt.legend(title="Risk", labels=["safe", "low", "medium"])
            # Direct output
            print("veh penetration: {}".format(cp))
            print(data_to_plot[["control percentages", "veh_type", "risk",
                                risk_type]].groupby(
                ["control percentages", "veh_type", "risk"]).count())
            print(data_to_plot.groupby(
                ["control percentages", "risk"])[risk_type].median())
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            if self.should_save_fig:
                self.save_fig(fig, "histogram", risk_type, [sc])
            plt.show()

    def plot_row_of_lane_change_risk_histograms(
            self, risk_type: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, min_risk: int = 0.1
    ) -> None:
        """
        Plots histograms of lane change risks for a single penetration
        percentage and varied risk
        :param risk_type:
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type, scenarios)
        data = ResultAnalyzer._prepare_risk_data(data, risk_type, warmup_time,
                                                 min_risk)
        # ResultAnalyzer._check_if_data_is_uniform(data)

        all_accepted_risks = [sc.accepted_risk for sc in scenarios]

        n_risks = len(all_accepted_risks)
        fig, axes = plt.subplots(nrows=1, ncols=n_risks)
        fig.set_size_inches(12, 6)
        j = 0
        for ar, data_to_plot in data.groupby("accepted_risk"):
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat="count", hue="vehicles_per_lane",
                         palette="tab10", ax=axes[j])
            axes[j].set_title("accepted risk = " + str(ar))
            j += 1

        plt.tight_layout()
        if self.should_save_fig:
            self.save_fig(fig, "histogram_row_", risk_type, scenarios)
        plt.show()

    def plot_grid_of_lane_change_risk_histograms(
            self, risk_type: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, min_risk: int = 0.1
    ) -> None:
        """
        Creates a grid of size "# of vehicle types" vs "# of accepted risks"
        and plots the histogram of lane changes" total risks for each case

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type, scenarios)
        data = ResultAnalyzer._prepare_risk_data(data, risk_type, warmup_time,
                                                 min_risk)
        # ResultAnalyzer._check_if_data_is_uniform(data)

        control_percentages = data["control percentages"].unique()
        accepted_risks = data["accepted_risk"].unique()
        n_ctrl_pctgs = len(control_percentages)
        n_risks = len(accepted_risks)
        fig, axes = plt.subplots(nrows=n_ctrl_pctgs, ncols=n_risks)
        fig.set_size_inches(12, 6)
        # TODO: groupby
        for i in range(n_ctrl_pctgs):
            for j in range(n_risks):
                ctrl_percentage = control_percentages[i]
                ar = accepted_risks[j]
                data_to_plot = data[
                    (data["control percentages"] == ctrl_percentage)
                    & (data["accepted_risk"] == ar)
                    ]
                sns.histplot(data=data_to_plot, x=risk_type,
                             stat="count", hue="vehicles_per_lane",
                             palette="tab10", ax=axes[i, j])
                if i + j > 0 and axes[i, j].get_legend():
                    axes[i, j].get_legend().remove()
                if i == 0:
                    axes[i, j].set_title("accepted risk = " + str(ar))

        plt.tight_layout()
        if self.should_save_fig:
            self.save_fig(fig, "histogram_grid", risk_type, scenarios)
        plt.show()

    def hist_plot_lane_change_initial_risks(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 0) -> None:
        """

        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :return:
        """
        # lane_change_reader = readers.LaneChangeReader(self.scenario_name)
        # if scenarios is None:
        #     data = lane_change_reader.load_test_data()
        # else:
        #     data = lane_change_reader.load_data_from_several_scenarios(
        #         scenarios)
        data = self._load_data("initial_risk", scenarios)
        warmup_time *= 60  # minutes to seconds
        data.drop(index=data[data["start_time"] < warmup_time].index,
                  inplace=True)

        grouped = data.groupby("control percentages")
        # for control_percentage in data["control percentages"].unique():
        for control_percentage, data_to_plot in grouped:
            # data_to_plot = data[data["control percentages"]
            #                     == control_percentage]
            plt.rc("font", size=15)
            for veh_name in ["lo", "ld", "fd"]:
                sns.histplot(data=data_to_plot, x="initial_risk_to_" + veh_name,
                             stat="percent", hue="mandatory",
                             palette="tab10")
                plt.tight_layout()
                if self.should_save_fig:
                    fig = plt.gcf()
                    fig.set_dpi(200)
                    vehicles_per_lane_str = "_".join(
                        str(v) for v
                        in data_to_plot["vehicles_per_lane"].unique()
                    ) + "vehs_per_lane"
                    fig_name = "_".join(
                        ["initial_lane_change_risks", vehicles_per_lane_str,
                         str(control_percentage).replace("% ", "_")])
                    fig.savefig(os.path.join(self._figure_folder, fig_name))
                plt.show()

    def plot_heatmap_risk_vs_control(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, normalize: bool = False
    ) -> None:
        """

        :param y:
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: in minutes
        :param normalize:
        :return:
        """
        y_plotting_options = {
            "lane_change_count": {"title": "# lane changes",
                                  "aggregation_function": np.count_nonzero,
                                  "col_in_df": "veh_id"},
            "vehicle_count": {"title": "Output flow",
                              "aggregation_function": np.sum,
                              "col_in_df": y},
            "total_lane_change_risk": {"title": "Total risk",
                                       "aggregation_function": np.mean,
                                       "col_in_df": y},
            "initial_risk": {"title": "Initial risk",
                             "aggregation_function": np.mean,
                             "col_in_df": y}
        }

        col_in_df = y_plotting_options[y]["col_in_df"]
        aggregation_function = y_plotting_options[y]["aggregation_function"]
        title = y_plotting_options[y]["title"]

        plt.rc("font", size=17)
        data = self._load_data(y, scenarios)
        data = self._prepare_data_for_plotting(y, data, warmup_time,
                                               sensor_name=["out"])
        data.loc[
            data["control percentages"] == "100% HDV",
            "control percentages"] = "100% HD"
        # for vpl in vehicles_per_lane:
        for vpl, data_to_plot in data.groupby("vehicles_per_lane"):
            n_simulations = data_to_plot["simulation_number"].nunique()
            table = data_to_plot.pivot_table(values=col_in_df,
                                             index=["accepted_risk"],
                                             columns=["control percentages"],
                                             aggfunc=aggregation_function)
            if "count" in y:
                table /= n_simulations
            no_avs_no_risk_value = table.loc[0, "100% HD"]
            print("Base value (humans drivers): ", no_avs_no_risk_value)
            # Necessary because simulations without AVs only have LC values
            # with zero accepted risk.
            table.fillna(value=no_avs_no_risk_value, inplace=True)
            if normalize:
                table /= no_avs_no_risk_value
            max_table_value = np.round(np.nanmax(table.to_numpy()))
            fmt = ".2f" if max_table_value < 100 else ".0f"
            sns.heatmap(table.sort_index(axis=0, ascending=False),
                        annot=True, fmt=fmt,
                        xticklabels=table.columns.get_level_values(
                            "control percentages"))
            plt.xlabel("", fontsize=22)
            plt.ylabel("accepted initial risk", fontsize=22)
            plt.title(title + " at " + str(vpl) + " vehs/h/lane",
                      fontsize=22)
            plt.tight_layout()
            if self.should_save_fig:
                scenarios_subset = [sc for sc in scenarios
                                    if sc.vehicles_per_lane == vpl]
                self.save_fig(plt.gcf(), "heatmap", y, scenarios_subset)
            plt.show()

    def plot_risk_heatmap(
            self, risk_type: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            normalizer: float = None) -> float:
        """
        Plots a heatmap of the chosen risk type. Returns normalizer so that
        we can reuse the value again if needed.
        :param risk_type: total_risk, total_lane_change_risk or both. If
         both, the lane change risk is normalized using the values from
         total_risk
        :param scenarios: List of simulation parameters for several scenarios
        :param normalizer: All values are divided by the normalizer
        :returns: The value used to normalize all values
        """
        for sc in scenarios:
            if sc.accepted_risk is None:
                sc.accepted_risk = [0]

        if risk_type == "both":
            normalizer = self.plot_risk_heatmap("total_risk", scenarios,
                                                normalizer)
            self.plot_risk_heatmap("total_lane_change_risk", scenarios,
                                   normalizer)
            return normalizer

        data = self._load_data(risk_type, scenarios)
        warmup_time = 10
        post_processing.drop_warmup_samples(data, warmup_time)
        no_control_idx = (data["control percentages"]
                          == "100% HDV")
        data[["percentage", "Vehicle Type"]] = data[
            "control percentages"].str.split(" ", expand=True)
        data.loc[no_control_idx, "Vehicle Type"] = "HDV"
        data.loc[no_control_idx, "percentage"] = "0%"
        data.loc[no_control_idx, "accepted_risk"] = 0
        data["Accepted Risk"] = data["accepted_risk"].map(
            {0: "safe", 10: "low", 20: "medium", 30: "high"}
        )

        agg_function = np.sum
        if normalizer is None:
            normalizer = data.loc[
                (data["control percentages"] == "100% HDV")
                & (data["vehicles per hour"] == data[
                    "vehicles per hour"].min()),
                risk_type].sum()
        title = "Normalized " + " ".join(risk_type.split("_")[1:])
        fig, ax = _plot_heatmap(data, risk_type, "Vehicle Type",
                                "Accepted Risk", normalizer,
                                agg_function=agg_function, show_figure=False,
                                fill_na_cols=True)
        ax.set_title(title.title())  # title() capitalizes the first letter
        plt.yticks(rotation=0)
        fig.set_size_inches(6, 3)
        fig.tight_layout()
        fig.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "heatmap", risk_type, scenarios)
            self.save_fig(fig, fig_name=fig_name)
        return normalizer

    def plot_pointplot_vs_accepted_risk(self, y, scenarios, sensor_name=None,
                                        warmup_time=10) -> None:
        raw_data = self._load_data(y, scenarios)

        # We do it per group because some set of simulations might have
        # different parameters. The processing operations make them uniform
        data_list = []
        for _, group in raw_data.groupby("control percentages"):
            data_list.append(self._prepare_data_for_plotting(
                y, group, warmup_time, sensor_name))
        data = pd.concat(data_list).reset_index()
        no_control_idx = (data["control percentages"]
                          == "100% HDV")
        data[["percentage", "Vehicle Type"]] = data[
            "control percentages"].str.split(" ", expand=True)
        data.loc[no_control_idx, "Vehicle Type"] = "HDV"
        data.loc[no_control_idx, "percentage"] = "0%"

        # Repeat hdv data for all risks
        hdv_repeated_data_list = []
        data.loc[no_control_idx, "accepted_risk"] = 0
        for risk_value in data["accepted_risk"].unique():
            if risk_value == 0:
                continue
            temp = data.loc[no_control_idx].copy()
            temp["accepted_risk"] = risk_value
            hdv_repeated_data_list.append(temp)
        # hdv_repeated_data = pd.concat(hdv_repeated_data_list)
        full_data = pd.concat([data] + hdv_repeated_data_list)
        full_data["Accepted Risk"] = full_data["accepted_risk"].map(
            {0: "safe", 10: "low", 20: "medium", 30: "high"}
        )

        # Plot
        fig = plt.figure()
        plt.rc("font", size=20)
        ax = sns.pointplot(full_data, x="Accepted Risk", y=y,
                           hue="Vehicle Type",
                           hue_order=["HDV", "AV", "CAV"], dodge=True
                           )
        # sns.move_legend(ax, loc="lower left", bbox_to_anchor=(0, 1.1),
        #                 ncols=3)
        if "risk" in y:
            y = "lane_change_risk"
        y_label = _make_pretty_label(y)
        ax.set_ylabel(y_label)
        fig.set_size_inches(8, 4)
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "pointplot", y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_barplot_vs_accepted_risk(self, y, scenarios, sensor_name=None,
                                      warmup_time=10) -> None:
        raw_data = self._load_data(y, scenarios)

        # We do it per group because some set of simulations might have
        # different parameters. The processing operations make them uniform
        data_list = []
        for _, group in raw_data.groupby("control percentages"):
            data_list.append(self._prepare_data_for_plotting(
                y, group, warmup_time, sensor_name))
        data = pd.concat(data_list).reset_index()
        no_control_idx = (data["control percentages"]
                          == "100% HDV")

        # Repeat hdv data for all risks
        hdv_repeated_data_list = []
        data.loc[no_control_idx, "accepted_risk"] = 0
        for risk_value in data["accepted_risk"].unique():
            if risk_value == 0:
                continue
            temp = data.loc[no_control_idx].copy()
            temp["accepted_risk"] = risk_value
            hdv_repeated_data_list.append(temp)
        # hdv_repeated_data = pd.concat(hdv_repeated_data_list)
        full_data = pd.concat([data] + hdv_repeated_data_list)
        full_data["Accepted Risk"] = full_data["accepted_risk"].map(
            {0: "safe", 10: "low", 20: "medium", 30: "high"}
        )

        # Plot
        fig = plt.figure()
        plt.rc("font", size=20)
        if "connected_percentage" in full_data.columns:
            percentage_column = "connected_percentage"
            new_name = "% CAV"
        elif "autonomous_percentage" in full_data.columns:
            percentage_column = "autonomous_percentage"
            new_name = "% AV"
        else:
            percentage_column = "control percentage"
            new_name = percentage_column
        full_data.rename(columns={percentage_column: new_name}, inplace=True)
        ax = sns.barplot(full_data, hue=new_name, y=y,
                         x="Accepted Risk", dodge=True
                         )
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
        if "risk" in y:
            y = "lane_change_risk"
        y_label = _make_pretty_label(y)
        ax.set_ylabel(y_label)
        fig.set_size_inches(8, 4)
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name("barplot", y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_fd_discomfort(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            brake_threshold: int = 4) -> None:
        y = "fd_discomfort"
        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, 10)
        col_name = "_".join([y, str(brake_threshold)])
        agg_function = np.mean
        normalizer = data.loc[
            (data["control percentages"] == "100% HDV")
            & (data["vehicles per hour"] == data[
                "vehicles per hour"].min()),
            col_name].agg(agg_function)
        # normalizer = 1
        fig, ax = _plot_heatmap(data, col_name, "vehicles per hour",
                                "control percentages", normalizer,
                                agg_function=agg_function,
                                custom_colorbar_range=True, show_figure=False)
        ax.set_title("Dest Lane Follower Discomfort")
        if self.should_save_fig:
            fig_name = self.create_figure_name("heatmap", col_name, scenarios)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(data, col_name, scenarios, agg_function,
                                show_variation=True)

    def plot_discomfort_heatmap(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            max_brake: int = 4) -> None:
        data = self._load_data("discomfort", scenarios)
        y = "discomfort_" + str(max_brake)
        post_processing.drop_warmup_samples(data, 10)
        normalizer = data.loc[
            (data["control percentages"] == "100% HDV")
            & (data["vehicles_per_lane"] == data["vehicles_per_lane"].min()),
            y].sum()
        fig, _ = _plot_heatmap(data, y, "vehicles_per_lane",
                               "control percentages", normalizer)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "heatmap", y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_total_output_heatmap(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        y = "vehicle_count"
        data = self._load_data("vehicle_count", scenarios)
        data = self._prepare_data_collection_data(data, sensor_identifier="out",
                                                  warmup_time=10)
        normalizer = data.loc[
            (data["control percentages"] == "100% HDV")
            & (data["vehicles_per_lane"] == data[
                "vehicles_per_lane"].min()),
            y].sum()
        fig, ax = _plot_heatmap(data, "vehicle_count", "vehicles_per_lane",
                                "control percentages", normalizer,
                                show_figure=False, fill_na_cols=True)
        ax.set_title("Output flow")
        fig.tight_layout()
        fig.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "heatmap", "vehicle_count", scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_total_output_heatmap_vs_risk(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        y = "vehicle_count"
        data = self._load_data("vehicle_count", scenarios)
        data = self._prepare_data_collection_data(data, sensor_identifier="out",
                                                  warmup_time=10)

        no_control_idx = (data["control percentages"]
                          == "100% HDV")
        data[["percentage", "Vehicle Type"]] = data[
            "control percentages"].str.split(" ", expand=True)
        data.loc[no_control_idx, "Vehicle Type"] = "HDV"
        data.loc[no_control_idx, "percentage"] = "0%"
        data.loc[no_control_idx, "accepted_risk"] = 0
        data["Accepted Risk"] = data["accepted_risk"].map(
            {0: "safe", 10: "low", 20: "medium", 30: "high"}
        )

        normalizer = data.loc[
            (data["control percentages"] == "100% HDV")
            & (data["vehicles_per_lane"] == data[
                "vehicles_per_lane"].min()),
            y].sum()
        fig, ax = _plot_heatmap(data, "vehicle_count", "Vehicle Type",
                                "Accepted Risk", normalizer,
                                show_figure=False, fill_na_cols=True)
        ax.set_title("Output flow")
        fig.tight_layout()
        fig.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "heatmap", "vehicle_count", scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_emission_heatmap(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            pollutant_id: int = 91
    ) -> None:
        y = "emission_per_volume"
        data = self._load_data(y, scenarios)
        data = ResultAnalyzer._prepare_pollutant_data(data, pollutant_id)
        title = "Normalized " + self._pollutant_id_to_string[pollutant_id]

        normalizer = data.loc[
            (data["control percentages"] == "100% HDV")
            & (data["vehicles per hour"]
               == data["vehicles per hour"].min()),
            y].sum()

        fig, ax = _plot_heatmap(data, y, "vehicles per hour",
                                "control percentages",
                                normalizer, show_figure=False)
        ax.set_title(title)
        fig.tight_layout()
        fig.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "heatmap", self._pollutant_id_to_string[pollutant_id],
                scenarios)
            self.save_fig(fig, fig_name=fig_name)
        _produce_console_output(data, y, scenarios,
                                np.sum, show_variation=True)

    def plot_lane_change_count_heatmap(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        y = "lane_change_count"
        agg_function = np.count_nonzero
        col_to_count = "veh_id"

        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        n_simulations = data["simulation_number"].nunique()
        fig, ax = _plot_heatmap(data, col_to_count, "vehicles per hour",
                                "control percentages", normalizer=n_simulations,
                                agg_function=agg_function)
        ax.set_title("lane change count")
        fig.tight_layout()
        fig.show()
        _produce_console_output(data, col_to_count, scenarios, agg_function)

    def plot_lane_change_count_heatmap_per_risk(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        y = "lane_change_count"
        agg_function = np.count_nonzero
        col_to_count = "veh_id"

        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        no_control_idx = (data["control percentages"]
                          == "100% HDV")
        data[["percentage", "control_type"]] = data[
            "control percentages"].str.split(" ", expand=True)
        data.loc[no_control_idx, "control_type"] = "human"
        data.loc[no_control_idx, "percentage"] = "0%"
        data.loc[no_control_idx, "accepted_risk"] = 0
        data["Accepted Risk"] = data["accepted_risk"].map(
            {0: "safe", 10: "low", 20: "medium", 30: "high"}
        )

        n_simulations = data["simulation_number"].nunique()
        fig, ax = _plot_heatmap(data, col_to_count, "control_type",
                                "Accepted Risk",
                                normalizer=n_simulations,
                                agg_function=agg_function, show_figure=False)
        ax.set_title("lane change count")
        fig.tight_layout()
        fig.show()
        _produce_console_output(data, col_to_count, scenarios, agg_function)

    def print_unfinished_lane_changes_for_risky_scenario(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        data = self._load_data("lane_change_issues",
                               scenarios)
        # n_simulations = data["simulation_number"].nunique()
        print(data.groupby(
            ["vehicles_per_lane", "control percentages",
             "accepted_risk"])["percent unfinished"].mean())

    def print_summary_of_issues(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        """
        Prints out a summary with average number of times the AV requested
        human intervention and average number of vehicles removed from
        simulation
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :return:
        """
        data = self._load_data("lane_change_issues",
                               scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        n_simulations = data["simulation_number"].nunique()
        if ("accepted_risk" in data.columns
                and data["accepted_risk"].nunique() > 1):
            issue_count = data.groupby(
                ["vehicles_per_lane", "control percentages",
                 "accepted_risk", "issue"])["veh_id"].count()
        else:
            issue_count = data.groupby(
                ["vehicles_per_lane", "control percentages",
                 "issue"])["veh_id"].count()
        issue_count /= n_simulations
        print(issue_count)
        print("NOTE: the vissim intervention count from the result above is "
              "not reliable")
        data = self._load_data("lane_change_count",
                               scenarios)

        print(data.groupby(["vehicles_per_lane", "control percentages"])[
                  "vissim_in_control"].mean())

    def plot_all_pollutant_heatmaps(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:

        for p_id in self._pollutant_id_to_string:
            self.plot_emission_heatmap(scenarios, p_id)

    def count_lane_changes_from_vehicle_record(
            self, scenario_info: scenario_handling.ScenarioInfo
    ) -> None:
        """
        Counts the lane changes over all simulations under a determined
        simulation configuration
        """
        print(scenario_info.vehicle_percentages)
        warmup_time = 10
        # lc_reader = readers.VissimLaneChangeReader(self.scenario_name)
        # lc_data = lc_reader.load_data_with_controlled_percentage(
        #     [vehicle_percentages], vehicles_per_lane, accepted_risks)
        # # print(lc_data.groupby(["simulation_number"])["veh_id"].count())
        # lc_data.drop(index=lc_data[lc_data["time"] < warmup_time].index,
        #              inplace=True)
        # print("LCs from LC file: ", lc_data.shape[0])

        links = [3, 10002]
        veh_reader = readers.VehicleRecordReader(self.scenario_name)
        lc_counter = []
        data_generator = veh_reader.generate_all_data_from_scenario(
            scenario_info)
        for (data, _) in data_generator:
            data.drop(index=data[~data["link"].isin(links)].index,
                      inplace=True)
            post_processing.drop_warmup_samples(data, warmup_time)
            data.sort_values("veh_id", kind="stable", inplace=True)
            data["is_lane_changing"] = data["lane_change"] != "None"
            data["lc_transition"] = data["is_lane_changing"].diff()
            lc_counter.append(np.count_nonzero(data["lc_transition"]) / 2)
        print("LCs from veh records: ", sum(lc_counter))

    # Platoon LC plots ======================================================= #

    def plot_fundamental_diagram_per_strategy(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: str,
            link_segment_number: int = None,
            aggregation_period: int = 30, warmup_time: float = 10) -> None:
        """
        Uses volume and density from link evaluation.
        """
        link, lane_numbers = self._select_link_and_lanes(
            before_or_after_lc_point, lanes)
        axes = self.plot_fundamental_diagram(
            scenarios, link, link_segment_number=link_segment_number,
            lanes=lane_numbers, hue="lane", col="lane_change_strategy",
            aggregation_period=aggregation_period, warmup_time=warmup_time)
        # title = (lanes + " lane" + ("s" if lanes == "both" else "")
        #          + "; dest lane speed " + orig_and_dest_lane_speeds[1])
        # ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_y_vs_vehicle_input(
            self, y: str,
            scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        """
        Line plot with vehicle input on the x-axis and LC strategies as hue

        :param y: Options: lane_change_completed, maneuver_time,
        travel_time, accel_cost, stayed_in_platoon
        :param scenarios: List of simulation parameters for several scenarios
        """
        self.plot_results_for_platoon_scenario(
            y, "Main Road Input (vehs/h)", "lane_change_strategy", scenarios,
            is_bar_plot=True)

    def plot_y_vs_platoon_lc_strategy(
            self, y: str,
            scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        """
        Line plot with strategies on the x-axis and vehicle input as hue

        :param y: Options: lane_change_completed, maneuver_time,
        travel_time, accel_cost, stayed_in_platoon
        :param scenarios: List of simulation parameters for several scenarios
        """
        self.plot_results_for_platoon_scenario(
            y, "lane_change_strategy", "Main Road Input (vehs/h)", scenarios
        )

    def plot_results_for_platoon_scenario(
            self, y: str, x: str, hue: str,
            scenarios: list[scenario_handling.ScenarioInfo],
            is_bar_plot: bool = False) -> None:
        """

        """
        if "platoon" not in self.scenario_name:
            raise ValueError("Must be scenario with platoon lane changes")

        data = self._load_data(y, scenarios)

        # TODO: Drop samples that didn"t finish simulation
        data.drop(index=data.loc[~data["traversed_network"]].index,
                  inplace=True)
        # TODO: if plotting was_lc_completed, don"t remove
        # TODO: if plotting platoon_maneuver_time, must remove cases where
        #  platoons split

        # Presentation naming
        y_name_map = {
            "lane_change_completed": "% Successful Lane Changes",
            "vehicle_maneuver_time": "Maneuver Time per Vehicle (s)",
            "platoon_maneuver_time": "Platoon Maneuver Time (s)",
            "travel_time": "Travel Time (s)",
            "accel_cost": "Accel Cost (m2/s3)",
            "stayed_in_platoon": "% Stayed in Platoon"
        }
        # y = y.replace("_", " ").title()
        # col_names_map = {name: name.replace("_", " ").title() for name in
        #                  data.columns}
        # data.rename(col_names_map, axis=1, inplace=True)
        data["Main Road Input (vehs/h)"] = data["vehicles_per_lane"]
        sns.set(font_scale=1)
        plot_function = sns.barplot if is_bar_plot else sns.pointplot
        plot_function(data, x=x, y=y,
                      hue=hue, errorbar=("se", 2),
                      palette="tab10"
                      )
        plt.ylabel(y_name_map[y])
        plt.tight_layout()
        plt.show()

    def plot_results_for_platoon_scenario_comparing_speeds(
            self, y: str,
            scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        """

        """
        if "platoon" not in self.scenario_name:
            raise ValueError("Must be scenario with platoon lane changes")

        data = self._load_data(y, scenarios)

        # Presentation naming
        y_name_map = {
            "lane_change_completed": "% Successful Lane Changes",
            "vehicle_maneuver_time": "Maneuver Time per Vehicle (s)",
            "platoon_maneuver_time": "Platoon Maneuver Time (s)",
            "travel_time": "Travel Time (s)",
            "accel_costs": "Accel Cost (m2/s3)",
            "stayed_in_platoon": "% Stayed in Platoon"
        }

        data["Main Road Input (vehs/h)"] = data["vehicles_per_lane"]
        sns.set(font_scale=1)
        sns.pointplot(data, x="lane_change_strategy", y=y,
                      hue="dest_lane_speed", errorbar=("se", 2),
                      palette="tab10"
                      )
        plt.ylabel(y_name_map[y])
        plt.tight_layout()
        plt.show()

    def plot_successful_maneuvers(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        y = "lane_change_completed"
        data = self._load_data(y, scenarios)
        hue_order = data["lane_change_strategy"].unique()
        data["successful maneuver"] = (data["lane_change_completed"]
                                       & data["stayed_in_platoon"])
        data["$\\Delta v$"] = (data["dest_lane_speed"].astype(int)
                               - data["orig_lane_speed"].astype(int))
        grouped_per_sim = data.groupby(
            ["simulation_number", "vehicles_per_lane", "lane_change_strategy",
             "$\\Delta v$", "platoon_size"])
        success_df = grouped_per_sim[
            "successful maneuver"].all().reset_index().rename(
            columns={"lane_change_strategy": "Strategy", }
            # "dest_lane_speed": "dest lane speed"}
        )
        # success_df["successful maneuver"] *= 10
        plt.rc("font", size=17)
        # g = sns.catplot(success_df, kind="bar", y="successful maneuver",
        #                 x=x, hue="Strategy", errorbar=None,
        #                 hue_order=hue_order, col="$\\Delta v$", aspect=1)
        for x in ["platoon_size", "vehicles_per_lane", "$\\Delta v$"]:
            g = sns.catplot(success_df, kind="bar", y="successful maneuver",
                            x=x, hue="Strategy", errorbar=None,
                            hue_order=hue_order, aspect=1)
            # linewidth=4)
            figure = g.figure
            y_label = "% successful maneuvers"
            x_label = " ".join(x.split("_"))
            sns.move_legend(g, loc="upper left", bbox_to_anchor=(0.9, 0.9),
                            frameon=True)
            g.set_axis_labels(x_label, y_label)
            figure.tight_layout()
            if self.should_save_fig:
                fig_name = "platoon_lane_change_success_rate"
                self.save_fig(figure, fig_name=fig_name)
            figure.show()

    def plot_comparison_to_LVF(
            self, scenarios: list[scenario_handling.ScenarioInfo]):
        costs = ["platoon_maneuver_time", "accel_cost", "dist_cost"]
        graph_strategies = ['Graph Min Control', 'Graph Min Time']
        other_strategies = ['LVF']

        # all costs are loaded by the same reader
        data = self._load_data(costs[0], scenarios)
        data.rename(columns={"lane_change_strategy": "Approach"}, inplace=True)
        data["delta_v"] = (pd.to_numeric(data["dest_lane_speed"])
                           - pd.to_numeric(data["orig_lane_speed"]))
        data.drop(
            index=data[~data["Approach"].isin(
                graph_strategies + other_strategies)].index, inplace=True
        )
        # Results by platoon
        simulation_identifiers = ["platoon_size", "delta_v",
                                  "vehicles_per_lane", "simulation_number"]
        # Data is indexed by the simulation identifier. Strategy stays as a
        # column
        grouped_by_platoon = data.groupby(
            simulation_identifiers + ["platoon_id", "Approach"])
        data_by_platoon = grouped_by_platoon.agg(
            {"lane_change_completed": "min", "platoon_maneuver_time": "first",
             "accel_cost": "sum", "dist_cost": "mean"}).reset_index(level=[-1])
        data_by_platoon.fillna(np.inf, inplace=True)
        failure_identifiers = dict()
        for strat in data_by_platoon["Approach"].unique():
            data_per_strategy = data_by_platoon.loc[
                data_by_platoon["Approach"] == strat]
            failure_identifiers[strat] = (
                data_per_strategy.loc[
                    ~data_per_strategy["lane_change_completed"],
                ].index
            )

        graph_failure = failure_identifiers[graph_strategies[0]]
        data_without_graph_failures = data_by_platoon.drop(index=graph_failure)
        name_map = {"platoon_maneuver_time": "Maneuver Time",
                    "accel_cost": "Control Effort",
                    "dist_cost": "Dist. Travelled Before LC"}
        for c in costs:
            # Per other strategy
            for other in other_strategies:
                other_failure = failure_identifiers[other]
                relevant_results = data_without_graph_failures[
                    data_without_graph_failures["Approach"].isin(
                        graph_strategies + [other]
                    )
                ].drop(index=other_failure, errors="ignore")
                ax = sns.pointplot(
                    data=relevant_results.reset_index(),
                    x="platoon_size", y=c, hue="Approach", errorbar=None
                )
                sns.move_legend(ax, "upper left")
                # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                #           title="Approach", ncols=1)
                ax.set_ylabel(name_map[c])
                fig = ax.figure
                fig.tight_layout()
                fig.show()

    def print_comparative_costs(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            costs: Iterable[str] = None
    ) -> None:
        """
        Compares graph approaches (min time and min accel) to other fixed order
        strategies. The accepted costs are 'platoon_maneuver_time' and
        'accel_cost'
        """
        # TODO: split the function
        all_costs = ["platoon_maneuver_time", "accel_cost", "dist_cost"]
        if costs is None:
            costs = all_costs
        else:
            for c in costs:
                if c not in all_costs:
                    raise ValueError(
                        f"{c} not accepted as a cost. The accepted costs "
                        f"are 'platoon_maneuver_time' and 'accel_cost'")

        # all costs are loaded by the same reader
        data = self._load_data(costs[0], scenarios)
        data["delta_v"] = (pd.to_numeric(data["dest_lane_speed"])
                           - pd.to_numeric(data["orig_lane_speed"]))
        # Results by platoon
        simulation_identifiers = ["platoon_size", "delta_v",
                                  "vehicles_per_lane", "simulation_number"]
        # Data is indexed by the simulation identifier. Strategy stays as a
        # column
        grouped_by_platoon = data.groupby(
            simulation_identifiers + ["platoon_id", "lane_change_strategy"])
        data_by_platoon = grouped_by_platoon.agg(
            {"lane_change_completed": "min", "platoon_maneuver_time": "first",
             "accel_cost": "sum", "dist_cost": "mean"}).reset_index(level=[-1])
        n_scenarios = data_by_platoon.index.nunique()
        data_by_platoon.fillna(np.inf, inplace=True)
        failure_identifiers = dict()
        success_counter = dict()
        graph_strategies = []
        other_strategies = []
        for strat in data_by_platoon["lane_change_strategy"].unique():
            data_per_strategy = data_by_platoon.loc[
                data_by_platoon["lane_change_strategy"] == strat]
            failure_identifiers[strat] = (
                data_per_strategy.loc[
                    ~data_per_strategy["lane_change_completed"],
                ].index
            )
            success_counter[strat] = n_scenarios - len(
                failure_identifiers[strat])

            if strat.lower().startswith("graph"):
                graph_strategies.append(strat)
            else:
                other_strategies.append(strat)

        # Sanity check 1
        if (len(failure_identifiers[graph_strategies[0]])
                != len(failure_identifiers[graph_strategies[1]])
            or np.any(failure_identifiers[graph_strategies[0]]
                      != failure_identifiers[graph_strategies[1]])):
            print("Different successful maneuvers between graph approach with "
                  "different costs")
        graph_failure = failure_identifiers[graph_strategies[0]]
        # Sanity check 2
        for other in other_strategies:
            other_failure = failure_identifiers[other]
            for simulation in graph_failure:
                if simulation not in other_failure:
                    print(
                        f"{simulation} failed for graphs but not for {other}. "
                        f"Check!")

        data_without_graph_failures = data_by_platoon.drop(index=graph_failure)
        for c in costs:
            # Per other strategy
            for other in other_strategies:
                other_failure = failure_identifiers[other]
                relevant_results = data_without_graph_failures[
                    data_without_graph_failures["lane_change_strategy"].isin(
                        graph_strategies + [other]
                    )
                ].drop(index=other_failure, errors="ignore")
                # TODO: plot?
                # print(relevant_results.groupby(
                #     "lane_change_strategy")[c].mean())
                # Compute relative change
                other_results = relevant_results[
                    relevant_results["lane_change_strategy"] == other]
                relevant_results[c + " variation"] = (
                        (relevant_results[c] - other_results[c])
                        # / other_results[cost]
                )
                print(relevant_results.groupby("lane_change_strategy")[
                          c + " variation"].mean() / other_results[c].mean())

        # Best heuristic
        # We reset and set index multiple times to make it easier to select and
        # drop only the right simulations
        # Get only non-graph strategies, and select the one with best cost
        # other_results = data_by_platoon[
        #     data_by_platoon["lane_change_strategy"].isin(
        #         other_strategies)].reset_index()
        # best_results = other_results.loc[other_results.groupby(
        #     simulation_identifiers)[cost].idxmin()].set_index(
        #     simulation_identifiers)
        # best_results["lane_change_strategy"] = "Best Fixed-Order"
        # best_failure = best_results.loc[
        #     ~best_results["lane_change_completed"]].index
        # success_counter["Best Fixed-Order"] = n_scenarios - len(best_failure)
        # graph_results = data_without_graph_failures[
        #     data_without_graph_failures['lane_change_strategy'].isin(
        #         graph_strategies)]
        # relevant_results = pd.concat([graph_results, best_results]).drop(
        #     index=best_failure, errors="ignore")
        # print(relevant_results.groupby("lane_change_strategy")[cost].mean())
        #
        print(success_counter)

    def plot_flow_box_plot_vs_strategy(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: str,
            hue: str = "vehicles_per_lane",
            warmup_time: float = 5,
            aggregation_period: int = 30) -> None:
        """

        """
        y = "flow"
        aggregated_data = self._load_flow_data_for_platoon_lc_scenario(
            scenarios, before_or_after_lc_point, lanes, warmup_time,
            aggregation_period)

        fig, ax = _my_boxplot(aggregated_data, "lane_change_strategy",
                              y, hue=hue, will_show=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                  title="Vehicles per lane", ncols=1)
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name("box_plot", y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_link_data_box_plot_vs_strategy(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: str,
            segment_number: int = None, warmup_time: float = 5,
            sim_time: float = None,
            aggregation_period: int = 30) -> None:
        """

        """
        link, lane_numbers = self._select_link_and_lanes(
            before_or_after_lc_point, lanes)
        data = self._load_data(y, scenarios)
        data["# CAVs (veh/h)"] = data["vehicles_per_lane"] * 2
        relevant_data = self._prepare_link_evaluation_data(
            data, link, segment_number, lane_numbers,
            warmup_time=warmup_time, sim_time=sim_time,
            aggregation_period=aggregation_period)
        fig, ax = _my_boxplot(relevant_data, "lane_change_strategy",
                              y, hue="# CAVs (veh/h)",
                              will_show=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                  ncols=1, title="# CAVs (veh/h)")
        y_label = _make_pretty_label(y)
        ax.set_ylabel(y_label)
        segment_str = " " if segment_number is None else str(segment_number)
        ax.set_title(" ".join([before_or_after_lc_point, segment_str]))
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name("box_plot", y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_flows_vs_time_per_strategy(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: str,
            warmup_time: float = 5, aggregation_period: int = 30,
            use_all_simulations: bool = False) -> None:

        y = "vehicle_count"
        aggregated_data = self._load_flow_data_for_platoon_lc_scenario(
            scenarios, before_or_after_lc_point, lanes, warmup_time,
            aggregation_period)

        in_common_data = self._make_data_uniform(aggregated_data,
                                                 use_all_simulations)

        plt.rc("font", size=17)
        sns.relplot(in_common_data, x="time", y=y,
                    kind="line", hue="sensor position",
                    row="lane_change_strategy",
                    aspect=4)
        plt.tight_layout()
        plt.show()

    def plot_link_data_vs_time_per_strategy(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: str,
            link_segment_number: int = None,
            aggregation_period: int = 30, warmup_time: float = 10,
            sim_time: float = None, use_all_simulations: bool = False) -> None:

        link, lane_numbers = self._select_link_and_lanes(
            before_or_after_lc_point, lanes)
        data = self._load_data(y, scenarios)
        aggregated_data = self._prepare_link_evaluation_data(
            data, link, link_segment_number, lane_numbers,
            aggregate_lanes=False, warmup_time=warmup_time, sim_time=sim_time,
            aggregation_period=aggregation_period)

        in_common_data = self._make_data_uniform(aggregated_data,
                                                 use_all_simulations)
        lane_map = {1: "Origin", 2: "Destination"}
        in_common_data["lane"] = in_common_data["lane"].map(lane_map)
        in_common_data.rename(columns={"lane_change_strategy": "Strategy",
                                       "lane": "Lane"},
                              inplace=True)
        # Main plot
        hue_order = in_common_data["Strategy"].unique()
        x_label = "time (s)"
        y_label = _make_pretty_label(y)
        plt.rc("font", size=17)
        g = sns.relplot(in_common_data, x="time", y=y,
                        kind="line", hue="Strategy", hue_order=hue_order,
                        row="Lane", aspect=2)
        g.set_axis_labels(x_label, y_label)

        # Include relevant maneuver points
        lc_scenarios = [sc for sc in scenarios
                        if sc.special_case != "no_lane_change"]
        if len(lc_scenarios) > 0:
            maneuver_times = self._get_platoon_maneuver_times(lc_scenarios)
            maneuver_times["time"] = (
                    (maneuver_times["time_exact"] - warmup_time * 60
                        + aggregation_period / 2 - 0.1)
                    // aggregation_period * aggregation_period
            )
            maneuver_times["time"] = maneuver_times["time"].astype(int)
            merged_data = pd.merge(
                left=maneuver_times, left_on=["Strategy", "time"],
                right=in_common_data[["Strategy", "time", "Lane", y]],
                right_on=["Strategy", "time"])
            self._add_maneuver_phases_to_plot(
                merged_data.rename(columns={y: "y"}), g, hue_order)

        fig = g.figure
        fig.tight_layout()
        fig.show()

    def plot_relevant_vehicles_states(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        """
        Plots the acceleration and velocity of platoon vehicles and
        destination lane follower. All strategies are shown in the same plot for
        easier comparison. Designed for scenarios with a single lane change
        """
        reader = readers.VehicleRecordReader(self.scenario_name)
        generator = reader.generate_data_from_several_scenarios(scenarios)
        platoon_vehicles_per_strategy = []
        dest_lane_follower_per_strategy = []
        for vehicle_records, _ in generator:
            platoon_vehicles = vehicle_records[
                vehicle_records["veh_type"] == Vehicle.VISSIM_PLATOON_CAR_ID]
            platoon_vehicle_ids = platoon_vehicles["veh_id"].unique()
            dest_lane_follower_id = vehicle_records.loc[
                (vehicle_records["leader_id"].isin(platoon_vehicle_ids))
                & (~vehicle_records["veh_id"].isin(platoon_vehicle_ids))
                & (vehicle_records["lane"]
                   == platoon_vehicles.iloc[-1]["lane"]),
                "veh_id"
            ].unique()
            dest_lane_follower = vehicle_records[
                vehicle_records["veh_id"].isin(dest_lane_follower_id)]
            platoon_vehicles_per_strategy.append(platoon_vehicles)
            dest_lane_follower_per_strategy.append(dest_lane_follower)
        all_platoon_vehicles = pd.concat(platoon_vehicles_per_strategy)
        all_dest_lane_followers = pd.concat(dest_lane_follower_per_strategy)

        sns.relplot(all_platoon_vehicles, y="ax", x="time", kind="line",
                    hue="lane_change_strategy", style="lane_change_strategy",
                    col="veh_id", col_wrap=2)
        plt.show()
        sns.lineplot(all_dest_lane_followers, y="ax", x="time",
                     hue="lane_change_strategy", style="lane_change_strategy")
        plt.show()

    def plot_sample_trajectories(
            self, scenario: scenario_handling.ScenarioInfo) -> None:
        """
        Plots the trajectories (x vs t) of the platoon vehicles and some
        surrounding vehicles
        """
        reader = readers.VehicleRecordReader(self.scenario_name)
        vehicle_records = reader.load_sample_data_from_scenario(scenario)
        main_link = self.file_handler.get_main_links()[1]
        vehicle_records = vehicle_records[vehicle_records["link"] == main_link]

        platoon_vehicles = vehicle_records[
            vehicle_records["veh_type"] == Vehicle.VISSIM_PLATOON_CAR_ID]
        first_time = platoon_vehicles["time"].min()
        last_time = platoon_vehicles["time"].max()
        vehicle_records = vehicle_records[
            (vehicle_records["time"] >= first_time)
            & (vehicle_records["time"] <= last_time)]

        platoon_vehicle_ids = platoon_vehicles["veh_id"].unique()
        follower_ids = vehicle_records.loc[
            (vehicle_records["leader_id"].isin(platoon_vehicle_ids))
            & (~vehicle_records["veh_id"].isin(platoon_vehicle_ids)),
            "veh_id"].unique()
        leader_ids = vehicle_records.loc[
            (vehicle_records["veh_id"].isin(
                platoon_vehicles["leader_id"].unique()))
            & (~vehicle_records["veh_id"].isin(platoon_vehicle_ids)),
            "veh_id"].unique()
        leader_ids = leader_ids[leader_ids > 2]

        followers = vehicle_records[vehicle_records["veh_id"].isin(
            follower_ids)]
        leaders = vehicle_records[vehicle_records["veh_id"].isin(
            leader_ids)]
        vehicles_to_plot = pd.concat([platoon_vehicles, followers, leaders])
        color_map = {veh_id: "red" if veh_id in platoon_vehicle_ids else "grey"
                     for veh_id in vehicle_records["veh_id"].unique()}

        sns.relplot(vehicles_to_plot, y="x", x="time", kind="line",
                    hue="veh_id", col="lane", palette=color_map,
                    aspect=1)
        plt.show()

    def plot_platoon_states(self,
                            scenario: scenario_handling.ScenarioInfo) -> None:
        reader = readers.VehicleRecordReader(self.scenario_name)
        veh_record = reader.load_sample_data_from_scenario(scenario)
        platoon_vehicles = veh_record[veh_record["veh_type"]
                                      == Vehicle.VISSIM_PLATOON_CAR_ID].copy()
        # The highest veh id is the platoon leader
        platoon_vehicles["platoon position"] = (np.abs(
            platoon_vehicles["veh_id"] - platoon_vehicles["veh_id"].max()) + 1)
        lc_time = platoon_vehicles.loc[platoon_vehicles["state"]
                                       != "lane keeping", "time"]
        lc_start_time, lc_end_time = lc_time.iloc[0], lc_time.iloc[-1]
        plot_start_time = lc_start_time - 10
        plot_end_time = lc_end_time + 10
        platoon_vehicles = platoon_vehicles[
            (platoon_vehicles["time"] >= plot_start_time)
            & (platoon_vehicles["time"] <= plot_end_time)]
        platoon_vehicles["time"] -= plot_start_time

        fig = plt.figure()
        plt.rc("font", size=15)
        ax = sns.lineplot(platoon_vehicles, x="time", y="state",
                          hue="platoon position", style="platoon position",
                          palette="tab10", linewidth=3)
        # ax.legend().set_title("")
        handles, labels = ax.get_legend_handles_labels()
        labels = ["$p_" + str(i) + "$" for i in labels]
        ax.legend(handles=handles, labels=labels)
        ax.invert_yaxis()
        ax.set_ylabel("")
        ax.set_xlabel("time (s)")
        fig.tight_layout()
        if self.should_save_fig:
            fig_name = "_".join([
                "platoon_states",
                scenario.platoon_lane_change_strategy.get_print_name()])
            self.save_fig(fig, fig_name=fig_name)
        fig.show()

    def compare_travel_times(
            self, scenarios: list[scenario_handling.ScenarioInfo], x: str,
            plot_cols: str = None,
            warmup_time: float = 10, sim_time: float = None) -> None:
        """
        Plots a bar plot to compare how the strategies perform in each scenario.
        Strategy is passed as hue.

        :param scenarios: List of simulation parameters for several scenarios
        :param x: The variable on the x-axis of the plot
        :param plot_cols: Makes one plot for each value of the variable in
         plot_cols (passed relplot/catplot)
        :param warmup_time: Time *in minutes* below which samples are discarded
        :param sim_time: Time *in minutes* after which samples are discarded
         (after subtracting the warmup time)
        """
        y = "travel_time"
        data = self._load_data(y, scenarios)
        data = self._prepare_travel_time_data(data, warmup_time, sim_time)

        # We subtract the mean travel time of all vehicle ahead of the platoons
        # from all other travel times to get the delay caused by the platoon
        # maneuver
        independent_variables = ["delta_v", "vehicles_per_lane",
                                 "platoon_size"]
        by_list = ["platoon_ahead", "Strategy", "Lane"] + independent_variables
        # if x != "Lane":
        #     by_list.append(x)
        grouped = data.groupby(by_list)
        normalizer = grouped[y].mean().loc[-1]
        delays = dict()
        for key, group in grouped:
            group[y] = (group[y] - normalizer.loc[key[1:]]) / normalizer.loc[key[1:]]
            delays[key] = group[y].sum()

        new_y = "travel_time_delay"
        delay_df = pd.DataFrame(data=delays.values(), index=delays.keys(),
                                columns=[new_y])
        delay_df = delay_df.drop(index=-1).droplevel(0).reset_index(
            names=by_list[1:]
        )

        # # Direct output
        # print(data.groupby(["Strategy", x, "platoon_ahead"]).agg(
        #     {"travel_time": ["mean", "std"]}))
        # Plot
        if plot_cols == "dest_lane_speed" and x == "delta_v":
            plot_cols = x

        plt.rc("font", size=17)
        hue = "Strategy"
        hue_order = data["Strategy"].unique()
        if plot_cols and plot_cols != x and plot_cols != hue:
            plt.rc("font", size=25)
            g = sns.catplot(delay_df, x=x, y=new_y, hue=hue,
                            hue_order=hue_order, col=plot_cols,
                            col_wrap=3, kind="bar",
                            errorbar=None)
            sns.move_legend(g, loc="upper left", bbox_to_anchor=(0.07, 0.88),
                            frameon=True)
        else:
            g = sns.catplot(delay_df, x=x, y=new_y, hue=hue,
                            hue_order=hue_order, kind="bar",
                            errorbar=None)
            sns.move_legend(g, loc="lower left", bbox_to_anchor=(0.2, 0.5),
                            frameon=True)
        figure = g.figure
        x_label = _make_pretty_label(x)
        y_label = _make_pretty_label(new_y)
        g.set_axis_labels(x_label, y_label)
        if self.should_save_fig:
            main_variable = plot_cols if plot_cols else x
            fig_name = "_".join([new_y, self.scenario_name, "varying",
                                 main_variable])
            self.save_fig(figure, fig_name=fig_name)
        figure.tight_layout()
        figure.show()

    def plot_travel_times_vs_entrance_times(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10, sim_time: float = None) -> None:
        """

        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: Time *in minutes* below which samples are discarded
        :param sim_time: Time *in minutes* after which samples are discarded
         (after subtracting the warmup time)
        """
        y = "travel_time"
        data = self._load_data(y, scenarios)
        data = self._prepare_travel_time_data(data, warmup_time, sim_time)

        # Drop platoon vehicles
        platoon_data = data[data["veh_type"] == Vehicle.VISSIM_PLATOON_CAR_ID]
        data.drop(index=platoon_data.index, inplace=True)

        # Main plot
        plt.rc("font", size=17)
        hue_order = data["Strategy"].unique()
        row_order = ["Origin", "Destination"]  # data["Lane"].unique()
        # col = "dest_lane_speed" if data["dest_lane_speed"].nunique() > 1
        data["entrance_time"] = (data["entrance_time"]
                                 - platoon_data["entrance_time"].min())
        g = sns.relplot(data, kind="line", x="entrance_time", y="travel_time",
                        hue="Strategy", row="Lane", col="dest_lane_speed",
                        hue_order=hue_order, row_order=row_order,
                        )
        x_label = "relative entrance time (s)"
        y_label = _make_pretty_label(y)
        g.set_axis_labels(x_label, y_label)

        plt.tight_layout()
        plt.show()

    def illustrate_travel_time_delay(
            self, scenario: scenario_handling.ScenarioInfo,
            lane: str = "Destination",
            warmup_time: float = 10, sim_time: float = None) -> None:
        """

        :param scenario: List of simulation parameters for several scenarios
        :param lane: The lane to be shown: Origin or Destination
        :param warmup_time: Time *in minutes* below which samples are discarded
        :param sim_time: Time *in minutes* after which samples are discarded
         (after subtracting the warmup time)
        """
        y = "travel_time"
        data = self._load_data(y, [scenario])
        data = self._prepare_travel_time_data(data, warmup_time, sim_time)

        # Drop platoon vehicles
        platoon_data = data[data["veh_type"] == Vehicle.VISSIM_PLATOON_CAR_ID]
        data["entrance_time"] = (data["entrance_time"]
                                 - platoon_data["entrance_time"].min())
        data.drop(index=platoon_data.index, inplace=True)
        data = data[data["Lane"] == lane]

        const = data.loc[data["entrance_time"] < 0, "travel_time"].mean()

        # Main plot
        plt.rc("font", size=17)
        x = "entrance_time"
        ax = sns.lineplot(data, x=x, y=y)
        ax.fill_between(data[x], data[y], const, alpha=0.4)
        x_label = "vehicle entrance time (s)"
        y_label = "vehicle travel time (s)"  # _make_pretty_label(y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Travel time delay computation example")
        # xy = (data[x].mean(), data[y].max()/2)
        ax.annotate("cooperation\nstarts", xy=(0, const),
                    xytext=(-30, const + 5), ha="right",
                    arrowprops=dict(facecolor="black"))
        ax.text(x=50, y=210, s="travel time\ndelay", ha="center",
                bbox=dict(boxstyle="square", fc="white"))
        if self.should_save_fig:
            figure = plt.gcf()
            fig_name = "travel_time_delay_illustration"
            self.save_fig(figure, fig_name=fig_name)
        plt.tight_layout()
        plt.show()

    def get_vehicles_impacted_by_lane_change(
            self,
            scenarios: list[scenario_handling.ScenarioInfo]) -> pd.DataFrame:

        y = "travel_time"
        data = self._load_data(y, scenarios)
        data = self._prepare_travel_time_data(data)

        # We subtract the mean travel time of all vehicle ahead of the platoons
        # from all other travel times to get the delay caused by the platoon
        # maneuver
        independent_variables = ["dest_lane_speed", "vehicles_per_lane",
                                 "platoon_size"]
        by_list = ["platoon_ahead", "Strategy", "Lane"] + independent_variables
        grouped = data.groupby(by_list)
        normalizer = grouped[y].mean().loc[-1]
        impacted_vehicles = []
        for key, group in grouped:
            free_flow_travel_time = normalizer.loc[key[1:]]
            group[y] -= free_flow_travel_time
            impacted_vehicles.append(group[np.abs(group[y])
                                           > 0.01 * free_flow_travel_time])

        return pd.concat(impacted_vehicles).rename(
            columns={"travel_time": "travel time delay"})

    # TODO [march 24, 23]: merge with plot_emission_heatmap?
    def compare_emissions_for_platoon_scenarios(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            pollutant_id: int = 91
    ) -> None:
        y = "emission"
        data = self._load_data(y, scenarios)
        data = ResultAnalyzer._prepare_pollutant_data(data, pollutant_id)
        data.rename(columns={"lane_change_strategy": "Strategy"}, inplace=True)
        data["delta_v"] = (data["dest_lane_speed"].astype(int)
                           - data["orig_lane_speed"].astype(int))
        # title = self._pollutant_id_to_string[pollutant_id]
        x = "delta_v"

        normalizer = data[y].min()
        fig, ax = _plot_heatmap(data, y, "Strategy", x,
                                normalizer, show_figure=False)
        fig.tight_layout()
        fig.show()

        fig = plt.figure()
        plt.rc("font", size=18)
        ax = sns.pointplot(data, x=x, y=y,
                           hue="Strategy")
        # ax.set_title(title)
        ax.set_xlabel(_make_pretty_label(x))
        ax.set_ylabel(_make_pretty_label(
            self._pollutant_id_to_string[pollutant_id]))
        fig.tight_layout()
        fig.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                "barplot", "pollutant_" + str(pollutant_id),
                scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def normalize_lane_change_times(self, output_data: pd.DataFrame,
                                    lane_change_data: pd.DataFrame
                                    ) -> pd.DataFrame:
        """
        :param output_data: Link evaluation or data collection measurements
        :param lane_change_data: Detailed lane change data for the scenarios
        """
        max_duration = lane_change_data["platoon_maneuver_time"].max()
        grouped_lc_data = lane_change_data.groupby(
            ["lane_change_strategy", "simulation_number",
             "initial_platoon_id"])
        grouped_data = output_data.groupby(
            ["lane_change_strategy", "simulation_number"])

        t1, t2 = output_data["time_interval"].iloc[0].split("-")
        interval = int(t2) - int(t1)
        normalized_data_list = []
        margin = 2
        for group_id, single_lc_data in grouped_lc_data:
            lc_t0 = single_lc_data["maneuver_start_time"].iloc[0]
            lc_tf = lc_t0 + max_duration
            t0 = (lc_t0 // interval - margin) * interval
            tf = (lc_tf // interval + margin) * interval
            single_simulation_data = grouped_data.get_group(group_id[0:2])
            data_during_lc = single_simulation_data[
                (single_simulation_data["time"] >= t0)
                & (single_simulation_data["time"] < tf)].copy()
            data_during_lc["time"] = data_during_lc["time"] - t0
            normalized_data_list.append(data_during_lc)
        return pd.concat(normalized_data_list)

    @staticmethod
    def _add_maneuver_phases_to_plot(
            data: pd.DataFrame, facet_grid: sns.FacetGrid,
            hue_order: list[str]) -> None:
        """
        Given a facet grid of some variable y vs time, this function adds
        markers for relevant lane change phases.

        :param data: Dataframe containing the phase times as well as the values
         of variable y at these times (most likely result of a merge operation).
         The column to be plotted must be called "y"
        :param facet_grid: The facet grid generated by seaborn
        :param hue_order: Same as the seaborn hue_order parameter. We use it to
         ensure the lines and markers have matching colors.
        """

        grouped_per_lane = data.groupby("Lane")
        markers = ["$\\mathbf{" + str(i) + "}$" for i in "ABCD"]
        for name, group in grouped_per_lane:
            ax = facet_grid.axes_dict[name]
            sns.scatterplot(group, x="time", y="y", hue="Strategy",
                            hue_order=hue_order, style="Phase",
                            s=300, ax=ax, markers=markers)

        # Manage all legends
        n_strategies = len(hue_order)
        n_phases = data["Phase"].nunique()

        handles, labels = facet_grid.axes[0][0].get_legend_handles_labels()
        new_handles = (handles[n_strategies:n_strategies + 1]  # "Strategy"
                       + handles[:n_strategies]  # Strategy names
                       + handles[-n_phases - 1:])  # Phases
        new_labels = (labels[n_strategies:n_strategies + 1]  # "Strategy"
                      + labels[:n_strategies]  # Strategy names
                      + labels[-n_phases - 1:])  # Phases
        old_legend = facet_grid.legend
        old_legend.remove()
        facet_grid.axes[1][0].legend().remove()
        facet_grid.axes[0][0].legend(new_handles, new_labels, loc="upper left",
                                     bbox_to_anchor=(1, 1), markerscale=1.5)

    # Traffic Light Scenario Plots =========================================== #

    def plot_violations_per_control_percentage(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        """
        Plots number of violations .

        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: must be given in minutes. Samples before
         warmup_time are ignored.
        :return: Nothing, just plots figures
        """
        # if not isinstance(vehicles_per_lane, list):
        #     vehicles_per_lane = [vehicles_per_lane]

        n_simulations = 10
        # violations_reader = readers.ViolationsReader(self.scenario_name)
        data = self._load_data("violations", scenarios)

        # TODO: temporary
        data.drop(index=data[data["simulation_number"] == 11].index,
                  inplace=True)

        post_processing.drop_warmup_samples(data, warmup_time)
        results = data.groupby(["control percentages", "vehicles_per_lane"],
                               as_index=False)["veh_id"].count().rename(
            columns={"veh_id": "violations count"})
        results["mean violations"] = results["violations count"] / n_simulations
        print(results)

    def plot_heatmap_for_traffic_light_scenario(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        """
        Plots a heatmap
        :param y:
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :return:
        """

        title_map = {"vehicle_count": "Output Flow",
                     "discomfort": "Discomfort",
                     "barrier_function_risk": "Risk"}
        n_simulations = 10
        plt.rc("font", size=17)
        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        subsets_by_vpl = scenario_handling.split_scenario_by(
            scenarios, "vehicles_per_lane")
        for vpl, data_to_plot in data.groupby("vehicles_per_lane"):
            table = pd.pivot_table(data_to_plot, values=y,
                                   index=["traffic_light_cacc_percentage"],
                                   columns=["traffic_light_acc_percentage"],
                                   aggfunc=np.sum)
            if y == "vehicle_count":
                table_norm = (table / n_simulations
                              * 3 / 2)  # 60min / 20min / 2 lanes
                value_format = ".0f"
            else:
                table_norm = table / table[0].iloc[0]
                value_format = ".2g"
            max_table_value = np.round(np.nanmax(table_norm.to_numpy()))
            sns.heatmap(table_norm.sort_index(axis=0, ascending=False),
                        annot=True, fmt=value_format,
                        vmin=0, vmax=max(1, int(max_table_value)))
            plt.xlabel("% of CAVs without V2V", fontsize=22)
            plt.ylabel("% of CAVs with V2V", fontsize=22)
            plt.title(title_map[y] + " at " + str(vpl) + " vehs/h/lane",
                      fontsize=22)
            if self.should_save_fig:
                fig = plt.gcf()
                self.save_fig(fig, "heatmap", y, subsets_by_vpl[vpl])
            plt.tight_layout()
            plt.show()

    def plot_violations_heatmap(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            warmup_time: float = 10) -> None:
        n_simulations = 10
        plt.rc("font", size=17)
        data = self._load_data("violations", scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        subsets_by_vpl = scenario_handling.split_scenario_by(
            scenarios, "vehicles_per_lane")
        for vpl, data_to_plot in data.groupby(["vehicles_per_lane"]):
            table = pd.pivot_table(
                data_to_plot[["veh_id", "traffic_light_acc_percentage",
                              "traffic_light_cacc_percentage"]],
                values="veh_id",
                index=["traffic_light_cacc_percentage"],
                columns=["traffic_light_acc_percentage"],
                aggfunc=np.count_nonzero)
            # Create results for 100% controlled vehicles
            if 100 not in table.columns and 100 not in table.index:
                table[100] = np.nan
                table.loc[100] = np.nan
                for i in table.index:
                    table.loc[i, 100 - i] = 0
            table_norm = table / n_simulations
            # max_table_value = np.round(np.nanmax(table_norm.to_numpy()))
            sns.heatmap(table_norm.sort_index(axis=0, ascending=False),
                        annot=True,
                        vmin=0)
            plt.xlabel("% of CAVs without V2V", fontsize=22)
            plt.ylabel("% of CAVs with V2V", fontsize=22)
            plt.title("Violations at " + str(vpl) + " vehs/h/lane",
                      fontsize=22)
            if self.should_save_fig:
                fig = plt.gcf()
                self.save_fig(fig, "heatmap", "violations", subsets_by_vpl[vpl])
            plt.show()

    # Others ================================================================= #

    def accel_vs_time_for_different_vehicle_pairs(self) -> None:
        """
        Plots acceleration vs time for human following CAV and CAV following
        human. This requires very specific knowledge of the simulation being
        loaded. Currently this function plots a result for the traffic_lights
        scenario with 25% CACC-equipped vehicles.
        :return:
        """
        # scenario_name = "traffic_lights"
        # vehicle_percentage = {VehicleType.TRAFFIC_LIGHT_CACC: 25}
        # vehicles_per_lane = 500
        # all_cases = [
        #     {"follower": "human", "leader": "human", "id": 202},
        #     {"follower": "human", "leader": "CAV", "id": 203},
        #     {"follower": "CAV", "leader": "human", "id": 201},
        #     # {"follower": "CAV", "leader": "CAV", "id": 152},
        #     {"follower": "CAV", "leader": "CAV", "id": 196},
        #     # {"follower": "CAV", "leader": "CAV", "id": 208},
        #     # {"follower": "CAV", "leader": "CAV", "id": 234},
        #     # {"follower": "CAV", "leader": "CAV", "id": 239}
        # ]
        # time = [910, 950]
        scenario_name = "in_and_out_safe"
        vehicle_percentage = {VehicleType.CONNECTED: 50}
        vehicles_per_lane = 2000
        all_cases = [
            {"follower": "human", "leader": "human",
             "id": 35, "time": [30, 40]},
            {"follower": "human", "leader": "CAV",
             "id": 33, "time": [30, 40]},
            {"follower": "CAV", "leader": "human",
             "id": 53, "time": [37, 53]},
            {"follower": "CAV", "leader": "CAV",
             "id": 17, "time": [22, 31]},
        ]
        reader = readers.VehicleRecordReader(scenario_name)
        scenario_info = scenario_handling.ScenarioInfo(vehicle_percentage,
                                                       vehicles_per_lane,
                                                       accepted_risk=0)
        veh_record = reader.load_single_file_from_scenario(1, scenario_info)
        # fig, axes = plt.subplots(len(all_cases), 1)
        # fig.set_size_inches(12, 16)
        plt.rc("font", size=25)
        # full_data = []
        for case in all_cases:
            # case = all_cases[i]
            veh_id = case["id"]
            time = case["time"]
            veh_record_slice = veh_record[(veh_record["time"] > time[0])
                                          & (veh_record["time"] < time[1])]
            follower_data = veh_record_slice.loc[veh_record_slice["veh_id"]
                                                 == veh_id]
            leader_id = follower_data.iloc[0]["leader_id"]
            leader_data = veh_record_slice.loc[veh_record_slice["veh_id"]
                                               == leader_id]
            data = pd.concat([follower_data, leader_data])
            data["name"] = ""
            data.loc[data["veh_id"] == veh_id, "name"] = (case["follower"]
                                                          + " follower")
            data.loc[data["veh_id"] == leader_id, "name"] = (case["leader"]
                                                             + " leader")
            # data["gap"] = 0
            # follower_gap = post_processing.compute_gap_between_vehicles(
            #     leader_data["rear_x"].to_numpy(),
            #     leader_data["rear_y"].to_numpy(),
            #     follower_data["front_x"].to_numpy(),
            #     follower_data["front_y"].to_numpy())
            # data.loc[data["veh_id"] == veh_id, "gap"] = follower_gap
            # full_data.append(data)
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(9, 4)
            sns.lineplot(data=data,
                         x="time", y="ax", hue="name",
                         palette="tab10", linewidth=4, ax=axes,
                         # legend=False
                         )
            axes.legend(loc="lower left",
                        bbox_to_anchor=(-0.1, 0.9), ncol=2,
                        frameon=False)
            axes.set_xlabel("t ($s$)")
            axes.set_ylabel("a ($m/s^2$)")
            fig.tight_layout()
            if self.should_save_fig:
                fig = plt.gcf()
                fig.set_dpi(300)
                fig_name = "_".join(["accel_vs_time", scenario_name,
                                     case["follower"], "follower",
                                     case["leader"], "leader"])
                fig.savefig(os.path.join(self._figure_folder, fig_name))
            fig.show()

    def risk_vs_time_example(self) -> None:
        scenario_name = "in_and_out_safe"
        vehicle_percentage = {VehicleType.CONNECTED: 0}
        vehicles_per_lane = 1000
        reader = readers.VehicleRecordReader(scenario_name)
        scenario_info = scenario_handling.ScenarioInfo(vehicle_percentage,
                                                       vehicles_per_lane,
                                                       accepted_risk=0)
        veh_record = reader.load_single_file_from_scenario(1, scenario_info)
        veh_id = 675
        single_veh = veh_record[veh_record["veh_id"] == veh_id]
        min_t = single_veh["time"].iloc[0]
        max_t = single_veh["time"].iloc[-1]
        relevant_data = veh_record[(veh_record["time"] >= min_t)
                                   & (veh_record["time"] <= max_t)].copy()
        pp = post_processing.SSMProcessor(scenario_name)
        pp.post_process(relevant_data)
        single_veh = relevant_data[relevant_data["veh_id"] == veh_id]

        plt.rc("font", size=40)
        sns.lineplot(data=single_veh, x="time", y="risk", linewidth=4)
        plt.xlabel("time (s)")
        plt.ylabel("risk (m/s)")
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        plt.tight_layout()
        if self.should_save_fig:
            fig_name = "risk_vs_time_example"
            self.save_fig(fig, fig_name=fig_name)
        plt.show()
        sampling = single_veh["time"].diff().iloc[1]
        print("total risk:", single_veh["risk"].sum() * sampling)

    def speed_color_map(self, scenario: scenario_handling.ScenarioInfo,
                        link: int, lane: int = None, warmup_time: float = 0,
                        sim_time: float = None) -> None:
        """
        :param scenario: Simulation scenario parameters
        :param link: road link for which we want the color map
        :return:
        """

        reader = readers.VehicleRecordReader(self.scenario_name)
        if self.is_debugging:
            data = reader.load_test_data(scenario)
        else:
            data = reader.load_data_from_scenario(scenario)

        post_processing.drop_warmup_samples(data, warmup_time,
                                            normalize_time=True)
        post_processing.drop_late_samples(data, sim_time)

        data = data[data["link"] == link]
        if lane is not None:
            data = data[data["lane"] == lane]
        data["time (s)"] = ((data["time"] // 10) * 10).astype(int)
        space_bins = [i for i in
                      range(0, int(data["x"].max()), 25)]
        data["x (m)"] = pd.cut(data["x"], bins=space_bins,
                               labels=space_bins[:-1])
        plotted_data = data.groupby(["time (s)", "x (m)"],
                                    as_index=False)["vx"].mean()
        plotted_data = plotted_data.pivot("time (s)", "x (m)", "vx")
        fig = plt.figure()
        ax = sns.heatmap(plotted_data)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.show()
        return fig

    # Multiple plots  ======================================================== #

    def get_flow_and_risk_plots(
            self, scenarios: list[scenario_handling.ScenarioInfo]) -> None:
        """Generates the plots used in the Safe Lane Changes paper."""

        self.plot_flow_box_plot_vs_controlled_percentage(scenarios,
                                                         warmup_time=10)
        self.plot_risk_histograms("total_risk", scenarios, min_risk=1)
        self.plot_risk_histograms("total_lane_change_risk", scenarios,
                                  min_risk=1)

    # Support methods ======================================================== #
    def _load_data(
            self, y: str, scenarios: list[scenario_handling.ScenarioInfo]
    ) -> pd.DataFrame:
        """
        :param y: Desired variable; it determines which reader will be used
        :param scenarios: List of simulation parameters for several scenarios
        """
        reader = self._data_reader_map[y](self.scenario_name)
        if self.is_debugging:
            if len(scenarios) > 1:
                print("[Warning] Debugging several scenarios at once "
                      "is not possible")
            data = reader.load_test_data(scenarios[0])
        else:
            data = reader.load_data_from_several_scenarios(scenarios)
        data["vehicles per hour"] = (
                self.file_handler.get_n_lanes()
                * data["vehicles_per_lane"])
        return data

    def _load_flow_data_for_platoon_lc_scenario(
            self, scenarios: list[scenario_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: str,
            warmup_time: float = 5, aggregation_period: int = 30
    ) -> pd.DataFrame:

        y = "vehicle_count"
        data = self._load_data(y, scenarios)
        data["# CAVs (veh/h)"] = data["vehicles_per_lane"] * 2
        in_flow_sensors, out_flow_sensors = (
            self._select_sensors_for_platoon_lc_scenario(
                before_or_after_lc_point, lanes)
        )

        data_in = self._prepare_data_collection_data(
            data, in_flow_sensors, warmup_time=warmup_time,
            aggregate_sensors=True, aggregation_period=aggregation_period)
        data_in["sensor position"] = "in"
        data_out = self._prepare_data_collection_data(
            data, out_flow_sensors, warmup_time=warmup_time,
            aggregate_sensors=True, aggregation_period=aggregation_period)
        data_out["sensor position"] = "out"

        return pd.concat([data_in, data_out])

    # def _get_maneuver_times(self,
    # scenarios: list[scenario_handling.ScenarioInfo]):
    #     lc_data = self._load_data("platoon_maneuver_time", scenarios)
    #     lc_data.rename(columns={"first_platoon_time": "first_lane keeping"},
    #                    inplace=True)
    #     lc_data["last_platoon_time"] = (lc_data["first_lane keeping"] +
    #                                     lc_data["travel_time"])
    #     first_time_cols = [col for col in lc_data if col.startswith("first")]
    #     last_time_cols = [col for col in lc_data if col.startswith("last")]
    #     # Create mapping from state name to number
    #     mask = ~lc_data.isnull().any(axis=1)
    #     full_row = lc_data.loc[mask, first_time_cols].iloc[0]
    #     full_row.rename(index={col: col[len("first_"):]
    #                            for col in full_row.index}, inplace=True)
    #     full_row.transpose()
    #     full_row = full_row.sort_values().reset_index()
    #     state_to_value_map = dict(zip(full_row["index"], full_row.index))
    #     grouped = lc_data.groupby("lane_change_strategy")

    def _get_platoon_maneuver_times(self, scenarios: list[
        scenario_handling.ScenarioInfo]
                                    ) -> pd.DataFrame:
        lc_data = self._load_data("platoon_maneuver_time", scenarios)
        lc_data.rename(columns={"first_platoon_time": "entrance_time"},
                       inplace=True)
        first_time_cols = [col for col in lc_data if col.startswith("first")]
        last_time_cols = [col for col in lc_data if col.startswith("last")]

        grouped = lc_data.groupby("lane_change_strategy")
        start_times = grouped[first_time_cols].min().min(axis=1)
        final_times = grouped[last_time_cols].max().max(axis=1)
        first_lc_times = grouped["first_lane changing"].min()
        last_lc_times = grouped["last_lane changing"].max()
        maneuver_times = pd.concat(
            [start_times, final_times, first_lc_times, last_lc_times], axis=1)
        maneuver_times.columns = ["coop. request sent", "platoon back together",
                                  "first lane change starts",
                                  "last lane change ends"]
        ordered_cols = ["coop. request sent", "first lane change starts",
                        "last lane change ends", "platoon back together"]
        maneuver_times = maneuver_times[ordered_cols]
        maneuver_times = maneuver_times.stack().reset_index()
        maneuver_times.columns = ["Strategy", "Phase", "time_exact"]

        return maneuver_times

    def _prepare_data_collection_data(
            self, data: pd.DataFrame,
            sensor_identifier: Union[list[int], str] = None,
            aggregate_sensors: bool = True,
            warmup_time: float = 10,
            aggregation_period: int = 30) -> pd.DataFrame:
        """
        Keeps only data from given sensors, discards data before warmup time,
        and computes flow for the given aggregation period.
        """
        # Drop early samples
        post_processing.drop_warmup_samples(data, warmup_time)
        # Select fata from the sensors
        if self.scenario_name.startswith("in_and_out"):
            selected_sensor_data = (
                _select_flow_sensors_from_in_and_out_scenario(
                    data, sensor_identifier)
            )
            aggregate_sensors = False
        else:
            selected_sensor_data = data.drop(data[~data["sensor_number"].isin(
                sensor_identifier)].index)
        # Aggregate sensors
        if aggregate_sensors:
            selected_sensor_data.reset_index(drop=True, inplace=True)
            n_sensors = len(sensor_identifier)
            aggregated_data = selected_sensor_data.iloc[
                [i for i in range(0, selected_sensor_data.shape[0],
                                  n_sensors)]].copy()
            aggregated_data["vehicle_count"] = selected_sensor_data[
                "vehicle_count"].groupby(
                selected_sensor_data.index // n_sensors).sum().to_numpy()
            aggregated_data = _aggregate_data_over_time(
                aggregated_data, "vehicle_count", aggregation_period, np.sum)
        else:
            temp = []
            data_per_sensor = selected_sensor_data.groupby("sensor_number")
            for _, group in data_per_sensor:
                temp.append(_aggregate_data_over_time(
                    group, "vehicle_count", aggregation_period, np.sum))
            aggregated_data = pd.concat(temp)
        # Aggregate time
        # aggregated_data.sort_values(["sensor_number", "time"], kind="stable",
        #                             inplace=True)

        aggregated_data["flow"] = (3600 / aggregation_period
                                   * aggregated_data["vehicle_count"])
        return aggregated_data

    def _prepare_link_evaluation_data(
            self, data: pd.DataFrame, link: int, segment: int = None,
            lanes: list[int] = None, sensor_name: str = None,
            aggregate_lanes: bool = True, warmup_time: float = 10,
            sim_time: float = None,
            aggregation_period: int = 30) -> pd.DataFrame:
        # Drop early samples
        post_processing.drop_warmup_samples(data, warmup_time,
                                            normalize_time=True)
        post_processing.drop_late_samples(data, sim_time)
        # Select link
        data.drop(index=data[data["link_number"] != link].index,
                  inplace=True)
        # Select segment
        if segment is not None:
            data.drop(index=data[data["link_segment"] != segment].index,
                      inplace=True)
        elif data["link_segment"].nunique() > 1:
            print("WARNING: the chosen link has several segments, and we are "
                  "keeping all of them")
        # Select lanes
        if self.scenario_name.startswith("in_and_out"):
            lanes = _select_lanes_for_in_and_out_scenario(data, sensor_name)
        elif lanes is not None and _has_per_lane_results(data):
            data.drop(data[~data["lane"].isin(lanes)].index,
                      inplace=True)
        # Aggregate
        aggregation_map = {"density": np.sum, "volume": np.sum,
                           "average_speed": np.mean, "delay_relative": np.mean}
        relevant_columns = list(aggregation_map.keys())
        if lanes is not None and aggregate_lanes:
            data.reset_index(drop=True, inplace=True)
            n_lanes = len(lanes)
            aggregated_data = data.iloc[[i for i in range(0, data.shape[0],
                                                          n_lanes)]].copy()
            aggregated_data[relevant_columns] = data[
                relevant_columns].groupby(
                data.index // n_lanes).agg(aggregation_map).to_numpy()
            aggregated_data = _aggregate_data_over_time(
                aggregated_data, relevant_columns,
                aggregation_period, np.mean)
        else:
            temp = []
            data_per_lane = data.groupby("lane")
            for _, group in data_per_lane:
                temp.append(_aggregate_data_over_time(
                    group, relevant_columns, aggregation_period,
                    np.mean))
            aggregated_data = pd.concat(temp)

        return aggregated_data.fillna(0)

    def _prepare_data_for_plotting(
            self, y: str, data: pd.DataFrame, warmup_time: float = 0,
            sim_time: float = None, sensor_name: list[str] = None,
            pollutant_id: int = None) -> pd.DataFrame:
        """
        Performs several operations to make the data proper for plotting:
        1. Fill NaN entries in columns describing controlled vehicle
        percentage
        2. Aggregates data from all columns describing controlled vehicle
        percentage into a single "control percentages" column
        3. [Optional] Removes samples before warm-up time
        4. [Optional] Filter out certain sensor groups
        :param data: data aggregated over time
        :param warmup_time: Samples earlier than warmup_time are dropped.
         Must be passed in minutes
        :param sensor_name: if plotting flow or density, we can determine
         which sensor/lane is shown
        """
        if y == "flow":
            processed_data = self._prepare_data_collection_data(
                data, sensor_name, warmup_time=warmup_time)
        elif y in {"density", "relative_delay", "averga_seed", "volume"}:
            link = self.file_handler.get_main_links()[0]
            processed_data = self._prepare_link_evaluation_data(
                data, link, sensor_name=sensor_name, warmup_time=warmup_time)
        elif y == "emission_per_volume":
            processed_data = ResultAnalyzer._prepare_pollutant_data(
                data, pollutant_id)
        elif "risk" in y:
            processed_data = ResultAnalyzer._prepare_risk_data(data, y,
                                                               warmup_time, 0)
        else:
            post_processing.drop_warmup_samples(data, warmup_time,
                                                normalize_time=True)
            post_processing.drop_late_samples(data, sim_time)
            processed_data = data
        return processed_data

    @staticmethod
    def _prepare_pollutant_data(data: pd.DataFrame,
                                pollutant_id: int) -> pd.DataFrame:
        return data.drop(data[~(data["pollutant_id"] == pollutant_id)].index)

    @staticmethod
    def _prepare_risk_data(data: pd.DataFrame, risky_type: str,
                           warmup_time: float, min_risk: float) -> pd.DataFrame:
        """
        Removes samples before warmup time and with risk below min_risk
        :param data: Any VISSIM output or post processed data
        :param warmup_time: Must be in minutes
        :param min_risk: Threshold value
        """
        post_processing.drop_warmup_samples(data, warmup_time)
        return data.drop(index=data[data[risky_type] < min_risk].index)

    @staticmethod
    def _prepare_travel_time_data(data: pd.DataFrame, warmup_time: float = 0,
                                  sim_time: float = None) -> pd.DataFrame:
        """
        Computes the travel time, and removes samples which entered the
        simulation before warmup time or after warmup plus simulation time.
        Modifies data in place.
        """
        # Drop vehicles that didn't have time to exit the network (temporary,
        # because this will be dealt with in the post-processing step)
        data.drop(index=data[data["exit_time"]
                             == data["exit_time"].max()].index, inplace=True)
        data["travel_time"] = data["exit_time"] - data["entrance_time"]
        warmup_time *= 60
        if sim_time:
            cutoff_time = sim_time * 60 + warmup_time
        else:
            cutoff_time = data["entrance_time"].max() + 1
        data.drop(index=data[(data["entrance_time"] <= warmup_time)
                             | (data["entrance_time"] >= cutoff_time)].index,
                  inplace=True)
        data["entrance_time"] -= warmup_time
        data["delta_v"] = (data["dest_lane_speed"].astype(int)
                           - data["orig_lane_speed"].astype(int))

        # Pretty names
        lane_map = {1: "Origin", 2: "Destination"}
        data["lane"] = data["lane"].map(lane_map)
        return data.rename(columns={"lane_change_strategy": "Strategy",
                                    "lane": "Lane"})

    @staticmethod
    def _select_sensors_for_platoon_lc_scenario(
            before_or_after_lc_point: str, lanes: str
    ) -> tuple[list[int], list[int]]:
        if lanes == "orig":
            lane_numbers = [1]
        elif lanes == "dest":
            lane_numbers = [2]
        elif lanes == "both":
            lane_numbers = [1, 2]
        else:
            raise ValueError("Parameter lanes must be 'orig', 'dest', "
                             "or 'both'.")
        if before_or_after_lc_point == "before":
            in_flow_sensors = [1, 2]
            out_flow_sensors = [3, 4]
        elif before_or_after_lc_point == "after":
            in_flow_sensors = [3, 4]
            out_flow_sensors = [5, 6]
        else:
            raise ValueError("Parameter 'before_or_after_lc_point' must be "
                             "either 'before' or 'after'")
        in_flow_sensors = in_flow_sensors[lane_numbers[0] - 1
                                          :lane_numbers[-1]]
        out_flow_sensors = out_flow_sensors[lane_numbers[0] - 1
                                            :lane_numbers[-1]]
        return in_flow_sensors, out_flow_sensors

    def _select_link_and_lanes(self, before_or_after_lc_point: str = None,
                               lanes: str = None) -> tuple[int, list[int]]:
        if "in_and_out" in self.scenario_name:
            link = FileHandler(self.scenario_name).get_main_links()[0]
            lane_numbers = [1, 2, 30]
        elif "platoon" in self.scenario_name:
            link, lane_numbers = (
                self._select_link_and_lanes_for_two_lane_highway(
                    before_or_after_lc_point, lanes))
        elif "risky" in self.scenario_name:
            link, lane_numbers = link, lane_number = (
                self._select_link_and_lanes_for_two_lane_highway(
                    before_or_after_lc_point, lanes))
        else:
            raise ValueError("Evaluation link and lanes not defined for "
                             "scenario %s".format(self.scenario_name))
        return link, lane_numbers

    def _select_link_and_lanes_for_two_lane_highway(
            self, before_or_after_lc_point: str, lanes: str
    ) -> tuple[int, list[int]]:
        if lanes == "orig":
            lane_numbers = [1]
        elif lanes == "dest":
            lane_numbers = [2]
        elif lanes == "both":
            lane_numbers = [1, 2]
        else:
            raise ValueError("Parameter lanes must be 'orig', 'dest', "
                             "or 'both'.")
        main_links = self.file_handler.get_main_links()
        if before_or_after_lc_point == "before":
            link = main_links[0]
        elif before_or_after_lc_point == "after":
            link = main_links[1]
        else:
            raise ValueError("Parameter 'before_or_after_lc_point' must be "
                             "either 'before' or 'after'")
        return link, lane_numbers

    @staticmethod
    def _check_if_data_is_uniform(data: pd.DataFrame) -> None:
        """
        Checks whether all loaded scenarios were run with the same random seeds.
        :param data:
        :return:
        """
        grouped = data.groupby("random_seed")
        group_sizes = grouped["simulation_number"].count()
        if any(group_sizes != group_sizes.iloc[0]):
            print("Not all scenarios have the same number of samples.\n"
                  "This might create misleading plots.")

    def _make_data_uniform(self, data: pd.DataFrame,
                           use_all_simulations: bool = True) -> pd.DataFrame:
        """
        During tests, some strategies might be run with more random seeds or
        longer than others. This function will keep only the simulation
        parameters used by all strategies.
        """
        if "lane_change_strategy" not in data.columns:
            return data
        grouped = data.groupby("lane_change_strategy")
        max_time = grouped["time"].max().min()
        if use_all_simulations:
            max_simulations = grouped["simulation_number"].max().min()
        else:
            max_simulations = 1

        cropped_data = data[(data["simulation_number"] <= max_simulations)
                            & (data["time"] <= max_time)]
        if cropped_data.shape[0] < data.shape[0]:
            print("Keeping only data up to simulation {} "
                  "and until time {}".format(max_simulations, max_time))
        return cropped_data.copy()

    @staticmethod
    def remove_deadlock_simulations(data) -> None:
        deadlock_entries = (data.loc[
                                data["flow"] == 0,
                                ["vehicles_per_lane", "random_seed"]
                            ].drop_duplicates())
        for element in deadlock_entries.values:
            idx = data.loc[(data["vehicles_per_lane"] == element[0])
                           & (data["random_seed"] == element[1])].index
            data.drop(idx, inplace=True)
            print("Removed results from simulation with input {}, random "
                  "seed {} due to deadlock".
                  format(element[0], element[1]))

    def save_fig(self, fig: plt.Figure, plot_type: str = None,
                 measurement_name: str = None,
                 scenarios: list[scenario_handling.ScenarioInfo] = None,
                 fig_name: str = None
                 ) -> None:
        """
        Saves the figure. We must provide either the fig_name or
        enough information for the figure name to be automatically generated
        """
        if not fig_name:
            # TODO: should be a function
            vehicle_percentage_strings = []
            vehicles_per_lane_strings = set()
            accepted_risk_strings = set()
            for sc in scenarios:
                vehicles_per_lane_strings.add(str(sc.vehicles_per_lane))
                vehicle_percentages = sc.vehicle_percentages
                vp_str = ["_".join([str(p), vt.name.lower])
                          for vt, p in vehicle_percentages.items()]
                vehicle_percentage_strings.extend(vp_str)
                if sc.accepted_risk is not None:
                    accepted_risk_strings.add(str(sc.accepted_risk))
            all_vehicles_per_lane = ("_".join(sorted(
                vehicles_per_lane_strings)) + "_vehs_per_lane")
            all_vehicle_percentages = "_".join(sorted(
                vehicle_percentage_strings))
            fig_name = "_".join(
                [plot_type, measurement_name, self.scenario_name,
                 all_vehicles_per_lane, all_vehicle_percentages])
            if len(accepted_risk_strings) > 0:
                all_risks = "risks_" + "_".join(sorted(accepted_risk_strings))
                fig_name = "_".join([fig_name, all_risks])
        fig.set_dpi(400)
        # axes = fig.axes
        plt.tight_layout()
        print("Figure name:", fig_name)
        # a = input("Save? [y/n]")
        # if a != "y":
        #     print("not saving")
        # else:
        #     print("saving")
        fig.savefig(os.path.join(self._figure_folder, fig_name), dpi=400)

    def create_figure_name(
            self, plot_type: str, measurement_name: str,
            scenarios: list[scenario_handling.ScenarioInfo]) -> str:

        # TODO [March 1, 2023]: not tested for platoon scenarios
        vehicles_per_lane_strings = set()
        percentage_strings = set()
        vehicle_type_strings = set()
        accepted_risk_strings = set()
        for sc in scenarios:
            vehicles_per_lane_strings.add(str(sc.vehicles_per_lane))
            veh_percentages = sc.vehicle_percentages
            if sum(veh_percentages.values()) == 0:
                veh_percentages = {VehicleType.HDV: 100}
            for vt, p in veh_percentages.items():
                percentage_strings.add(str(p))
                vehicle_type_strings.add(vt.name.lower())
            if sc.accepted_risk is not None:
                accepted_risk_strings.add(str(sc.accepted_risk))
        all_vehicles_per_lane = ("_".join(sorted(
            vehicles_per_lane_strings)) + "_vehs_per_lane")
        all_percentage_strings = "_".join(sorted(percentage_strings))
        all_vehicle_type_strings = "_".join(sorted(vehicle_type_strings))
        fig_name = "_".join(
            [plot_type, measurement_name, self.scenario_name,
             all_vehicles_per_lane, all_percentage_strings,
             all_vehicle_type_strings])
        if len(accepted_risk_strings) > 1:
            all_risks = "risks_" + "_".join(sorted(accepted_risk_strings))
            fig_name = "_".join([fig_name, all_risks])
        return fig_name

    @staticmethod
    def widen_fig(fig: plt.Figure, n_boxes: int) -> None:
        if n_boxes >= 4:
            fig.set_size_inches(6.4 * 2, 4.8)

    # Plots for a single simulation - OUTDATED: might not work =============== #

    # These methods require post-processed data with SSMs already computed #
    def plot_ssm(self, ssm_names, vehicle_record) -> None:
        """Plots the sum of the surrogate safety measure over all vehicles
        versus time
        Plotting this way is not recommended as we get too much noise.
        Better to use the moving average plot.

        :param vehicle_record: dataframe with the vehicle record data
        :param ssm_names: list of strings with the desired surrogate safety
         measures"""

        if isinstance(ssm_names, str):
            ssm_names = [ssm_names]

        fig, ax = plt.subplots()
        aggregated_data = (vehicle_record[["time"].extend(ssm_names)].
                           groupby("time").sum())
        for ssm in ssm_names:
            if ssm in self._ssm_pretty_name_map:
                ssm_plot_name = self._ssm_pretty_name_map[ssm]
            else:
                ssm_plot_name = ssm
            ax.plot(aggregated_data[ssm].index, aggregated_data[ssm][ssm],
                    label=ssm_plot_name + " (" + _get_units(ssm) + ")")
        ax.legend()
        plt.show()

    def plot_ssm_moving_average(self, vehicle_record, ssm_name,
                                window=100, save_path=None) -> None:
        """
        Plots the moving average of the surrogate safety measure over all
        vehicles versus time
        :param vehicle_record: dataframe with the vehicle record data
        :param window: moving average window
        :param ssm_name: string with the desired surrogate safety measure
        :param save_path: path including folder and simulation name. The
        function adds the ssm_name to the figure name.
        """
        label_font_size = 18

        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots()
        aggregated_data = vehicle_record[["time", ssm_name]].groupby(
            "time").sum()
        # sns.set_theme()
        aggregated_data["Mov. Avg. " + ssm_name] = aggregated_data[
            ssm_name].rolling(window=window, min_periods=10).mean()
        sns.lineplot(x=aggregated_data.index,
                     y=aggregated_data["Mov. Avg. " + ssm_name],
                     ax=ax)
        ax.set_xlabel("time (s)", fontsize=label_font_size)
        if ssm_name in self._ssm_pretty_name_map:
            ssm_plot_name = self._ssm_pretty_name_map[ssm_name]
        else:
            ssm_plot_name = ssm_name
        ax.set_ylabel(ssm_plot_name + " (" + _get_units(ssm_name) + ")",
                      fontsize=label_font_size)
        plt.show()

        if save_path is not None:
            fig.savefig(save_path + ssm_name)

    @staticmethod
    def plot_risk_counter(ssm_name, vehicle_record_data) -> None:
        """Temporary function to compare how the exact and estimated risks
        vary over time"""
        fig, ax = plt.subplots()
        ssm_counter = ssm_name + "_counter"
        vehicle_record_data[ssm_counter] = vehicle_record_data[ssm_name] > 0
        aggregated_data = vehicle_record_data[["time", ssm_counter]].groupby(
            np.floor(vehicle_record_data["time"])).sum()
        ax.plot(aggregated_data.index, aggregated_data[ssm_counter],
                label=ssm_counter)
        ax.legend()
        plt.show()

    # Data integrity checks ================================================== #

    def find_unfinished_simulations(self, scenarios: list[
        scenario_handling.ScenarioInfo]
                                    ) -> None:
        """
        Checks whether simulations crashed. This is necessary because,
        when doing multiple runs from the COM interface, VISSIM does not
        always indicate that a simulation crashed.

        :param scenarios: List of simulation parameters for several scenarios
        """
        raw_data = self._load_data("flow", scenarios)
        data = self._prepare_data_collection_data(raw_data, "in",
                                                  warmup_time=0,
                                                  aggregation_period=60)
        grouped = data.groupby(
            ["control percentages", "vehicles_per_lane", "accepted_risk",
             "simulation_number"])["flow"]
        all_end_times = grouped.last()
        issues = all_end_times[all_end_times == 0]
        if issues.empty:
            print("All simulations seem to have run till the end")
            return
        # Find the last valid simulation time
        issue_time = []
        for i in issues.index:
            print(i)
            s = grouped.get_group(i)
            issue_time.append(data.loc[s.index[
                s.to_numpy().nonzero()[0][-1]], "time"])
            print(issue_time[-1])
        print("Min issue time {} at simulation {}".format(
            min(issue_time), issues.index[np.argmin(issue_time)]))


# Some methods used by the class ============================================= #

def _make_pretty_label(var: str) -> str:
    with_unit = " ".join(var.split("_")) + " (" + _get_units(var) + ")"
    return with_unit.replace("delta v", "$\\Delta v$")


def _get_units(var: str) -> str:
    var = var.lower()
    if "time" in var:
        y = "time"
    elif "risk" in var:
        y = "risk"
    elif "speed" in var or "delta_v" in var:
        y = "speed"
    elif "count" in var or "size" in var:
        y = "count"
    elif "vehicles" in var:
        y = "flow"
    elif "energy" in var:
        y = "energy"
    else:
        y = var
    units_map = {"TTC": "s", "low_TTC": "# vehicles", "DRAC": "m/s^2",
                 "high_DRAC": "# vehicles", "CPI": " ", "DTSG": "m",
                 "risk": "m/s", "flow": "veh/h", "volume": "veh/h",
                 "density": "veh/km", "time": "s", "speed": "km/h",
                 "delay_relative": "%", "count": "# vehs", "energy": "J"}
    return units_map[y]


def list_of_dicts_to_1d_list(dict_list: list[dict]) -> tuple[list, list]:
    keys = []
    values = []
    for d in dict_list:
        keys += list(d.keys())
        values += list(d.values())
    return keys, values


def vehicle_percentage_dict_to_string(vp_dict: dict[VehicleType, int]) -> str:
    if sum(vp_dict.values()) == 0:
        return "100% HDV"
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + "% " + veh_type.get_print_name())
    return " ".join(sorted(ret_str))


def _aggregate_data_over_time(data: pd.DataFrame,
                              aggregated_variable: Union[str, list[str]],
                              aggregation_period: int, aggregation_function
                              ) -> pd.DataFrame:
    data.reset_index(drop=True, inplace=True)
    interval_per_group = data.groupby("control percentages")[
        "time_interval"].first()
    if np.any(interval_per_group.iloc[0] != interval_per_group):
        # Samples with different sampling intervals
        aggregated_per_group = []
        for key, group in data.groupby("control percentages"):
            aggregated_per_group.append(_aggregate_data_over_time(
                group, aggregated_variable, aggregation_period,
                aggregation_function))
        aggregated_data = pd.concat(aggregated_per_group)
    else:
        t1, t2 = data["time_interval"].iloc[0].split("-")
        interval = int(t2) - int(t1)
        if aggregation_period % interval != 0:
            print("Aggregation period should be a multiple of the data "
                  "collection interval used in simulation. Keeping the "
                  "original data collection interval.")
            aggregation_period = interval
        n = aggregation_period // interval
        aggregated_data = data.iloc[[i for i in range(0, data.shape[0],
                                                      n)]].copy()
        aggregated_data[aggregated_variable] = data[
            aggregated_variable].groupby(
            data.index // n).agg(aggregation_function).to_numpy()
    return aggregated_data


def _has_per_lane_results(data: pd.DataFrame) -> bool:
    """
    :param data: Data from link evaluation records
    :return: boolean indicating whether the data has results individualized
    per lane
    """
    if "lane" in data.columns and data["lane"].iloc[0] != 0:
        return True
    return False


def _select_flow_sensors_from_in_and_out_scenario(
        data: pd.DataFrame, sensor_name: str = None) -> pd.DataFrame:
    """
    Keeps only data from the sensors with the given sensor names.
    """
    if sensor_name is None:
        sensor_name = "in"
    sensor_name_map = {
        "in": 1, "out": 2,
        "in_main_left": 3, "in_main_right": 4, "in_ramp": 5,
        "out_main_left": 6, "out_main_right": 7, "out_ramp": 8
    }
    sensor_number = sensor_name_map[sensor_name]
    return data.drop(data[data["sensor_number"] != sensor_number].index)


def _select_lanes_for_in_and_out_scenario(data: pd.DataFrame,
                                          sensor_name: str = None) -> list[int]:
    """
    Chooses which lanes from the link evaluation data to keep based on the
    sensor names.
    """
    if sensor_name is None:
        sensor_name = "in"
    sensor_lane_map = {
        "in": [1, 2, 3], "out": [1, 2, 3],
        "in_main_left": [1], "in_main_right": [2], "in_ramp": [3],
        "out_main_left": [1], "out_main_right": [2], "out_ramp": [3]
    }
    lanes = None
    if _has_per_lane_results(data):
        lanes = sensor_lane_map[sensor_name]
        data.drop(data[~data["lane"].isin(lanes)].index,
                  inplace=True)
    return lanes


def _plot_heatmap(
        data: pd.DataFrame, value: str, rows: str, columns: str,
        normalizer: int = 1, agg_function=np.sum,
        custom_colorbar_range: bool = False,
        show_figure: bool = True, fill_na_cols: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure()
    plt.rc("font", size=17)
    table = data.pivot_table(values=value,
                             index=[columns],
                             columns=[rows],
                             aggfunc=agg_function, sort=False)
    if normalizer > 0:
        table /= normalizer
    if fill_na_cols:
        table.ffill(inplace=True)
    max_value = np.ceil(table.max().max())
    min_value = np.floor(table.min().min())
    if custom_colorbar_range:
        vmin, vmax = min_value, max_value
    else:
        vmin, vmax = None, None
    # string_format = ".2f" if max_value < 1000 else ".2g"
    string_format = ".2f"
    ax = sns.heatmap(table, annot=True, vmin=vmin, vmax=vmax, fmt=string_format)
    if show_figure:
        title = " ".join(value.split("_"))
        ax.set_title(title, fontsize=22)
        fig.tight_layout()
        fig.show()
    return fig, ax


def _my_boxplot(relevant_data: pd.DataFrame, x: str, y: str,
                hue: str = "vehicles per hour", will_show: bool = True
                ) -> tuple[plt.Figure, plt.Axes]:
    # Plot
    fig = plt.figure()
    plt.rc("font", size=22)
    fig.set_size_inches(12, 6)
    ax = sns.boxplot(data=relevant_data, x=x, y=y, hue=hue)
    ax.legend(loc="upper center",  # bbox_to_anchor=(0.5, 1.1),
              title=hue, ncols=2)
    if will_show:
        plt.tight_layout()
        plt.show()
    return fig, ax


def _fit_fundamental_diagram(data: pd.DataFrame, by: str) -> None:
    """
    :param data: Link evaluation treated data
    :param by: How to split the data; must be column of data
    """
    grouped = data.groupby(by)
    coefficients = []
    fig, ax = plt.subplots()
    for group_id, group in grouped:
        find_critical_density(group)
        # ax.plot(x, y, linewidth=2.0, label=group_id)
        ax.scatter(group["density"], group["volume"])
    ax.legend()
    plt.show()


def find_critical_density(data) -> None:
    max_density = data["density"].max()
    reg_u = LinearRegression(fit_intercept=False)
    reg_c = LinearRegression()
    best_d = 0
    best_coefficients = (0, 0)
    best_score = np.float("inf")
    for d in np.linspace(max_density / 4, 3 * max_density / 4, 10):
        idx = data["density"] < d
        uncongested = data[idx]
        congested = data[~idx]
        x_u = uncongested["density"].to_numpy().reshape(-1, 1)
        y_u = uncongested["volume"]
        x_c = congested["density"].to_numpy().reshape(-1, 1)
        y_c = congested["volume"]

        reg_u.fit(x_u, y_u)
        reg_c.fit(x_c, y_c)
        y_u_hat = reg_u.predict(x_u)
        y_c_hat = reg_c.predict(x_c)
        mse = ((y_u - y_u_hat) ** 2 + (y_c - y_c_hat) ** 2) / data.shape[0]
        if mse < best_score:
            best_score = mse
            best_d = d
            best_coefficients = (reg_u.coef_, reg_c.coef_)


def _produce_console_output(
        data: pd.DataFrame, y: str,
        scenarios: list[scenario_handling.ScenarioInfo], aggregation_functions,
        show_variation: bool = False,
        normalizer: int = None) -> None:
    if normalizer is None:
        try:
            normalizer = data["simulation_number"].nunique()
        except KeyError:
            normalizer = 1
    group_by_cols = ["vehicles per hour", "control percentages"]

    mixed_traffic = False
    all_accepted_risks = set()
    for sc in scenarios:
        if sc.accepted_risk is not None:
            all_accepted_risks.add(sc.accepted_risk)
        if any(0 < p < 100 for p in sc.vehicle_percentages.values()):
            mixed_traffic = True

    if len(all_accepted_risks) > 1:
        group_by_cols += ["accepted_risk"]

    print(y)
    # Easier to transfer results to paper in this order
    data.sort_values("vehicles per hour",
                     kind="stable", inplace=True)
    result = data.groupby(group_by_cols, sort=False)[y].agg(
        aggregation_functions) / normalizer
    if show_variation:
        baselines = result.groupby(level=0).first()
        variation = post_processing.find_percent_variation(baselines, result)
        print(baselines)
        print(variation)
    else:
        print(result)
    if mixed_traffic and "veh_type" in data.columns:
        group_by_cols += ["veh_type"]
        result_by_type = data.groupby(group_by_cols, sort=False)[y].agg(
            aggregation_functions) / normalizer
        print(result_by_type)
        print(result_by_type / result)
