import os
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

import file_handling
import post_processing
import readers
from vehicle import VehicleType, vehicle_type_to_str_map


class ResultAnalyzer:
    _data_reader_map = {
        'flow': readers.DataCollectionReader,
        'vehicle_count': readers.DataCollectionReader,
        'density': readers.LinkEvaluationReader,
        'volume': readers.LinkEvaluationReader,
        'risk': readers.SSMDataReader,
        'barrier_function_risk': readers.SSMDataReader,
        'total_risk': readers.RiskyManeuverReader,
        'discomfort': readers.DiscomfortReader,
        'violations': readers.ViolationsReader,
        'lane_change_count': readers.LaneChangeReader,
        'total_lane_change_risk': readers.LaneChangeReader,
        'initial_risk': readers.LaneChangeReader,
        'initial_risk_to_lo': readers.LaneChangeReader,
        'initial_risk_to_ld': readers.LaneChangeReader,
        'initial_risk_to_fd': readers.LaneChangeReader,
        'lane_change_issues': readers.LaneChangeIssuesReader,
        'fd_discomfort': readers.LaneChangeReader,
        'average_speed': readers.LinkEvaluationReader,
        'emission_per_volume': readers.MOVESDatabaseReader,
        'was_lane_change_completed': readers.PlatoonLaneChangeEfficiencyReader,
        'platoon_maneuver_time': readers.PlatoonLaneChangeEfficiencyReader,
        'travel_time': readers.PlatoonLaneChangeEfficiencyReader,
        'accel_cost': readers.PlatoonLaneChangeEfficiencyReader,
        'stayed_in_platoon': readers.PlatoonLaneChangeEfficiencyReader
    }

    _units_map = {'TTC': 's', 'low_TTC': '# vehicles',
                  'DRAC': 'm/s^2', 'high_DRAC': '# vehicles',
                  'CPI': 'dimensionless', 'DTSG': 'm',
                  'risk': 'm/s', 'estimated_risk': 'm/s',
                  'flow': 'veh/h', 'density': 'veh/km',
                  'time': 'min', 'time_interval': 's'}

    _ssm_pretty_name_map = {'low_TTC': 'Low TTC',
                            'high_DRAC': 'High DRAC',
                            'CPI': 'CPI',
                            'risk': 'CRI'}

    _strategy_name_map = {-1: 'humans', 0: 'CAVs', 1: 'SBP', 2: 'Ld First',
                          3: 'LV First', 4: 'Ld First Rev.'}

    # TODO: these two only work for the in_and_out scenarios. Reorganize somehow
    _sensor_name_map = {
        'in': 1, 'out': 2,
        'in_main_left': 3, 'in_main_right': 4, 'in_ramp': 5,
        'out_main_left': 6, 'out_main_right': 7, 'out_ramp': 8
    }
    _sensor_lane_map = {
        'in': [1, 2, 3], 'out': [1, 2, 3],
        'in_main_left': [1], 'in_main_right': [2], 'in_ramp': [3],
        'out_main_left': [1], 'out_main_right': [2], 'out_ramp': [3]
    }

    _pollutant_id_to_string = {
        1: 'Gaseous Hydrocarbons', 5: 'Methane (CH4)',
        6: 'Nitrous Oxide (N2O)', 90: 'CO2',
        91: 'Energy Consumption', 98: 'CO2 Equivalent',
    }

    def __init__(self, scenario_name: str, should_save_fig: bool = False,
                 is_debugging: bool = False):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
            self._figure_folder = ('C:\\Users\\fvall\\Google Drive\\'
                                   'PhD Research\\Lane Change\\images')
        else:
            self._figure_folder = ('G:\\My Drive\\PhD Research\\Lane Change'
                                   '\\images')
        self.file_handler = file_handling.FileHandler(scenario_name)
        self.scenario_name = scenario_name
        self.should_save_fig = should_save_fig
        self.is_debugging = is_debugging

    # Plots aggregating results from multiple simulations ==================== #
    def plot_xy(self, x: str, y: str, hue: str,
                scenarios: List[file_handling.ScenarioInfo],
                warmup_time: int = 0):
        """

        """
        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue, ci='sd')
        plt.show()
        return ax

    def plot_y_vs_time(self, y: str,
                       scenarios: List[file_handling.ScenarioInfo],
                       warmup_time: int = 10):
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
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=data, x='time', y=y,
                          hue='simulation_number', errorbar='sd')
        if self.should_save_fig:
            self.save_fig(plt.gcf(), 'time_plot', y, scenarios)
        plt.show()
        return ax

    def plot_fundamental_diagram(
            self, scenarios: List[file_handling.ScenarioInfo],
            link_number: int,
            link_segment_number: int = None, lanes: List[int] = None,
            hue: str = None, col: str = None,
            aggregation_period: int = 30, warmup_time: int = 10,
            will_plot: bool = False):
        """
        Computes flow from link evaluation data.
        """
        plt.rc('font', size=15)
        aggregate_lanes = True
        if hue == 'lane' or col == 'lane':
            aggregate_lanes = False

        link_eval_data = self._load_data(
            'density', scenarios)
        link_eval_data = self._prepare_link_evaluation_data(
            link_eval_data, link_number, link_segment_number, lanes,
            aggregate_lanes=aggregate_lanes, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # ResultAnalyzer._check_if_data_is_uniform(link_eval_data)
        if col is None:
            ax = sns.scatterplot(data=link_eval_data, x='density', y='volume',
                                 hue=hue, palette='tab10')
            ax.set_xlabel('density (veh/km)')
            ax.set_ylabel('flow (veh/h)')
            axes = [ax]
        else:
            rp = sns.relplot(data=link_eval_data, x='density', y='volume',
                             hue=hue, col=col, col_wrap=2,
                             palette='tab10')
            axes = []
            for ax in rp.axes:
                ax.set_xlabel('density (veh/km)')
                ax.set_ylabel('flow (veh/h)')
                axes.append(ax)

        if will_plot:
            plt.tight_layout()
            plt.show()

        return axes

    def plot_fundamental_diagram_from_flow(
            self, scenarios: List[file_handling.ScenarioInfo],
            flow_sensor_identifier: Union[str, int] = None,
            link_segment_number: int = None,
            aggregation_period: int = 30, warmup_time: int = 10):
        """
        Computes flow from data collection sensors and uses density from link
        evaluation.
        """

        density_data = self._load_data('density', scenarios)
        flow_data = self._load_data('flow', scenarios)

        flow_data = self._prepare_data_collection_data(
            flow_data, flow_sensor_identifier, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # We're assuming scenarios with a single main link
        link = self.file_handler.get_main_links()[0]
        density_data = self._prepare_link_evaluation_data(
            density_data, link, link_segment_number,
            warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        intersection_columns = ['vehicles_per_lane', 'control percentages',
                                'simulation_number', 'time_interval',
                                'random_seed']
        data = flow_data.merge(density_data, on=intersection_columns)

        # ResultAnalyzer._check_if_data_is_uniform(data)
        sns.scatterplot(data=data, x='density', y='flow',
                        hue='control percentages')
        plt.show()

    def plot_fundamental_diagram_per_control_percentage(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, aggregation_period: int = 30):
        main_link = self.file_handler.get_main_links()[0]
        self.plot_fundamental_diagram(
            scenarios, main_link, hue='control percentages',
            aggregation_period=aggregation_period,
            warmup_time=warmup_time)
        plt.tight_layout()
        plt.show()

    def plot_flow_box_plot_vs_controlled_percentage(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10,
            flow_sensor_name: str = 'in',
            aggregation_period: int = 30):

        # for sc in scenarios:
        #     if sc.accepted_risk is None:
        #         sc.accepted_risk = 0
        data = self._load_data('flow', scenarios)
        relevant_data = self._prepare_data_collection_data(
            data, flow_sensor_name, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # ResultAnalyzer._check_if_data_is_uniform(relevant_data)
        fig, ax = _my_boxplot(relevant_data, 'control percentages', 'flow',
                              'vehicles per hour')
        if self.should_save_fig:
            fig_name = self.create_figure_name('box_plot', 'flow', scenarios)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(relevant_data, 'flow', scenarios, np.median,
                                show_variation=True)

    def plot_volume_box_plot_vs_controlled_percentage(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, aggregation_period: int = 30):

        y = 'volume'
        link = file_handling.FileHandler(self.scenario_name).get_main_links()[0]
        data = self._load_data(y, scenarios)
        relevant_data = self._prepare_link_evaluation_data(
            data, link, lanes=[1, 2, 3], warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        fig, ax = _my_boxplot(relevant_data, 'control percentages', y,
                              'vehicles per hour')
        if self.should_save_fig:
            fig_name = self.create_figure_name('box_plot', y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(relevant_data, y, scenarios, np.median,
                                show_variation=True)

    def box_plot_y_vs_controlled_percentage(
            self, y: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10):
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
        fig, ax = _my_boxplot(data, 'control percentages', y,
                              'vehicles per hour')
        if self.should_save_fig:
            print('Must double check whether fig is being saved')
            self.save_fig(fig, 'box_plot', y, scenarios)

    def box_plot_y_vs_vehicle_type(
            self, y: str, hue: str,
            scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, sensor_name: str = None):
        """
        Plots averaged y over several runs with the same vehicle input
        versus vehicles type as a box plot. Parameter hue controls how to
        split data for each vehicle type.

        :param y: name of the variable being plotted.
        :param hue: must be 'percentage' or 'accepted_risk'
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param sensor_name: if plotting flow, we can determine choose
         which data collection measurement will be shown
        """
        data = self._load_data(y, scenarios)
        self._prepare_data_for_plotting(y, data, warmup_time, sensor_name)
        # ResultAnalyzer._check_if_data_is_uniform(data)

        no_control_idx = (data['control percentages']
                          == '100% HDV')
        data[['percentage', 'control_type']] = data[
            'control percentages'].str.split(' ', expand=True)
        data.loc[no_control_idx, 'control_type'] = 'human'
        data.loc[no_control_idx, 'percentage'] = '0%'
        data['Accepted Risk'] = data['accepted_risk'].map(
            {0: 'safe', 10: 'low', 20: 'medium', 30: 'high'}
        )
        if hue == 'accepted_risk':
            hue = 'Accepted Risk'

        # Plot
        plt.rc('font', size=25)
        sns.set_style('whitegrid')
        sns.boxplot(data=data, x='control_type', y=y, hue=hue)

        # Direct output
        _produce_console_output(data, y, scenarios, np.median)

        plt.legend(title=hue, ncol=1,
                   bbox_to_anchor=(1.01, 1))
        # plt.title(str(vehicles_per_lane) + ' vehs/h/lane', fontsize=22)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        if self.should_save_fig:
            self.save_fig(fig, 'box_plot', y, scenarios)
        # self.widen_fig(plt.gcf(), len(percentages_per_vehicle_types))
        plt.tight_layout()
        plt.show()

    def plot_risk_histograms(
            self, risk_type: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, min_risk: float = 0.1
    ):
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
        self._prepare_risk_data(data, risk_type, warmup_time, min_risk)
        plt.rc('font', size=28)
        sns.set_style('whitegrid')
        if ('accepted_risk' not in data.columns
                or data['accepted_risk'].nunique() <= 1):
            grouped_data = data.groupby('control percentages')
        else:
            grouped_data = data.groupby(['control percentages',
                                         'accepted_risk'])

        scenarios_per_vp = file_handling.split_scenario_by(
            scenarios, 'vehicle_percentages')
        for group_name, data_to_plot in grouped_data:
            if data_to_plot.empty:
                continue
            ax = sns.histplot(data=data_to_plot, x=risk_type,
                              stat='count', hue='vehicles per hour',
                              palette='tab10')
            ax.set_xlabel(risk_type.replace('_', ' '))
            ax.set_title(group_name)
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.tight_layout()
            plt.show()
            if self.should_save_fig:
                fig_name = self.create_figure_name(
                    'histogram', risk_type, scenarios_per_vp[group_name])
                self.save_fig(fig, fig_name=fig_name)

        normalizer = data.loc[
            (data['control percentages'] == '100% HDV')
            & (data['vehicles per hour'] == data[
                'vehicles per hour'].min()),
            risk_type].sum()
        _produce_console_output(data, risk_type, scenarios, [np.size, np.sum],
                                show_variation=True,
                                normalizer=normalizer)

    def plot_lane_change_risk_histograms_risk_as_hue(
            self, risk_type: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, min_risk: int = 0.1
    ):
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
        self._prepare_risk_data(data, risk_type, warmup_time, min_risk)
        data['risk'] = data['accepted_risk'].map({0: 'safe', 10: 'low',
                                                  20: 'medium', 30: 'high'})
        # ResultAnalyzer._check_if_data_is_uniform(data)

        plt.rc('font', size=30)
        # sns.set(font_scale=2)
        # for cp in control_percentages:
        for sc in scenarios:
            cp = vehicle_percentage_dict_to_string(sc.vehicle_percentages)
            if cp == '100% HDV':
                continue
            data_to_plot = data[
                (data['control percentages'] == cp)
            ]
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat='count', hue='risk',
                         palette='tab10')
            # plt.legend(title='Risk', labels=['safe', 'low', 'medium'])
            # Direct output
            print('veh penetration: {}'.format(cp))
            print(data_to_plot[['control percentages', 'veh_type', 'risk',
                                risk_type]].groupby(
                ['control percentages', 'veh_type', 'risk']).count())
            print(data_to_plot.groupby(
                ['control percentages', 'risk'])[risk_type].median())
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            if self.should_save_fig:
                self.save_fig(fig, 'histogram', risk_type, [sc])
            plt.show()

    def plot_row_of_lane_change_risk_histograms(
            self, risk_type: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, min_risk: int = 0.1
    ):
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
        ResultAnalyzer._prepare_risk_data(data, risk_type, warmup_time,
                                          min_risk)
        # ResultAnalyzer._check_if_data_is_uniform(data)

        all_accepted_risks = [sc.accepted_risk for sc in scenarios]

        n_risks = len(all_accepted_risks)
        fig, axes = plt.subplots(nrows=1, ncols=n_risks)
        fig.set_size_inches(12, 6)
        j = 0
        for ar, data_to_plot in data.groupby('accepted_risk'):
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat='count', hue='vehicles_per_lane',
                         palette='tab10', ax=axes[j])
            axes[j].set_title('accepted risk = ' + str(ar))
            j += 1

        plt.tight_layout()
        if self.should_save_fig:
            self.save_fig(fig, 'histogram_row_', risk_type, scenarios)
        plt.show()

    def plot_grid_of_lane_change_risk_histograms(
            self, risk_type: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, min_risk: int = 0.1
    ):
        """
        Creates a grid of size '# of vehicle types' vs '# of accepted risks'
        and plots the histogram of lane changes' total risks for each case

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type, scenarios)
        ResultAnalyzer._prepare_risk_data(data, risk_type, warmup_time,
                                          min_risk)
        # ResultAnalyzer._check_if_data_is_uniform(data)

        control_percentages = data['control percentages'].unique()
        accepted_risks = data['accepted_risk'].unique()
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
                    (data['control percentages'] == ctrl_percentage)
                    & (data['accepted_risk'] == ar)
                    ]
                sns.histplot(data=data_to_plot, x=risk_type,
                             stat='count', hue='vehicles_per_lane',
                             palette='tab10', ax=axes[i, j])
                if i + j > 0 and axes[i, j].get_legend():
                    axes[i, j].get_legend().remove()
                if i == 0:
                    axes[i, j].set_title('accepted risk = ' + str(ar))

        plt.tight_layout()
        if self.should_save_fig:
            self.save_fig(fig, 'histogram_grid', risk_type, scenarios)
        plt.show()

    def hist_plot_lane_change_initial_risks(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 0):
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
        data = self._load_data('initial_risk', scenarios)
        warmup_time *= 60  # minutes to seconds
        data.drop(index=data[data['start_time'] < warmup_time].index,
                  inplace=True)

        grouped = data.groupby('control percentages')
        # for control_percentage in data['control percentages'].unique():
        for control_percentage, data_to_plot in grouped:
            # data_to_plot = data[data['control percentages']
            #                     == control_percentage]
            plt.rc('font', size=15)
            for veh_name in ['lo', 'ld', 'fd']:
                sns.histplot(data=data_to_plot, x='initial_risk_to_' + veh_name,
                             stat='percent', hue='mandatory',
                             palette='tab10')
                plt.tight_layout()
                if self.should_save_fig:
                    fig = plt.gcf()
                    fig.set_dpi(200)
                    vehicles_per_lane_str = '_'.join(
                        str(v) for v
                        in data_to_plot['vehicles_per_lane'].unique()
                    ) + 'vehs_per_lane'
                    fig_name = '_'.join(
                        ['initial_lane_change_risks', vehicles_per_lane_str,
                         str(control_percentage).replace('% ', '_')])
                    fig.savefig(os.path.join(self._figure_folder, fig_name))
                plt.show()

    def plot_heatmap_risk_vs_control(
            self, y: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10, normalize: bool = False
    ):
        """

        :param y:
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time: in minutes
        :param normalize:
        :return:
        """
        y_plotting_options = {
            'lane_change_count': {'title': '# lane changes',
                                  'aggregation_function': np.count_nonzero,
                                  'col_in_df': 'veh_id'},
            'vehicle_count': {'title': 'Output flow',
                              'aggregation_function': np.sum,
                              'col_in_df': y},
            'total_lane_change_risk': {'title': 'Total risk',
                                       'aggregation_function': np.mean,
                                       'col_in_df': y},
            'initial_risk': {'title': 'Initial risk',
                             'aggregation_function': np.mean,
                             'col_in_df': y}
        }

        col_in_df = y_plotting_options[y]['col_in_df']
        aggregation_function = y_plotting_options[y]['aggregation_function']
        title = y_plotting_options[y]['title']

        plt.rc('font', size=17)
        data = self._load_data(y, scenarios)
        self._prepare_data_for_plotting(y, data, warmup_time,
                                        sensor_name=['out'])
        data.loc[data['control percentages'] == '100% HDV',
                 'control percentages'] = '100% HD'
        # for vpl in vehicles_per_lane:
        for vpl, data_to_plot in data.groupby('vehicles_per_lane'):
            n_simulations = data_to_plot['simulation_number'].nunique()
            table = data_to_plot.pivot_table(values=col_in_df,
                                             index=['accepted_risk'],
                                             columns=['control percentages'],
                                             aggfunc=aggregation_function)
            if 'count' in y:
                table /= n_simulations
            no_avs_no_risk_value = table.loc[0, '100% HD']
            print("Base value (humans drivers): ", no_avs_no_risk_value)
            # Necessary because simulations without AVs only have LC values
            # with zero accepted risk.
            table.fillna(value=no_avs_no_risk_value, inplace=True)
            if normalize:
                table /= no_avs_no_risk_value
            max_table_value = np.round(np.nanmax(table.to_numpy()))
            fmt = '.2f' if max_table_value < 100 else '.0f'
            sns.heatmap(table.sort_index(axis=0, ascending=False),
                        annot=True, fmt=fmt,
                        xticklabels=table.columns.get_level_values(
                            'control percentages'))
            plt.xlabel('', fontsize=22)
            plt.ylabel('accepted initial risk', fontsize=22)
            plt.title(title + ' at ' + str(vpl) + ' vehs/h/lane',
                      fontsize=22)
            plt.tight_layout()
            if self.should_save_fig:
                scenarios_subset = [sc for sc in scenarios
                                    if sc.vehicles_per_lane == vpl]
                self.save_fig(plt.gcf(), 'heatmap', y, scenarios_subset)
            plt.show()

    def plot_risk_heatmap(
            self, risk_type: str, scenarios: List[file_handling.ScenarioInfo],
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

        if risk_type == 'both':
            normalizer = self.plot_risk_heatmap('total_risk', scenarios,
                                                normalizer)
            self.plot_risk_heatmap('total_lane_change_risk', scenarios,
                                   normalizer)
            return normalizer

        data = self._load_data(risk_type, scenarios)
        post_processing.drop_warmup_samples(data, 10)
        agg_function = np.sum
        if normalizer is None:
            normalizer = data.loc[
                (data['control percentages'] == '100% HDV')
                & (data['vehicles per hour'] == data[
                    'vehicles per hour'].min()),
                risk_type].sum()
        title = ' '.join(risk_type.split('_')[1:])
        fig = _plot_heatmap(data, risk_type, 'vehicles per hour',
                            'control percentages', normalizer, title,
                            agg_function=agg_function)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', risk_type, scenarios)
            self.save_fig(fig, fig_name=fig_name)
        return normalizer

    def plot_fd_discomfort(
            self, scenarios: List[file_handling.ScenarioInfo],
            brake_threshold: int = 4):
        y = 'fd_discomfort'
        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, 10)
        col_name = '_'.join([y, str(brake_threshold)])
        agg_function = np.mean
        normalizer = data.loc[
            (data['control percentages'] == '100% HDV')
            & (data['vehicles per hour'] == data[
                'vehicles per hour'].min()),
            col_name].agg(agg_function)
        # normalizer = 1
        fig = _plot_heatmap(data, col_name, 'vehicles per hour',
                            'control percentages', normalizer,
                            'Dest Lane Follower Discomfort',
                            agg_function=agg_function,
                            custom_colorbar_range=True)
        if self.should_save_fig:
            fig_name = self.create_figure_name('heatmap', col_name, scenarios)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(data, col_name, scenarios, agg_function,
                                show_variation=True)

    def plot_discomfort_heatmap(
            self, scenarios: List[file_handling.ScenarioInfo],
            max_brake: int = 4):
        data = self._load_data('discomfort', scenarios)
        y = 'discomfort_' + str(max_brake)
        post_processing.drop_warmup_samples(data, 10)
        normalizer = data.loc[
            (data['control percentages'] == '100% HDV')
            & (data['vehicles_per_lane'] == data['vehicles_per_lane'].min()),
            y].sum()
        fig = _plot_heatmap(data, y, 'vehicles_per_lane',
                            'control percentages', normalizer)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_total_output_heatmap(
            self, scenarios: List[file_handling.ScenarioInfo]):
        y = 'vehicle_count'
        data = self._load_data('vehicle_count', scenarios)
        data = self._prepare_data_collection_data(data, sensor_identifier='out',
                                                  warmup_time=10)
        normalizer = data.loc[
            (data['control percentages'] == '100% HDV')
            & (data['vehicles_per_lane'] == data[
                'vehicles_per_lane'].min()),
            y].sum()
        fig = _plot_heatmap(data, 'vehicle_count', 'vehicles_per_lane',
                            'control percentages', normalizer, 'Output flow')
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', 'vehicle_count', scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_emission_heatmap(
            self, scenarios: List[file_handling.ScenarioInfo],
            pollutant_id: int = 91
    ):
        y = 'emission_per_volume'
        data = self._load_data(y, scenarios)
        ResultAnalyzer._prepare_pollutant_data(data, pollutant_id)
        title = 'Normalized ' + self._pollutant_id_to_string[pollutant_id]
        normalizer = data.loc[
            (data['control percentages'] == '100% HDV')
            & (data['vehicles per hour']
               == data['vehicles per hour'].min()),
            y].sum()

        fig = _plot_heatmap(data, y, 'vehicles per hour', 'control percentages',
                            normalizer, title)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', self._pollutant_id_to_string[pollutant_id],
                scenarios)
            self.save_fig(fig, fig_name=fig_name)
        _produce_console_output(data, y, scenarios,
                                np.sum, show_variation=True)

    def plot_lane_change_count_heatmap(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10):
        y = 'lane_change_count'
        agg_function = np.count_nonzero
        col_to_count = 'veh_id'

        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        n_simulations = data['simulation_number'].nunique()
        _plot_heatmap(data, col_to_count, 'vehicles per hour',
                      'control percentages', normalizer=n_simulations,
                      title='lane change count',
                      agg_function=agg_function)
        _produce_console_output(data, col_to_count, scenarios, agg_function)

    def print_summary_of_issues(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10):
        """
        Prints out a summary with average number of times the AV requested
        human intervention and average number of vehicles removed from
        simulation
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :return:
        """
        data = self._load_data('lane_change_issues',
                               scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        n_simulations = data['simulation_number'].nunique()
        if data['accepted_risk'].nunique() <= 1:
            issue_count = data.groupby(
                ['vehicles_per_lane', 'control percentages',
                 'issue'])['veh_id'].count()
        else:
            issue_count = data.groupby(
                ['vehicles_per_lane', 'control percentages',
                 'accepted_risk', 'issue'])['veh_id'].count()
        issue_count /= n_simulations
        print(issue_count)
        print('NOTE: the vissim intervention count from the result above is '
              'not reliable')
        data = self._load_data('lane_change_count',
                               scenarios)

        print(data.groupby(['vehicles_per_lane', 'control percentages'])[
                  'vissim_in_control'].mean())

    def plot_all_pollutant_heatmaps(
            self, scenarios: List[file_handling.ScenarioInfo]):

        for p_id in self._pollutant_id_to_string:
            self.plot_emission_heatmap(scenarios, p_id)

    def count_lane_changes_from_vehicle_record(
            self, scenario_info: file_handling.ScenarioInfo
    ):
        """
        Counts the lane changes over all simulations under a determined
        simulation configuration
        """
        print(scenario_info.vehicle_percentages)
        warmup_time = 10
        # lc_reader = readers.VissimLaneChangeReader(self.scenario_name)
        # lc_data = lc_reader.load_data_with_controlled_percentage(
        #     [vehicle_percentages], vehicles_per_lane, accepted_risks)
        # # print(lc_data.groupby(['simulation_number'])['veh_id'].count())
        # lc_data.drop(index=lc_data[lc_data['time'] < warmup_time].index,
        #              inplace=True)
        # print('LCs from LC file: ', lc_data.shape[0])

        links = [3, 10002]
        veh_reader = readers.VehicleRecordReader(self.scenario_name)
        lc_counter = []
        data_generator = veh_reader.generate_all_data_from_scenario(
            scenario_info)
        for (data, _) in data_generator:
            data.drop(index=data[~data['link'].isin(links)].index,
                      inplace=True)
            post_processing.drop_warmup_samples(data, warmup_time)
            data.sort_values('veh_id', kind='stable', inplace=True)
            data['is_lane_changing'] = data['lane_change'] != 'None'
            data['lc_transition'] = data['is_lane_changing'].diff()
            lc_counter.append(np.count_nonzero(data['lc_transition']) / 2)
        print('LCs from veh records: ', sum(lc_counter))

    # Platoon LC plots ======================================================= #

    def plot_fundamental_diagram_per_strategy(
            self, scenarios: List[file_handling.ScenarioInfo],
            use_upstream_link: bool, lanes: str,
            link_segment_number: int = None,
            aggregation_period: int = 30, warmup_time: int = 10):
        """
        Uses volume and density from link evaluation.
        """
        if lanes == 'orig':
            lane_numbers = [1]
        elif lanes == 'dest':
            lane_numbers = [2]
        elif lanes == 'both':
            lane_numbers = [1, 2]
        else:
            raise ValueError("Parameter lanes must be 'orig', 'dest', "
                             "or 'both'.")

        main_links = self.file_handler.get_main_links()
        link = main_links[0] if use_upstream_link else main_links[1]
        axes = self.plot_fundamental_diagram(
            scenarios, link, link_segment_number=link_segment_number,
            lanes=lane_numbers, hue='lane', col='Strategy',
            aggregation_period=aggregation_period, warmup_time=warmup_time)
        # title = (lanes + ' lane' + ('s' if lanes == 'both' else '')
        #          + '; dest lane speed ' + orig_and_dest_lane_speeds[1])
        # ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_y_vs_vehicle_input(
            self, y: str, scenarios: List[file_handling.ScenarioInfo]):
        """
        Line plot with vehicle input on the x axis and LC strategies as hue

        :param y: Options: was_lane_change_completed, maneuver_time,
        travel_time, accel_cost, stayed_in_platoon
        :param scenarios: List of simulation parameters for several scenarios
        """
        self.plot_results_for_platoon_scenario(
            y, 'Main Road Input (vehs/h)', 'Strategy', scenarios,
            is_bar_plot=True)

    def plot_y_vs_platoon_lc_strategy(
            self, y: str, scenarios: List[file_handling.ScenarioInfo]):
        """
        Line plot with strategies on the x axis and vehicle input as hue

        :param y: Options: was_lane_change_completed, maneuver_time,
        travel_time, accel_cost, stayed_in_platoon
        :param scenarios: List of simulation parameters for several scenarios
        """
        self.plot_results_for_platoon_scenario(
            y, 'Strategy', 'Main Road Input (vehs/h)', scenarios
        )

    def plot_results_for_platoon_scenario(
            self, y: str, x: str, hue: str,
            scenarios: List[file_handling.ScenarioInfo],
            is_bar_plot: bool = False):
        """

        """
        if 'platoon' not in self.scenario_name:
            raise ValueError('Must be scenario with platoon lane changes')

        data = self._load_data(y, scenarios)

        # TODO: Drop samples that didn't finish simulation
        data.drop(index=data.loc[~data['traversed_network']].index,
                  inplace=True)
        # TODO: if plotting was_lc_completed, don't remove
        # TODO: if plotting platoon_maneuver_time, must remove cases where
        #  platoons split

        # Presentation naming
        y_name_map = {
            'was_lane_change_completed': '% Successful Lane Changes',
            'vehicle_maneuver_time': 'Maneuver Time per Vehicle (s)',
            'platoon_maneuver_time': 'Platoon Maneuver Time (s)',
            'travel_time': 'Travel Time (s)',
            'accel_cost': 'Accel Cost (m2/s3)',
            'stayed_in_platoon': '% Stayed in Platoon'
        }

        data['Strategy'] = data['lane_change_strategy'].map(
            self._strategy_name_map)
        # y = y.replace('_', ' ').title()
        # col_names_map = {name: name.replace('_', ' ').title() for name in
        #                  data.columns}
        # data.rename(col_names_map, axis=1, inplace=True)
        data['Main Road Input (vehs/h)'] = data['vehicles_per_lane']

        sns.set_style('whitegrid')
        sns.set(font_scale=1)
        plot_function = sns.barplot if is_bar_plot else sns.pointplot
        plot_function(data, x=x, y=y,
                      hue=hue, errorbar=('se', 2),
                      palette='tab10'
                      )
        plt.ylabel(y_name_map[y])
        plt.tight_layout()
        plt.show()

    def plot_results_for_platoon_scenario_comparing_speeds(
            self, y: str, scenarios: List[file_handling.ScenarioInfo]):
        """

        """
        if 'platoon' not in self.scenario_name:
            raise ValueError('Must be scenario with platoon lane changes')

        data = self._load_data(y, scenarios)

        # Presentation naming
        y_name_map = {
            'was_lane_change_completed': '% Successful Lane Changes',
            'vehicle_maneuver_time': 'Maneuver Time per Vehicle (s)',
            'platoon_maneuver_time': 'Platoon Maneuver Time (s)',
            'travel_time': 'Travel Time (s)',
            'accel_costs': 'Accel Cost (m2/s3)',
            'stayed_in_platoon': '% Stayed in Platoon'
        }
        data['Strategy'] = data['lane_change_strategy'].apply(
            lambda q: self._strategy_name_map[q])
        # y = y.replace('_', ' ').title()
        # col_names_map = {name: name.replace('_', ' ').title() for name in
        #                  data.columns}
        # data.rename(col_names_map, axis=1, inplace=True)
        data['Main Road Input (vehs/h)'] = data['vehicles_per_lane']

        sns.set_style('whitegrid')
        sns.set(font_scale=1)
        sns.pointplot(data, x='Strategy', y=y,
                      hue='dest_lane_speed', errorbar=('se', 2),
                      palette='tab10'
                      )
        plt.ylabel(y_name_map[y])
        plt.tight_layout()
        plt.show()

    def plot_flow_box_plot_vs_strategy(
            self, scenarios: List[file_handling.ScenarioInfo],
            flow_sensors: List[int], warmup_time: int = 5,
            aggregation_period: int = 30):
        """

        """
        y = 'flow'
        data = self._load_data(y, scenarios)
        data['Strategy'] = data['lane_change_strategy'].map(
            self._strategy_name_map)
        data['# CAVs (veh/h)'] = data['vehicles_per_lane'] * 2
        relevant_data = self._prepare_data_collection_data(
            data, flow_sensors, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # ResultAnalyzer._check_if_data_is_uniform(relevant_data)
        fig, ax = _my_boxplot(relevant_data, 'Strategy',
                              y, hue='vehicles_per_lane', will_show=False)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                  title='Vehicles per lane', ncols=1)
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name('box_plot', y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_volume_box_plot_vs_strategy(
            self, scenarios: List[file_handling.ScenarioInfo],
            segment: int, lanes: List[int], warmup_time: int = 5,
            aggregation_period: int = 30):
        """

        """
        y = 'volume'
        data = self._load_data(y, scenarios)
        data['Strategy'] = data['lane_change_strategy'].map(
            self._strategy_name_map)
        data['# CAVs (veh/h)'] = data['vehicles_per_lane'] * 2
        main_link = self.file_handler.get_main_links()[0]
        relevant_data = self._prepare_link_evaluation_data(
            data, main_link, segment, lanes, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        # ResultAnalyzer._check_if_data_is_uniform(relevant_data)
        sns.set_style('whitegrid')
        # ax = sns.pointplot(relevant_data, x='Strategy', y=y,
        #               hue='vehicles_per_lane', errorbar=('se', 2),
        #               palette='tab10'
        #               )
        # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
        #           title='Vehicles per lane', ncols=1)
        # plt.tight_layout()
        # plt.show()
        fig, ax = _my_boxplot(relevant_data, 'Strategy',
                              y, hue='# CAVs (veh/h)',
                              will_show=False)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                  ncols=1)
        ax.set_ylabel('flow (veh/h)')
        plt.tight_layout()
        plt.show()
        if self.should_save_fig:
            fig_name = self.create_figure_name('box_plot', y, scenarios)
            self.save_fig(fig, fig_name=fig_name)

    def plot_flows_vs_time_per_strategy(
            self, scenarios: List[file_handling.ScenarioInfo],
            before_or_after_lc_point: str, lanes: List[int],
            warmup_time: int = 5):
        data = self._load_data('vehicle_count', scenarios)
        data['Strategy'] = data[
            'lane_change_strategy'].map(self._strategy_name_map)
        if before_or_after_lc_point == 'before':
            in_flow_sensors = [1, 2]
            out_flow_sensors = [3, 4]
        elif before_or_after_lc_point == 'after':
            in_flow_sensors = [3, 4]
            out_flow_sensors = [5, 6]
        else:
            raise ValueError("Parameter 'before_or_after_lc_point' must be "
                             "either 'before' or 'after'")
        in_flow_sensors = in_flow_sensors[lanes[0] - 1:lanes[-1]]
        out_flow_sensors = out_flow_sensors[lanes[0] - 1:lanes[-1]]

        aggregation_period = 30
        data_in = self._prepare_data_collection_data(
            data, in_flow_sensors, warmup_time=warmup_time,
            aggregate_sensors=True, aggregation_period=aggregation_period)
        data_in['sensor position'] = 'in'
        data_out = self._prepare_data_collection_data(
            data, out_flow_sensors, warmup_time=warmup_time,
            aggregate_sensors=True, aggregation_period=aggregation_period)
        data_out['sensor position'] = 'out'

        aggregated_data = pd.concat([data_in, data_out])
        # aggregated_data['sensor position'] = aggregated_data[
        #     'sensor_number'].map(lambda x: 'origin' if x % 2 == 1 else 'dest')

        # if True:
        #     lc_data = self._load_data('platoon_maneuver_time', scenarios)
        #     aggregated_data = self.normalize_lane_change_times(
        #         aggregated_data, lc_data)

        plt.rc('font', size=17)
        sns.relplot(aggregated_data, x='time', y='vehicle_count',
                    kind='line', hue='sensor position', row='Strategy',
                    aspect=4)
        plt.tight_layout()
        plt.show()

    def plot_link_data_vs_time_per_strategy(
            self, y: str, scenarios: List[file_handling.ScenarioInfo],
            use_upstream_link: bool, lanes: str,
            link_segment_number: int = None,
            aggregation_period: int = 30, warmup_time: int = 10):
        if lanes == 'orig':
            lane_numbers = [1]
        elif lanes == 'dest':
            lane_numbers = [2]
        elif lanes == 'both':
            lane_numbers = [1, 2]
        else:
            raise ValueError("Parameter lanes must be 'orig', 'dest', "
                             "or 'both'.")
        main_links = self.file_handler.get_main_links()
        link = main_links[0] if use_upstream_link else main_links[1]
        data = self._load_data(y,  scenarios)
        aggregated_data = self._prepare_link_evaluation_data(
            data, link, link_segment_number, lane_numbers,
            aggregate_lanes=False, warmup_time=warmup_time,
            aggregation_period=aggregation_period)
        aggregated_data['lane'] = aggregated_data['lane'].astype(int)
        plt.rc('font', size=17)
        sns.relplot(aggregated_data, x='time', y=y,
                    kind='line', hue='lane', row='Strategy',
                    aspect=4, palette='tab10')
        plt.tight_layout()
        plt.show()

    def normalize_lane_change_times(self, output_data: pd.DataFrame,
                                    lane_change_data: pd.DataFrame):
        """
        :param output_data: Link evaluation or data collection measurements
        :param lane_change_data: Detailed lane change data for the scenarios
        """
        max_duration = lane_change_data['platoon_maneuver_time'].max()
        grouped_lc_data = lane_change_data.groupby(
                ['lane_change_strategy', 'simulation_number',
                 'initial_platoon_id'])
        grouped_data = output_data.groupby(
            ['lane_change_strategy', 'simulation_number'])

        t1, t2 = output_data['time_interval'].iloc[0].split('-')
        interval = int(t2) - int(t1)
        normalized_data_list = []
        margin = 2
        for group_id, single_lc_data in grouped_lc_data:
            lc_t0 = single_lc_data['maneuver_start_time'].iloc[0]
            lc_tf = lc_t0 + max_duration
            t0 = (lc_t0 // interval - margin) * interval
            tf = (lc_tf // interval + margin) * interval
            single_simulation_data = grouped_data.get_group(group_id[0:2])
            data_during_lc = single_simulation_data[
                (single_simulation_data['time'] >= t0)
                & (single_simulation_data['time'] < tf)].copy()
            data_during_lc['time'] = data_during_lc['time'] - t0
            normalized_data_list.append(data_during_lc)
        return pd.concat(normalized_data_list)

    # Traffic Light Scenario Plots =========================================== #

    def plot_violations_per_control_percentage(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10):
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
        data = self._load_data('violations', scenarios)

        # TODO: temporary
        data.drop(index=data[data['simulation_number'] == 11].index,
                  inplace=True)

        post_processing.drop_warmup_samples(data, warmup_time)
        results = data.groupby(['control percentages', 'vehicles_per_lane'],
                               as_index=False)['veh_id'].count().rename(
            columns={'veh_id': 'violations count'})
        results['mean violations'] = results['violations count'] / n_simulations
        print(results)

    def plot_heatmap_for_traffic_light_scenario(
            self, y: str, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10):
        """
        Plots a heatmap
        :param y:
        :param scenarios: List of simulation parameters for several scenarios
        :param warmup_time:
        :return:
        """

        title_map = {'vehicle_count': 'Output Flow',
                     'discomfort': 'Discomfort',
                     'barrier_function_risk': 'Risk'}
        n_simulations = 10
        plt.rc('font', size=17)
        data = self._load_data(y, scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        subsets_by_vpl = file_handling.split_scenario_by(
            scenarios, 'vehicles_per_lane')
        for vpl, data_to_plot in data.groupby('vehicles_per_lane'):
            table = pd.pivot_table(data_to_plot, values=y,
                                   index=['traffic_light_cacc_percentage'],
                                   columns=['traffic_light_acc_percentage'],
                                   aggfunc=np.sum)
            if y == 'vehicle_count':
                table_norm = (table / n_simulations
                              * 3 / 2)  # 60min / 20min / 2 lanes
                value_format = '.0f'
            else:
                table_norm = table / table[0].iloc[0]
                value_format = '.2g'
            max_table_value = np.round(np.nanmax(table_norm.to_numpy()))
            sns.heatmap(table_norm.sort_index(axis=0, ascending=False),
                        annot=True, fmt=value_format,
                        vmin=0, vmax=max(1, int(max_table_value)))
            plt.xlabel('% of CAVs without V2V', fontsize=22)
            plt.ylabel('% of CAVs with V2V', fontsize=22)
            plt.title(title_map[y] + ' at ' + str(vpl) + ' vehs/h/lane',
                      fontsize=22)
            if self.should_save_fig:
                fig = plt.gcf()
                self.save_fig(fig, 'heatmap', y, subsets_by_vpl[vpl])
            plt.tight_layout()
            plt.show()

    def plot_violations_heatmap(
            self, scenarios: List[file_handling.ScenarioInfo],
            warmup_time: int = 10):
        n_simulations = 10
        plt.rc('font', size=17)
        data = self._load_data('violations', scenarios)
        post_processing.drop_warmup_samples(data, warmup_time)
        subsets_by_vpl = file_handling.split_scenario_by(
            scenarios, 'vehicles_per_lane')
        for vpl, data_to_plot in data.groupby(['vehicles_per_lane']):
            table = pd.pivot_table(
                data_to_plot[['veh_id', 'traffic_light_acc_percentage',
                             'traffic_light_cacc_percentage']],
                values='veh_id',
                index=['traffic_light_cacc_percentage'],
                columns=['traffic_light_acc_percentage'],
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
            plt.xlabel('% of CAVs without V2V', fontsize=22)
            plt.ylabel('% of CAVs with V2V', fontsize=22)
            plt.title('Violations at ' + str(vpl) + ' vehs/h/lane',
                      fontsize=22)
            if self.should_save_fig:
                fig = plt.gcf()
                self.save_fig(fig, 'heatmap', 'violations', subsets_by_vpl[vpl])
            plt.show()

    # Others ================================================================= #

    def accel_vs_time_for_different_vehicle_pairs(self):
        """
        Plots acceleration vs time for human following CAV and CAV following
        human. This requires very specific knowledge of the simulation being
        loaded. Currently this function plots a result for the traffic_lights
        scenario with 25% CACC-equipped vehicles.
        :return:
        """
        # scenario_name = 'traffic_lights'
        # vehicle_percentage = {VehicleType.TRAFFIC_LIGHT_CACC: 25}
        # vehicles_per_lane = 500
        # all_cases = [
        #     {'follower': 'human', 'leader': 'human', 'id': 202},
        #     {'follower': 'human', 'leader': 'CAV', 'id': 203},
        #     {'follower': 'CAV', 'leader': 'human', 'id': 201},
        #     # {'follower': 'CAV', 'leader': 'CAV', 'id': 152},
        #     {'follower': 'CAV', 'leader': 'CAV', 'id': 196},
        #     # {'follower': 'CAV', 'leader': 'CAV', 'id': 208},
        #     # {'follower': 'CAV', 'leader': 'CAV', 'id': 234},
        #     # {'follower': 'CAV', 'leader': 'CAV', 'id': 239}
        # ]
        # time = [910, 950]
        scenario_name = 'in_and_out_safe'
        vehicle_percentage = {VehicleType.CONNECTED: 50}
        vehicles_per_lane = 2000
        all_cases = [
            {'follower': 'human', 'leader': 'human',
             'id': 35, 'time': [30, 40]},
            {'follower': 'human', 'leader': 'CAV',
             'id': 33, 'time': [30, 40]},
            {'follower': 'CAV', 'leader': 'human',
             'id': 53, 'time': [37, 53]},
            {'follower': 'CAV', 'leader': 'CAV',
             'id': 17, 'time': [22, 31]},
        ]
        reader = readers.VehicleRecordReader(scenario_name)
        scenario_info = file_handling.ScenarioInfo(vehicle_percentage,
                                                   vehicles_per_lane,
                                                   accepted_risk=0)
        veh_record = reader.load_single_file_from_scenario(1, scenario_info)
        sns.set_style('whitegrid')
        # fig, axes = plt.subplots(len(all_cases), 1)
        # fig.set_size_inches(12, 16)
        plt.rc('font', size=25)
        # full_data = []
        for case in all_cases:
            # case = all_cases[i]
            veh_id = case['id']
            time = case['time']
            veh_record_slice = veh_record[(veh_record['time'] > time[0])
                                          & (veh_record['time'] < time[1])]
            follower_data = veh_record_slice.loc[veh_record_slice['veh_id']
                                                 == veh_id]
            leader_id = follower_data.iloc[0]['leader_id']
            leader_data = veh_record_slice.loc[veh_record_slice['veh_id']
                                               == leader_id]
            data = pd.concat([follower_data, leader_data])
            data['name'] = ''
            data.loc[data['veh_id'] == veh_id, 'name'] = (case['follower']
                                                          + ' follower')
            data.loc[data['veh_id'] == leader_id, 'name'] = (case['leader']
                                                             + ' leader')
            # data['gap'] = 0
            # follower_gap = post_processing.compute_gap_between_vehicles(
            #     leader_data['rear_x'].to_numpy(),
            #     leader_data['rear_y'].to_numpy(),
            #     follower_data['front_x'].to_numpy(),
            #     follower_data['front_y'].to_numpy())
            # data.loc[data['veh_id'] == veh_id, 'gap'] = follower_gap
            # full_data.append(data)
            fig, axes = plt.subplots(1, 1)
            sns.lineplot(data=data,
                         x='time', y='ax', hue='name',
                         palette='tab10', linewidth=4, ax=axes,
                         legend=False
                         )
            # axes.legend(fontsize=35, loc='upper center',
            #             bbox_to_anchor=(0.45, 1.2), ncol=2,
            #             frameon=False)
            axes.set_xlabel('t [s]')
            axes.set_ylabel('a(t) [m/s^2]')
            # sns.lineplot(data=data.loc[data['veh_id'] == veh_id],
            #              x='time', y='gap',
            #              palette='tab10', linewidth=3, ax=axes[1])
            # axes[1].set_xlabel('t [s]', fontsize=24)
            # axes[1].set_ylabel('gap [m]', fontsize=24)
            # fig.set_size_inches(12, 8)
            plt.tight_layout()
            if self.should_save_fig:
                fig = plt.gcf()
                fig.set_dpi(300)
                fig_name = '_'.join(['accel_vs_time', scenario_name,
                                     case['follower'], 'follower',
                                     case['leader'], 'leader'])
                fig.savefig(os.path.join(self._figure_folder, fig_name
                                         + '_presentation'))
            plt.show()

    def risk_vs_time_example(self):
        scenario_name = 'in_and_out_safe'
        vehicle_percentage = {VehicleType.CONNECTED: 0}
        vehicles_per_lane = 1000
        reader = readers.VehicleRecordReader(scenario_name)
        scenario_info = file_handling.ScenarioInfo(vehicle_percentage,
                                                   vehicles_per_lane,
                                                   accepted_risk=0)
        veh_record = reader.load_single_file_from_scenario(1, scenario_info)
        veh_id = 675
        single_veh = veh_record[veh_record['veh_id'] == veh_id]
        min_t = single_veh['time'].iloc[0]
        max_t = single_veh['time'].iloc[-1]
        relevant_data = veh_record[(veh_record['time'] >= min_t)
                                   & (veh_record['time'] <= max_t)].copy()
        pp = post_processing.SSMProcessor(scenario_name)
        pp.post_process(relevant_data)
        single_veh = relevant_data[relevant_data['veh_id'] == veh_id]

        sns.set_style('whitegrid')
        plt.rc('font', size=40)
        sns.lineplot(data=single_veh, x='time', y='risk', linewidth=4)
        plt.xlabel('time (s)')
        plt.ylabel('risk (m/s)')
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        plt.tight_layout()
        if self.should_save_fig:
            fig_name = 'risk_vs_time_example'
            self.save_fig(fig, fig_name=fig_name)
        plt.show()
        sampling = single_veh['time'].diff().iloc[1]
        print('total risk:', single_veh['risk'].sum() * sampling)

    def speed_color_map(self, vehicles_per_lane: int,
                        vehicle_percentages: List[Dict[VehicleType, int]]):
        """

        :param vehicles_per_lane:
       :param vehicle_percentages: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :return:
        """
        raise NotImplementedError
        # if isinstance(controlled_percentage, list):
        #     percentages = np.sort(controlled_percentage[:])
        # else:
        #     percentages = np.sort([controlled_percentage])

        # for vehicle_type in self._vehicle_types:
        #     for p in percentages:
        #         veh_record_reader = readers.VehicleRecordReader(
        #             self.network_name)
        #         min_file, max_file =
        #             veh_record_reader.find_min_max_file_number(
        #             vehicle_type, p, vehicles_per_lane)
        #         # For now just one file. We'll see later if aggregating
        #         # makes sense
        #         veh_record = veh_record_reader.load_data(max_file,
        #                                                  vehicle_type, p,
        #                                                  vehicles_per_lane)
        #         # Get only main segment.
        #         # This must change for other scenarios
        #         veh_record.drop(
        #             index=veh_record[(veh_record['link'] != 3)
        #                              | (veh_record['time'] < 300)].index,
        #             inplace=True)
        #         veh_record['time [s]'] = veh_record['time'] // 10
        #         space_bins = [i for i in
        #                       range(0, int(veh_record['x'].max()), 25)]
        #         veh_record['x [m]'] = pd.cut(veh_record['x'], bins=space_bins,
        #                                      labels=space_bins[:-1])
        #         plotted_data = veh_record.groupby(['time [s]', 'x [m]'],
        #                                           as_index=False)['vx'].mean()
        #         plotted_data = plotted_data.pivot('time [s]', 'x [m]', 'vx')
        #         ax = sns.heatmap(plotted_data)
        #         ax.invert_yaxis()
        #         plt.show()
        #     # We only plot 0 percentage once
        #     if 0 in percentages:
        #         percentages = np.delete(percentages, 0)

    # Multiple plots  ======================================================== #

    def get_flow_and_risk_plots(self,
                                scenarios: List[file_handling.ScenarioInfo]):
        """Generates the plots used in the Safe Lane Changes paper."""

        self.plot_flow_box_plot_vs_controlled_percentage(scenarios,
                                                         warmup_time=10)
        self.plot_risk_histograms('total_risk', scenarios, min_risk=1)
        self.plot_risk_histograms('total_lane_change_risk', scenarios,
                                  min_risk=1)

    # Support methods ======================================================== #
    def _load_data(
            self, y: str,
            scenarios: List[file_handling.ScenarioInfo]) -> pd.DataFrame:
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
        data['vehicles per hour'] = (
                self.file_handler.get_n_lanes()
                * data['vehicles_per_lane'])
        return data

    def _prepare_data_collection_data(
            self, data: pd.DataFrame,
            sensor_identifier: Union[List[int], str] = None,
            aggregate_sensors: bool = True,
            warmup_time: int = 10,
            aggregation_period: int = 30) -> pd.DataFrame:
        """
        Keeps only data from given sensors, discards data before warmup time,
        and computes flow for the given aggregation period.
        """
        # Drop early samples
        post_processing.drop_warmup_samples(data, warmup_time)
        # Select fata from the sensors
        if self.scenario_name.startswith('in_and_out'):
            selected_sensor_data = (
                _select_flow_sensors_from_in_and_out_scenario(
                    data, sensor_identifier)
            )
            aggregate_sensors = False
        else:
            selected_sensor_data = data.drop(data[~data['sensor_number'].isin(
                sensor_identifier)].index)
        # Aggregate sensors
        if aggregate_sensors:
            selected_sensor_data.reset_index(drop=True, inplace=True)
            n_sensors = len(sensor_identifier)
            aggregated_data = selected_sensor_data.iloc[
                [i for i in range(0, selected_sensor_data.shape[0],
                                  n_sensors)]].copy()
            aggregated_data['vehicle_count'] = selected_sensor_data[
                'vehicle_count'].groupby(
                selected_sensor_data.index // n_sensors).sum().to_numpy()
            aggregated_data = _aggregate_data_over_time(aggregated_data, 'vehicle_count',
                                                        aggregation_period, np.sum)
        else:
            temp = []
            data_per_sensor = selected_sensor_data.groupby('sensor_number')
            for _, group in data_per_sensor:
                temp.append(_aggregate_data_over_time(group, 'vehicle_count',
                                                      aggregation_period, np.sum))
            aggregated_data = pd.concat(temp)
        # Aggregate time
        # aggregated_data.sort_values(['sensor_number', 'time'], kind='stable',
        #                             inplace=True)

        aggregated_data['flow'] = (3600 / aggregation_period
                                   * aggregated_data['vehicle_count'])
        return aggregated_data

    def _prepare_link_evaluation_data(
            self, data: pd.DataFrame, link: int, segment: int = None,
            lanes: List[int] = None, sensor_name: str = None,
            aggregate_lanes: bool = True, warmup_time: int = 10,
            aggregation_period: int = 30) -> pd.DataFrame:

        # Drop early samples
        post_processing.drop_warmup_samples(data, warmup_time)
        # Select link
        data.drop(index=data[data['link_number'] != link].index,
                  inplace=True)
        # Select segment
        if segment is not None:
            data.drop(index=data[data['link_segment'] != segment].index,
                      inplace=True)
        elif data['link_segment'].nunique() > 1:
            print("WARNING: the chosen link has several segments, and we're "
                  "keeping all of them")
        # Select lanes
        if self.scenario_name.startswith('in_and_out'):
            _select_lanes_for_in_and_out_scenario(data, sensor_name)
        elif lanes is not None and _has_per_lane_results(data):
            data.drop(data[~data['lane'].isin(lanes)].index,
                      inplace=True)
        # Aggregate
        if aggregate_lanes:
            data.reset_index(drop=True, inplace=True)
            n_lanes = len(lanes)
            aggregated_data = data.iloc[[i for i in range(0, data.shape[0],
                                                          n_lanes)]].copy()
            aggregated_data[['density', 'volume']] = data[
                ['density', 'volume']].groupby(
                data.index // n_lanes).sum().to_numpy()
            aggregated_data = _aggregate_data_over_time(
                aggregated_data, ['density', 'volume'],
                aggregation_period, np.mean)
        else:
            temp = []
            data_per_lane = data.groupby('lane')
            for _, group in data_per_lane:
                temp.append(_aggregate_data_over_time(
                    group, ['density', 'volume'], aggregation_period, np.mean))
            aggregated_data = pd.concat(temp)

        if 'lane_change_strategy' in aggregated_data.columns:
            aggregated_data['Strategy'] = aggregated_data[
                'lane_change_strategy'].map(self._strategy_name_map)
        return aggregated_data

    def _prepare_data_for_plotting(self, y: str, data: pd.DataFrame,
                                   warmup_time: int = 0,
                                   sensor_name: List[str] = None,
                                   pollutant_id: int = None):
        """
        Performs several operations to make the data proper for plotting:
        1. Fill NaN entries in columns describing controlled vehicle
        percentage
        2. Aggregates data from all columns describing controlled vehicle
        percentage into a single 'control percentages' column
        3. [Optional] Removes samples before warm-up time
        4. [Optional] Filter out certain sensor groups
        :param data: data aggregated over time
        :param warmup_time: Samples earlier than warmup_time are dropped.
         Must be passed in minutes
        :param sensor_name: if plotting flow or density, we can determine
         which sensor/lane is shown
        """
        if y == 'flow':
            self._prepare_data_collection_data(data, sensor_name,
                                               warmup_time=warmup_time)
        elif y == 'density':
            link = self.file_handler.get_main_links()[0]
            self._prepare_link_evaluation_data(
                data, link, sensor_name=sensor_name, warmup_time=warmup_time)
        elif y == 'emission_per_volume':
            ResultAnalyzer._prepare_pollutant_data(data, pollutant_id)
        elif 'risk' in y:
            ResultAnalyzer._prepare_risk_data(data, y, warmup_time, 0)
        else:
            post_processing.drop_warmup_samples(data, warmup_time)

    @staticmethod
    def _prepare_pollutant_data(data: pd.DataFrame,
                                pollutant_id: int):
        data.drop(data[~(data['pollutant_id'] == pollutant_id)].index,
                  inplace=True)

    @staticmethod
    def _prepare_risk_data(data: pd.DataFrame, risky_type: str,
                           warmup_time: int, min_risk: float):
        """
        Removes samples before warmup time and with risk below min_risk
        :param data: Any VISSIM output or post processed data
        :param warmup_time: Must be in minutes
        :param min_risk: Threshold value
        """
        post_processing.drop_warmup_samples(data, warmup_time)
        data.drop(index=data[data[risky_type] < min_risk].index,
                  inplace=True)

    @staticmethod
    def _check_if_data_is_uniform(data: pd.DataFrame):
        """
        Checks whether all loaded scenarios were run with the same random seeds.
        :param data:
        :return:
        """
        grouped = data.groupby('random_seed')
        group_sizes = grouped['simulation_number'].count()
        if any(group_sizes != group_sizes.iloc[0]):
            print('Not all scenarios have the same number of samples.\n'
                  'This might create misleading plots.')

        # TODO: [Feb 17] erase
        # # Get the intersection of random seeds for each control percentage
        # random_seeds = np.array([])
        # for sc in scenarios:
        #     for percent in data['control percentages'].unique():
        #         current_random_seeds = data.loc[
        #             (data['control percentages'] == percent)
        #             & (data['vehicles_per_lane'] == veh_input),
        #             'random_seed'].unique()
        #         if random_seeds.size == 0:
        #             random_seeds = current_random_seeds
        #         else:
        #             random_seeds = np.intersect1d(random_seeds,
        #                                           current_random_seeds)
        #
        # # Keep only the random_seeds used by all control percentages
        # relevant_data = data.drop(index=data[~data[
        #     'random_seed'].isin(random_seeds)].index)
        # if relevant_data.shape[0] != data.shape[0]:
        #     print('Some simulations results were dropped because not all '
        #           'controlled percentages or vehicle inputs had the same '
        #           'amount of simulation results')
        # return relevant_data

    @staticmethod
    def remove_deadlock_simulations(data):
        deadlock_entries = (data.loc[
                                data['flow'] == 0,
                                ['vehicles_per_lane', 'random_seed']
                            ].drop_duplicates())
        for element in deadlock_entries.values:
            idx = data.loc[(data['vehicles_per_lane'] == element[0])
                           & (data['random_seed'] == element[1])].index
            data.drop(idx, inplace=True)
            print('Removed results from simulation with input {}, random '
                  'seed {} due to deadlock'.
                  format(element[0], element[1]))

    def save_fig(self, fig: plt.Figure, plot_type: str = None,
                 measurement_name: str = None,
                 scenarios: List[file_handling.ScenarioInfo] = None,
                 fig_name: str = None
                 ):
        """
        Saves the figure. We must provide either the fig_name or
        enough information for the figure name to be automatically generated
        """
        # Making the figure nice for inclusion in documents
        # self.widen_fig(fig, controlled_percentage)
        # plt.rc('font', size=20)
        if not fig_name:
            # TODO: should be a function
            vehicle_percentage_strings = []
            vehicles_per_lane_strings = set()
            accepted_risk_strings = set()
            for sc in scenarios:
                vehicles_per_lane_strings.add(str(sc.vehicles_per_lane))
                vehicle_percentages = sc.vehicle_percentages
                vp_str = ['_'.join([str(p), vt.name.lower])
                          for vt, p in vehicle_percentages.items()]
                vehicle_percentage_strings.extend(vp_str)
                if sc.accepted_risk is not None:
                    accepted_risk_strings.add(str(sc.accepted_risk))
            all_vehicles_per_lane = ('_'.join(sorted(
                vehicles_per_lane_strings)) + '_vehs_per_lane')
            all_vehicle_percentages = '_'.join(sorted(
                vehicle_percentage_strings))
            fig_name = '_'.join(
                [plot_type, measurement_name, self.scenario_name,
                 all_vehicles_per_lane, all_vehicle_percentages])
            if len(accepted_risk_strings) > 0:
                all_risks = 'risks_' + '_'.join(sorted(accepted_risk_strings))
                fig_name = '_'.join([fig_name, all_risks])

        # TODO [Feb 17, 2023]: check if fig_name is coming out right
        # if not fig_name:
        #     vehicles_types, temp_pctg = list_of_dicts_to_1d_list(
        #         vehicle_percentages)
        #     veh_penetration_strings = []
        #     for i in range(len(vehicles_types)):
        #         veh_penetration_strings.append(
        #             str(temp_pctg[i]) + '_' + vehicles_types[i].name.lower())
        #     fig_name = (
        #             plot_type + '_' + measurement_name + '_'
        #             + self.scenario_name + '_'
        #             + '_'.join(str(v) for v in sorted(vehicles_per_lane))
        #             + '_' + 'vehs_per_lane' + '_'
        #             + '_'.join(sorted(veh_penetration_strings))
        #     )
        #     if accepted_risk:
        #         fig_name += '_risks_' + '_'.join(str(ar) for ar
        #                                          in accepted_risk)
        fig.set_dpi(400)
        # axes = fig.axes
        plt.tight_layout()
        print("Figure name:", fig_name)
        # a = input('Save? [y/n]')
        # if a != 'y':
        #     print('not saving')
        # else:
        #     print('saving')
        fig.savefig(os.path.join(self._figure_folder, fig_name), dpi=400)

    def create_figure_name(
            self, plot_type: str, measurement_name: str,
            scenarios: List[file_handling.ScenarioInfo]) -> str:

        # TODO [Feb 17, 2023]: not tested yet
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
        all_vehicles_per_lane = ('_'.join(sorted(
            vehicles_per_lane_strings)) + '_vehs_per_lane')
        all_percentage_strings = '_'.join(sorted(percentage_strings))
        all_vehicle_type_strings = '_'.join(sorted(vehicle_type_strings))
        fig_name = '_'.join(
            [plot_type, measurement_name, self.scenario_name,
             all_vehicles_per_lane, all_percentage_strings,
             all_vehicle_type_strings])
        if len(accepted_risk_strings) > 1:
            all_risks = 'risks_' + '_'.join(sorted(accepted_risk_strings))
            fig_name = '_'.join([fig_name, all_risks])
        return fig_name

        # vehicle_input_string = [str(vi) for vi in sorted(vehicles_per_lane)]
        # if vehicle_percentages:
        #     percentage_strings = set()
        #     vt_strings = set()
        #     for vp in vehicle_percentages:
        #         for vt, p in vp.items():
        #             percentage_strings.add(str(p))
        #             vt_strings.add(vt.name.lower())
        #     percentage_strings = sorted(list(percentage_strings))
        #     vt_strings = sorted(list(vt_strings))
        #     fig_name = '_'.join([plot_type, measurement_name,
        #                          self.scenario_name]
        #                         + vehicle_input_string + ['vehs_per_lane']
        #                         + percentage_strings + vt_strings)
        # else:
        #     fig_name = '_'.join([plot_type, measurement_name,
        #                          self.scenario_name]
        #                         + vehicle_input_string + ['vehs_per_lane']
        #                         + [control_percentage])
        # # if accepted_risk:
        # #     fig_name += ('_risks_' + '_'.join(
        # #                      str(ar) for ar in accepted_risk))
        # return fig_name

    @staticmethod
    def widen_fig(fig: plt.Figure, n_boxes: int):
        if n_boxes >= 4:
            fig.set_size_inches(6.4 * 2, 4.8)

    # Plots for a single simulation - OUTDATED: might not work =============== #

    # These methods require post-processed data with SSMs already computed #
    def plot_ssm(self, ssm_names, vehicle_record):
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
        aggregated_data = (vehicle_record[['time'].extend(ssm_names)].
                           groupby('time').sum())
        for ssm in ssm_names:
            if ssm in self._ssm_pretty_name_map:
                ssm_plot_name = self._ssm_pretty_name_map[ssm]
            else:
                ssm_plot_name = ssm
            ax.plot(aggregated_data[ssm].index, aggregated_data[ssm][ssm],
                    label=ssm_plot_name + ' (' + self._units_map[ssm] + ')')
        ax.legend()
        plt.show()

    def plot_ssm_moving_average(self, vehicle_record, ssm_name,
                                window=100, save_path=None):
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
        aggregated_data = vehicle_record[['time', ssm_name]].groupby(
            'time').sum()
        # sns.set_theme()
        aggregated_data['Mov. Avg. ' + ssm_name] = aggregated_data[
            ssm_name].rolling(window=window, min_periods=10).mean()
        sns.lineplot(x=aggregated_data.index,
                     y=aggregated_data['Mov. Avg. ' + ssm_name],
                     ax=ax)
        ax.set_xlabel('time (s)', fontsize=label_font_size)
        if ssm_name in self._ssm_pretty_name_map:
            ssm_plot_name = self._ssm_pretty_name_map[ssm_name]
        else:
            ssm_plot_name = ssm_name
        ax.set_ylabel(ssm_plot_name + ' (' + self._units_map[ssm_name] + ')',
                      fontsize=label_font_size)
        plt.show()

        if save_path is not None:
            fig.savefig(save_path + ssm_name)

    @staticmethod
    def plot_risk_counter(ssm_name, vehicle_record_data):
        """Temporary function to compare how the exact and estimated risks
        vary over time"""
        fig, ax = plt.subplots()
        ssm_counter = ssm_name + '_counter'
        vehicle_record_data[ssm_counter] = vehicle_record_data[ssm_name] > 0
        aggregated_data = vehicle_record_data[['time', ssm_counter]].groupby(
            np.floor(vehicle_record_data['time'])).sum()
        ax.plot(aggregated_data.index, aggregated_data[ssm_counter],
                label=ssm_counter)
        ax.legend()
        plt.show()

    # Data integrity checks ================================================== #

    def find_unfinished_simulations(
            self, scenarios: List[file_handling.ScenarioInfo]):
        """
        Checks whether simulations crashed. This is necessary because,
        when doing multiple runs from the COM interface, VISSIM does not
        always indicate that a simulation crashed.

        :param scenarios: List of simulation parameters for several scenarios
        """
        raw_data = self._load_data('flow', scenarios)
        data = self._prepare_data_collection_data(raw_data, 'in',
                                                  warmup_time=0,
                                                  aggregation_period=60)
        grouped = data.groupby(
            ['control percentages', 'vehicles_per_lane', 'accepted_risk',
             'simulation_number'])['flow']
        all_end_times = grouped.last()
        issues = all_end_times[all_end_times == 0]
        if issues.empty:
            print('All simulations seem to have run till the end')
            return
        # Find the last valid simulation time
        issue_time = []
        for i in issues.index:
            print(i)
            s = grouped.get_group(i)
            issue_time.append(data.loc[s.index[s.to_numpy().nonzero()[0][-1]],
                                       'time'])
            print(issue_time[-1])
        print("Min issue time {} at simulation {}".format(
            min(issue_time), issues.index[np.argmin(issue_time)]))


# Some methods used by the class ============================================= #

def list_of_dicts_to_1d_list(dict_list: List[Dict]):
    keys = []
    values = []
    for d in dict_list:
        keys += list(d.keys())
        values += list(d.values())
    return keys, values


def vehicle_percentage_dict_to_string(vp_dict: Dict[VehicleType, int]) -> str:
    if sum(vp_dict.values()) == 0:
        return '100% HDV'
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + '% ' + vehicle_type_to_str_map[veh_type])
    return ' '.join(sorted(ret_str))


def _aggregate_data_over_time(data: pd.DataFrame,
                              aggregated_variable: Union[str, List[str]],
                              aggregation_period: int, aggregation_function):
    data.reset_index(drop=True, inplace=True)
    t1, t2 = data['time_interval'].iloc[0].split('-')
    interval = int(t2) - int(t1)
    if aggregation_period % interval != 0:
        print('Aggregation period should be a multiple of the data '
              'collection interval used in simulation. Keeping the '
              'original data collection interval.')
        aggregation_period = interval
    n = aggregation_period // interval
    aggregated_data = data.iloc[[i for i in range(0, data.shape[0],
                                                  n)]].copy()
    aggregated_data[aggregated_variable] = data[aggregated_variable].groupby(
        data.index // n).agg(aggregation_function).to_numpy()
    return aggregated_data


def _has_per_lane_results(data: pd.DataFrame) -> bool:
    """
    :param data: Data from link evaluation records
    :return: boolean indicating whether the data has results individualized
    per lane
    """
    if 'lane' in data.columns and data['lane'].iloc[0] != 0:
        return True
    return False


def _select_flow_sensors_from_in_and_out_scenario(
        data: pd.DataFrame, sensor_name: str = None):
    """
    Keeps only data from the sensors with the given sensor names.
    """
    if sensor_name is None:
        sensor_name = 'in'
    sensor_name_map = {
        'in': 1, 'out': 2,
        'in_main_left': 3, 'in_main_right': 4, 'in_ramp': 5,
        'out_main_left': 6, 'out_main_right': 7, 'out_ramp': 8
    }
    sensor_number = sensor_name_map[sensor_name]
    return data.drop(data[data['sensor_number'] != sensor_number].index)


def _select_lanes_for_in_and_out_scenario(data: pd.DataFrame,
                                          sensor_name: str = None):
    """
    Chooses which lanes from the link evaluation data to keep based on the
    sensor names.
    """
    if sensor_name is None:
        sensor_name = 'in'
    sensor_lane_map = {
        'in': [1, 2, 3], 'out': [1, 2, 3],
        'in_main_left': [1], 'in_main_right': [2], 'in_ramp': [3],
        'out_main_left': [1], 'out_main_right': [2], 'out_ramp': [3]
    }
    if _has_per_lane_results(data):
        lanes = sensor_lane_map[sensor_name]
        data.drop(data[~data['lane'].isin(lanes)].index,
                  inplace=True)


def _plot_heatmap(data: pd.DataFrame, value: str, rows: str, columns: str,
                  normalizer: int = 1,
                  title: str = None, agg_function=np.sum,
                  custom_colorbar_range=False):
    fig = plt.figure()
    plt.rc('font', size=17)
    if title is None:
        title = ' '.join(value.split('_'))
    table = data.pivot_table(values=value,
                             index=[columns],
                             columns=[rows],
                             aggfunc=agg_function, sort=False)
    if normalizer > 0:
        table /= normalizer
    max_value = np.ceil(table.max().max())
    min_value = np.floor(table.min().min())
    if custom_colorbar_range:
        vmin, vmax = min_value, max_value
    else:
        vmin, vmax = None, None
    # string_format = '.2f' if max_value < 1000 else '.2g'
    string_format = '.2f'
    sns.heatmap(table, annot=True, vmin=vmin, vmax=vmax, fmt=string_format)
    plt.title(title, fontsize=22)
    # plt.xlabel('Control Percentages')
    plt.tight_layout()
    plt.show()
    return fig


def _my_boxplot(relevant_data: pd.DataFrame, x: str, y: str,
                hue: str = 'vehicles per hour', will_show: bool = True):
    # Plot
    fig = plt.figure()
    plt.rc('font', size=22)
    fig.set_size_inches(12, 6)
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=relevant_data, x=x, y=y, hue=hue)
    ax.legend(loc='upper center',  # bbox_to_anchor=(0.5, 1.1),
              title=hue, ncols=2)
    if will_show:
        plt.tight_layout()
        plt.show()
    return fig, ax


def _fit_fundamental_diagram(data: pd.DataFrame, by: str):
    """
    :param data: Link evaluation treated data
    :param by: How to split the data; must be column of data
    """
    grouped = data.groupby(by)
    coefficients = []
    fig, ax = plt.subplots()
    for group_id, group in grouped:
        find_critical_density()
        # ax.plot(x, y, linewidth=2.0, label=group_id)
        ax.scatter(group['density'], group['volume'])
    ax.legend()
    plt.show()


def find_critical_density(data):
    max_density = data['density'].max()
    reg_u = LinearRegression(fit_intercept=False)
    reg_c = LinearRegression()
    best_d = 0
    best_coefficients = (0, 0)
    best_score = np.float('inf')
    for d in np.linspace(max_density/4, 3*max_density/4, 10):
        idx = data['density'] < d
        uncongested = data[idx]
        congested = data[~idx]
        x_u = uncongested['density'].to_numpy().reshape(-1, 1)
        y_u = uncongested['volume']
        x_c = congested['density'].to_numpy().reshape(-1, 1)
        y_c = congested['volume']

        reg_u.fit(x_u, y_u)
        reg_c.fit(x_c, y_c)
        y_u_hat = reg_u.predict(x_u)
        y_c_hat = reg_c.predict(x_c)
        mse = ((y_u - y_u_hat)**2 + (y_c - y_c_hat)**2) / data.shape[0]
        if mse < best_score:
            best_score = mse
            best_d = d
            best_coefficients = (reg_u.coef_, reg_c.coef_)


def _produce_console_output(
        data: pd.DataFrame, y: str,
        scenarios: List[file_handling.ScenarioInfo], aggregation_functions,
        show_variation: bool = False,
        normalizer: int = None):
    if normalizer is None:
        try:
            normalizer = data['simulation_number'].nunique()
        except KeyError:
            normalizer = 1
    group_by_cols = ['vehicles per hour', 'control percentages']

    mixed_traffic = False
    all_accepted_risks = set()
    for sc in scenarios:
        if sc.accepted_risk is not None:
            all_accepted_risks.add(sc.accepted_risk)
        if any(0 < p < 100 for p in sc.vehicle_percentages.values()):
            mixed_traffic = True

    if len(all_accepted_risks) > 1:
        group_by_cols += ['accepted_risk']

    print(y)
    # Easier to transfer results to paper in this order
    data.sort_values('vehicles per hour',
                     kind='stable', inplace=True)
    result = data.groupby(group_by_cols, sort=False)[y].agg(
        aggregation_functions) / normalizer
    if show_variation:
        baselines = result.groupby(level=0).first()
        variation = post_processing.find_percent_variation(baselines, result)
        print(baselines)
        print(variation)
    else:
        print(result)
    if mixed_traffic and 'veh_type' in data.columns:
        group_by_cols += ['veh_type']
        result_by_type = data.groupby(group_by_cols, sort=False)[y].agg(
            aggregation_functions) / normalizer
        print(result_by_type)
        print(result_by_type / result)
