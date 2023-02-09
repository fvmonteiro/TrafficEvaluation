import os
from typing import List, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import file_handling
import post_processing
import readers
from vehicle import PlatoonLaneChangeStrategy, VehicleType, \
    vehicle_type_to_str_map


class ResultAnalyzer:
    _data_reader_map = {
        'flow': readers.DataCollectionReader,
        'vehicle_count': readers.DataCollectionReader,
        'density': readers.LinkEvaluationReader,
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
    }

    _data_processing_map = {

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

    def __init__(self, scenario_name: str, should_save_fig: bool = False):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
            self._figure_folder = ('C:\\Users\\fvall\\Google Drive\\'
                                   'PhD Research\\Lane Change\\images')
        else:
            self._figure_folder = ('G:\\My Drive\\PhD Research\\Lane Change'
                                   '\\images')
        self.scenario_name = scenario_name
        self.should_save_fig = should_save_fig

    # Plots aggregating results from multiple simulations ==================== #
    def plot_xy(self, x: str, y: str,
                vehicle_percentages: List[Dict[VehicleType, int]],
                vehicles_per_lane: int, accepted_risks: List[int] = None,
                warmup_time: int = 0):

        if not isinstance(vehicle_percentages, list):
            vehicle_percentages = [vehicle_percentages]
        data = self._load_data(y, vehicle_percentages,
                               [vehicles_per_lane], accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time * 60)
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=data, x=x, y=y,
                          hue='control percentages', ci='sd')
        plt.show()

    def plot_y_vs_time(self, y: str,
                       vehicle_percentages: List[Dict[VehicleType, int]],
                       vehicles_per_lane: int,
                       warmup_time: int = 10):
        """Plots averaged y over several runs with the same vehicle input 
        versus time.fi
        
        :param y: name of the variable being plotted.
        :param vehicle_percentages: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param vehicles_per_lane: input per lane used to generate the data
        :param warmup_time: must be given in minutes. Samples before start_time 
         are ignored.
         """

        data = self._load_data(y, vehicle_percentages,
                               [vehicles_per_lane])
        self._prepare_data_for_plotting(data, warmup_time * 60)
        relevant_data = self._ensure_data_source_is_uniform(data,
                                                            [vehicles_per_lane])
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=relevant_data, x='time', y=y,
                          hue='simulation_number', errorbar='sd')
        ax.set_title('Input: ' + str(vehicles_per_lane) + ' vehs per lane')
        if self.should_save_fig:
            self.save_fig(plt.gcf(), 'time_plot', y, [vehicles_per_lane],
                          vehicle_percentages)
        plt.show()

    def plot_fundamental_diagram(
            self, vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: List[int],
            warmup_time: int = 10, accepted_risk: int = None,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, str] = None,
            flow_sensor_identifier: Union[str, int] = None,
            link_segment_number: int = None,
            aggregation_period: int = 30):

        density_data = self._load_data('density', [vehicle_percentages],
                                       vehicles_per_lane, [accepted_risk],
                                       [platoon_lane_change_strategy],
                                       [orig_and_dest_lane_speeds])
        flow_data = self._load_data('flow', [vehicle_percentages],
                                    vehicles_per_lane, [accepted_risk],
                                    [platoon_lane_change_strategy],
                                    [orig_and_dest_lane_speeds])

        flow_data = self._prepare_data_collection_data(
            flow_data, flow_sensor_identifier, warmup_time=warmup_time * 60,
            aggregation_period=aggregation_period)
        # We're assuming scenarios with a single main link
        link = file_handling.get_scenario_main_links(self.scenario_name)[0]
        density_data = self._prepare_link_evaluation_data(
            density_data, link, link_segment_number,
            warmup_time=warmup_time * 60,
            aggregation_period=aggregation_period)
        intersection_columns = ['vehicles_per_lane', 'control percentages',
                                'simulation_number', 'time_interval',
                                'random_seed']
        data = flow_data.merge(density_data, on=intersection_columns)

        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        # for control_percentage in relevant_data.control_percentages.unique():
        #     data_to_plot = relevant_data[relevant_data['control percentages']
        #                                  == control_percentage]
        #     ax = sns.scatterplot(data=data_to_plot, x='density', y='flow')
        #     ax.set_title(control_percentage)
        #     plt.show()
        sns.scatterplot(data=relevant_data, x='density', y='flow',
                        hue='control percentages')
        plt.show()

    def plot_flow_box_plot_vs_controlled_percentage(
            self, vehicles_per_lane: List[int],
            vehicle_percentages: List[Dict[VehicleType, int]],
            accepted_risks: List[int] = None,
            warmup_time: int = 10,
            flow_sensor_name: str = 'in',
            aggregation_period: int = 30):

        if accepted_risks is None:
            accepted_risks = [0]
        data = self._load_data('flow', vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        relevant_data = self._prepare_data_collection_data(
            data, flow_sensor_name, warmup_time * 60, aggregation_period)
        self._ensure_data_source_is_uniform(relevant_data, vehicles_per_lane)
        fig = _box_plot_y_vs_controlled_percentage(relevant_data, 'flow')
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'box_plot', 'flow', vehicles_per_lane,
                vehicle_percentages=vehicle_percentages)
            self.save_fig(fig, fig_name=fig_name + '_presentation')

        _produce_console_output(relevant_data, 'flow', vehicle_percentages,
                                accepted_risks, np.median)
        # # Direct output
        # print(relevant_data.groupby(['vehicles per hour',
        #                              'control percentages'])[y].median())

    def box_plot_y_vs_controlled_percentage(
            self, y: str, vehicles_per_lane: Union[int, List[int]],
            vehicle_percentages: List[Dict[VehicleType, int]],
            warmup_time: int = 10, flow_sensor_name: List[str] = None):
        """Plots averaged y over several runs with the same vehicle input
        versus controlled vehicles percentage as a box plot.

        :param y: name of the variable being plotted.
        :param vehicles_per_lane: input per lane used to generate the data
        :param vehicle_percentages: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param flow_sensor_name: if plotting flow, we can determine choose
         which data collection measurement will be shown
        """

        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]
        data = self._load_data(y, vehicle_percentages,
                               vehicles_per_lane, accepted_risks=[0])
        self._prepare_data_for_plotting(data, warmup_time * 60,
                                        flow_sensor_name)
        relevant_data = self._ensure_data_source_is_uniform(data,
                                                            vehicles_per_lane)
        fig = _box_plot_y_vs_controlled_percentage(relevant_data, y)
        if self.should_save_fig:
            print('Must double check whether fig is being saved')
            self.save_fig(fig, 'box_plot', y, vehicles_per_lane,
                          vehicle_percentages)

    def box_plot_y_vs_vehicle_type(
            self, y: str, hue: str, vehicles_per_lane: List[int],
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            accepted_risks: List[int] = None,
            warmup_time: int = 10, sensor_name: str = None):
        """
        Plots averaged y over several runs with the same vehicle input
        versus vehicles type as a box plot. Parameter hue controls how to
        split data for each vehicle type.

        :param y: name of the variable being plotted.
        :param hue: must be 'percentage' or 'accepted_risk'
        :param vehicles_per_lane: input per lane used to generate the data
        :param percentages_per_vehicle_types: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param accepted_risks: maximum lane changing accepted risk
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param sensor_name: if plotting flow, we can determine choose
         which data collection measurement will be shown
        """
        data = self._load_data(y, percentages_per_vehicle_types,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time * 60, sensor_name)
        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)

        no_control_idx = (relevant_data['control percentages']
                          == 'human driven')
        relevant_data[['percentage', 'control_type']] = relevant_data[
            'control percentages'].str.split(' ', expand=True)
        relevant_data.loc[no_control_idx, 'control_type'] = 'human'
        relevant_data.loc[no_control_idx, 'percentage'] = '0%'
        relevant_data['Accepted Risk'] = relevant_data['accepted_risk'].map(
            {0: 'safe', 10: 'low', 20: 'medium', 30: 'high'}
        )
        if hue == 'accepted_risk':
            hue = 'Accepted Risk'

        # Plot
        plt.rc('font', size=25)
        sns.set_style('whitegrid')
        sns.boxplot(data=relevant_data, x='control_type', y=y, hue=hue)

        # Direct output
        print('vehs per lane: {}, {}'.format(vehicles_per_lane, y))
        print(relevant_data[['vehicles_per_lane', 'control percentages', y,
                             'Accepted Risk']].groupby(
            ['vehicles_per_lane', 'control percentages',
             'Accepted Risk']).median())
        plt.legend(title=hue, ncol=1,
                   bbox_to_anchor=(1.01, 1))
        plt.title(str(vehicles_per_lane) + ' vehs/h/lane', fontsize=22)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        if self.should_save_fig:
            self.save_fig(fig, 'box_plot', y, vehicles_per_lane,
                          percentages_per_vehicle_types, accepted_risks)
        self.widen_fig(plt.gcf(), len(percentages_per_vehicle_types))
        plt.tight_layout()
        plt.show()

    # def plot_risky_maneuver_histogram_per_vehicle_type(
    #         self, vehicle_percentages: List[Dict[VehicleType, int]],
    #         vehicles_per_lane: List[int],
    #         warmup_time: int = 10, min_total_risk: float = 1):
    #     """
    #     Plots histograms of risky maneuvers' total risk.
    #
    #     :param vehicle_percentages: each dictionary in the list must
    #      define the percentage of different vehicles type in the simulation
    #     :param vehicles_per_lane: input per lane used to generate the data.
    #      If this is a list, a single plot with different colors for each
    #      value is drawn.
    #     :param warmup_time: must be given in minutes. Samples before
    #      warmup_time are ignored.
    #     :param min_total_risk: risky maneuvers with total risk below this
    #      value are ignored.
    #     :return: Nothing, just plots figures
    #     """
    #
    #     data = self._load_data('total_risk', vehicle_percentages,
    #                            vehicles_per_lane, accepted_risks=[0])
    #     warmup_time *= 60  # minutes to seconds
    #     data.drop(index=data[data['total_risk'] < min_total_risk].index,
    #               inplace=True)
    #     self._prepare_data_for_plotting(data, warmup_time)
    #
    #     plt.rc('font', size=30)
    #     for control_percentage, data_to_plot in data.groupby(
    #             'control percentages'):
    #         data_to_plot = data[data['control percentages']
    #                             == control_percentage]
    #         # min_bin = int(np.floor(min_total_risk))
    #         # max_bin = int(np.ceil(data_to_plot['total_risk'].max()))
    #         # interval = max_bin // 200 + 1
    #         # bins = [i for i in range(min_bin, max_bin, interval)]
    #         sns.histplot(data=data_to_plot, x='total_risk',
    #                      stat='count', hue='vehicles per hour',
    #                      palette='tab10',)  # bins=bins)
    #         fig = plt.gcf()
    #         fig.set_size_inches(12, 6)
    #         plt.title(control_percentage)
    #         plt.tight_layout()
    #
    #         if self.should_save_fig:
    #             ctr_percent_string = str(control_percentage).replace(
    #                 '% ', '_').lower()
    #             fig_name = self.create_figure_name(
    #                 'histogram', 'risky_intervals', vehicles_per_lane,
    #                 control_percentage=ctr_percent_string)
    #             self.save_fig(plt.gcf(), fig_name=fig_name)
    #         plt.show()
    #
    #     # Direct output:
    #     n_simulations = len(data['simulation_number'].unique())
    #     group_by_cols = ['vehicles_per_lane', 'control percentages']
    #     print('Risky Maneuvers')
    #     mixed_traffic = False
    #     for vp in vehicle_percentages:
    #         mixed_traffic = mixed_traffic or any(
    #             [0 < p < 100 for p in vp.values()])
    #     if mixed_traffic:
    #         group_by_cols += ['veh_type']
    #     print(data.groupby(group_by_cols)
    #           ['total_risk'].agg([np.size, np.sum]) / n_simulations)

    def plot_risk_histograms(
            self, risk_type: str,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, min_risk: float = 0.1
    ):
        """
        Plot one histogram of lane change risks for each vehicle percentage
        each single risk value

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param vehicle_percentages: Describes the percentages of controlled
         vehicles in the simulations.
        :param vehicles_per_lane:
        :param accepted_risks:
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type, vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        self._prepare_risk_data(data, risk_type, warmup_time, min_risk)
        plt.rc('font', size=20)
        if len(accepted_risks) <= 1:
            grouped_data = data.groupby('control percentages')
        else:
            grouped_data = data.groupby(['control percentages',
                                         'accepted_risk'])
        for name, data_to_plot in grouped_data:
            if data_to_plot.empty:
                continue
            # min_bin = int(np.floor(min_total_risk))
            # max_bin = int(np.ceil(data_to_plot['total_risk'].max()))
            # interval = max_bin // 200 + 1
            # bins = [i for i in range(min_bin, max_bin, interval)]
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat='count', hue='vehicles per hour',
                         palette='tab10')
            fig = plt.gcf()
            # fig.set_size_inches(12, 6)
            # plt.title(name)
            plt.tight_layout()
            if self.should_save_fig:
                ctr_percent_string = str(name).replace('% ', '_').lower()
                fig_name = self.create_figure_name(
                    'histogram', risk_type, vehicles_per_lane,
                    control_percentage=ctr_percent_string)
                self.save_fig(fig, fig_name=fig_name + '_presentation')
            plt.show()

        normalizer = data.loc[
                (data['control percentages'] == 'human driven')
                & (data['vehicles per hour'] == data[
                    'vehicles per hour'].min()),
                risk_type].sum()
        _produce_console_output(data, risk_type, vehicle_percentages,
                                accepted_risks, [np.size, np.sum],
                                normalizer=normalizer)

    def plot_lane_change_risk_histograms_risk_as_hue(
            self, risk_type: str,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, min_risk: int = 0.1
    ):
        """
        Plot one histogram of lane change risks for each vehicle percentage.
        All risk values on the same plot

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param percentages_per_vehicle_types:
        :param vehicles_per_lane:
        :param accepted_risks:
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type,
                               percentages_per_vehicle_types,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time * 60)
        data['risk'] = data['accepted_risk'].map({0: 'safe', 10: 'low',
                                                  20: 'medium', 30: 'high'})
        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        relevant_data.drop(
            index=relevant_data[relevant_data[risk_type] < min_risk].index,
            inplace=True)
        plt.rc('font', size=30)
        # sns.set(font_scale=2)
        # for cp in control_percentages:
        for item in percentages_per_vehicle_types:
            cp = vehicle_percentage_dict_to_string(item)
            if cp == 'human driven':
                continue
            data_to_plot = relevant_data[
                (relevant_data['control percentages'] == cp)
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
                self.save_fig(fig, 'histogram', risk_type,
                              vehicles_per_lane, [item], accepted_risks)
            plt.show()

    def plot_row_of_lane_change_risk_histograms(
            self, risk_type: str,
            percentage_of_vehicle_type: Dict[VehicleType, int],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, min_risk: int = 0.1
    ):
        """
        Plots histograms of lane change risks for a single penetration
        percentage and varied risk
        :param risk_type:
        :param percentage_of_vehicle_type:
        :param vehicles_per_lane:
        :param accepted_risks:
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type,
                               [percentage_of_vehicle_type],
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time * 60)
        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        relevant_data.drop(
            index=relevant_data[relevant_data[risk_type] < min_risk].index,
            inplace=True)

        n_risks = len(accepted_risks)
        fig, axes = plt.subplots(nrows=1, ncols=n_risks)
        fig.set_size_inches(12, 6)
        for j in range(n_risks):
            ar = accepted_risks[j]
            data_to_plot = relevant_data[
                (relevant_data['accepted_risk'] == ar)
            ]
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat='count', hue='vehicles_per_lane',
                         palette='tab10', ax=axes[j])
            axes[j].set_title('accepted risk = ' + str(ar))

            print(percentage_of_vehicle_type)
            print('Count')
            print(data_to_plot[['vehicles_per_lane', risk_type]].groupby(
                'vehicles_per_lane').count())
            print('Median')
            print(data_to_plot[['vehicles_per_lane', risk_type]].groupby(
                'vehicles_per_lane').median())

        plt.tight_layout()
        if self.should_save_fig:
            self.save_fig(fig, 'histogram_row_', risk_type, vehicles_per_lane,
                          [percentage_of_vehicle_type], accepted_risks)
        plt.show()

    def plot_grid_of_lane_change_risk_histograms(
            self, risk_type: str,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, min_risk: int = 0.1
    ):
        """
        Creates a grid of size '# of vehicle types' vs '# of accepted risks'
        and plots the histogram of lane changes' total risks for each case

        :param risk_type: Options: total_risk, total_lane_change_risk and
         initial_risk
        :param percentages_per_vehicle_types:
        :param vehicles_per_lane:
        :param accepted_risks:
        :param warmup_time:
        :param min_risk:
        :return:
        """
        data = self._load_data(risk_type,
                               percentages_per_vehicle_types,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time * 60)
        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        relevant_data.drop(
            index=relevant_data[relevant_data[risk_type] < min_risk].index,
            inplace=True)

        # g = sns.FacetGrid(relevant_data, row='control percentages',
        #                   col='accepted_risk', hue='vehicles_per_lane',
        #                   margin_titles=True)
        # g.map(sns.histplot, risk_type, stat='count')

        control_percentages = relevant_data['control percentages'].unique()
        n_ctrl_pctgs = len(control_percentages)
        n_risks = len(accepted_risks)
        fig, axes = plt.subplots(nrows=n_ctrl_pctgs, ncols=n_risks)
        fig.set_size_inches(12, 6)
        for i in range(n_ctrl_pctgs):
            for j in range(n_risks):
                ctrl_percentage = control_percentages[i]
                ar = accepted_risks[j]
                data_to_plot = relevant_data[
                    (relevant_data['control percentages'] == ctrl_percentage)
                    & (relevant_data['accepted_risk'] == ar)
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
            self.save_fig(fig, 'histogram_grid', risk_type, vehicles_per_lane,
                          percentages_per_vehicle_types, accepted_risks)
        plt.show()

    def hist_plot_lane_change_initial_risks(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 0):
        """

        :param vehicle_percentages:
        :param vehicles_per_lane:
        :param warmup_time:
        :return:
        """
        lane_change_reader = readers.LaneChangeReader(self.scenario_name)
        if vehicle_percentages is None and vehicles_per_lane is None:
            data = lane_change_reader.load_test_data()
        else:
            data = lane_change_reader.load_data_with_controlled_percentage(
                vehicle_percentages, vehicles_per_lane)
        warmup_time *= 60  # minutes to seconds
        data.drop(index=data[data['start_time'] < warmup_time].index,
                  inplace=True)

        # TODO: iterate over groupby object
        for control_percentage in data['control percentages'].unique():
            data_to_plot = data[data['control percentages']
                                == control_percentage]
            plt.rc('font', size=15)
            for veh_name in ['lo', 'ld', 'fd']:
                sns.histplot(data=data_to_plot, x='initial_risk_to_' + veh_name,
                             stat='percent', hue='mandatory',
                             palette='tab10')
                plt.tight_layout()
                if self.should_save_fig:
                    fig = plt.gcf()
                    fig.set_dpi(200)
                    fig_name = ('initial_lane_change_risks_'
                                + '_'.join(str(v) for v in vehicles_per_lane)
                                + '_vehs_per_lane' + '_'
                                + control_percentage.replace('% ', '_'))
                    fig.savefig(os.path.join(self._figure_folder, fig_name))
                plt.show()

    def plot_heatmap_risk_vs_control(
            self, y: str,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, normalize: bool = False
    ):
        """

        :param y:
        :param vehicle_percentages:
        :param vehicles_per_lane:
        :param accepted_risks:
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
        for vpl in vehicles_per_lane:
            data = self._load_data(y, vehicle_percentages, [vpl],
                                   accepted_risks)
            self._prepare_data_for_plotting(data, warmup_time * 60,
                                            sensor_name=['out'])
            data.loc[data['control percentages'] == 'human driven',
                     'control percentages'] = '100% HD'
            n_simulations = len(data['simulation_number'].unique())
            table = data.pivot_table(values=col_in_df,
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
                self.save_fig(plt.gcf(), 'heatmap', y, [vpl],
                              vehicle_percentages, accepted_risks)
            plt.show()

    def plot_risk_heatmap(
            self, risk_type: str,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None,
            normalizer: float = None) -> float:
        """
        Plots a heatmap of the chosen risk type. Returns normalizer so that
        we can reuse the value again if needed.
        :param risk_type: total_risk, total_lane_change_risk or both. If
         both, the lane change risk is normalized using the values from
         total_risk
        :param vehicle_percentages: Describes the percentages of controlled
         vehicles in the simulations.
        :param vehicles_per_lane: Vehicle inputs for which we want SSMs
         computed.
        :param accepted_risks: Accepted lane change risk.
        :param normalizer: All values are divided by the normalizer
        :returns: The value used to normalize all values
        """
        if accepted_risks is None:
            accepted_risks = [0]

        if risk_type == 'both':
            normalizer = self.plot_risk_heatmap(
                'total_risk', vehicle_percentages,
                vehicles_per_lane, accepted_risks, normalizer)
            self.plot_risk_heatmap('total_lane_change_risk',
                                   vehicle_percentages, vehicles_per_lane,
                                   accepted_risks, normalizer)
            return normalizer

        data = self._load_data(risk_type, vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, 600)
        agg_function = np.sum
        if normalizer is None:
            normalizer = data.loc[
                (data['control percentages'] == 'human driven')
                & (data['vehicles per hour'] == data[
                    'vehicles per hour'].min()),
                risk_type].sum()
        title = ' '.join(risk_type.split('_')[1:])
        fig = _plot_heatmap(data, risk_type, 'vehicles per hour',
                            'control percentages', normalizer, title,
                            agg_function=agg_function)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', risk_type,
                vehicles_per_lane, vehicle_percentages)
            self.save_fig(fig, fig_name=fig_name)
        return normalizer

    def plot_fd_discomfort(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None,
            brake_threshold: int = 4):
        y = 'fd_discomfort'
        data = self._load_data(y, vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, 600)
        col_name = '_'.join([y, str(brake_threshold)])
        agg_function = np.mean
        normalizer = data.loc[
            (data['control percentages'] == 'human driven')
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
            fig_name = self.create_figure_name(
                'heatmap', col_name,
                vehicles_per_lane, vehicle_percentages)
            self.save_fig(fig, fig_name=fig_name)

        _produce_console_output(data, col_name, vehicle_percentages,
                                accepted_risks, agg_function,
                                show_variation=True)

    def plot_discomfort_heatmap(
            self,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None,
            max_brake: int = 4):
        data = self._load_data('discomfort', vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        y = 'discomfort_' + str(max_brake)
        self._prepare_data_for_plotting(data, 600)
        normalizer = data.loc[
            (data['control percentages'] == 'human driven')
            & (data['vehicles_per_lane'] == data['vehicles_per_lane'].min()),
            y].sum()
        fig = _plot_heatmap(data, y, 'vehicles_per_lane',
                            'control percentages', normalizer)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', y,
                vehicles_per_lane, vehicle_percentages)
            self.save_fig(fig, fig_name=fig_name)

    def plot_total_output_heatmap(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None):
        y = 'vehicle_count'
        data = self._load_data('vehicle_count', vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, 600, ['out'])
        normalizer = data.loc[
            (data['control percentages'] == 'human driven')
            & (data['vehicles_per_lane'] == data[
                'vehicles_per_lane'].min()),
            y].sum()
        fig = _plot_heatmap(data, 'vehicle_count', 'vehicles_per_lane',
                            'control percentages', normalizer, 'Output flow')
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', 'vehicle_count', vehicles_per_lane,
                vehicle_percentages)
            self.save_fig(fig, fig_name=fig_name)

    def plot_emission_heatmap(
            self,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            accepted_risks: List[int] = None,
            pollutant_id: int = 91
    ):
        y = 'emission_per_volume'
        data = self._load_data(y, vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        self._prepare_pollutant_data(data, pollutant_id)
        title = 'Normalized ' + self._pollutant_id_to_string[pollutant_id]
        normalizer = data.loc[
            (data['control percentages'] == 'human driven')
            & (data['vehicles per hour']
               == data['vehicles per hour'].min()),
            y].sum()

        fig = _plot_heatmap(data, y, 'vehicles per hour', 'control percentages',
                            normalizer, title)
        if self.should_save_fig:
            fig_name = self.create_figure_name(
                'heatmap', self._pollutant_id_to_string[pollutant_id],
                vehicles_per_lane, vehicle_percentages)
            self.save_fig(fig, fig_name=fig_name)
        _produce_console_output(data, y, vehicle_percentages,
                                [0], np.sum, show_variation=True)

    def plot_lane_change_count_heatmap(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None,
            warmup_time: int = 10):
        if accepted_risks is None:
            accepted_risks = [0]
        y = 'lane_change_count'
        agg_function = np.count_nonzero
        col_to_count = 'veh_id'

        data = self._load_data(y, vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        post_processing.drop_warmup_samples(data, warmup_time)
        n_simulations = len(data['simulation_number'].unique())
        _plot_heatmap(data, col_to_count, 'vehicles per hour',
                      'control percentages', normalizer=n_simulations,
                      title='lane change count',
                      agg_function=agg_function)
        _produce_console_output(data, col_to_count, vehicle_percentages,
                                accepted_risks, agg_function)

    def print_summary_of_issues(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10):
        """
        Prints out a summary with average number of times the AV requested
        human intervention and average number of vehicles removed from
        simulation
        :param vehicle_percentages:
        :param vehicles_per_lane:
        :param accepted_risks:
        :param warmup_time:
        :return:
        """
        data = self._load_data('lane_change_issues',
                               vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time * 60)
        n_simulations = len(data['simulation_number'].unique())
        if len(accepted_risks) <= 1:
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
                               vehicle_percentages,
                               vehicles_per_lane, accepted_risks)

        print(data.groupby(['vehicles_per_lane', 'control percentages'])[
                  'vissim_in_control'].mean())

    def plot_all_pollutant_heatmaps(
            self,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            accepted_risks: List[int] = None):

        for p_id in self._pollutant_id_to_string:
            self.plot_emission_heatmap(vehicle_percentages, vehicles_per_lane,
                                       accepted_risks, p_id)

    def count_lane_changes_from_vehicle_record(
            self, vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: int, accepted_risk: int = None,
            platoon_lane_change_strategy: PlatoonLaneChangeStrategy = None,
            orig_and_dest_lane_speeds: Tuple[int, str] = None
    ):
        """
        Counts the lane changes over all simulations under a determined
        simulation configuration
        """
        print(vehicle_percentages)
        warmup_time = 600
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
            vehicle_percentages, vehicles_per_lane,
            accepted_risk=accepted_risk,
            platoon_lane_change_strategy=platoon_lane_change_strategy,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds)
        for (data, simulation_parameters) in data_generator:
            data.drop(index=data[~data['link'].isin(links)].index,
                      inplace=True)
            data.drop(index=data[data['time'] < warmup_time].index,
                      inplace=True)
            data.sort_values('veh_id', kind='stable', inplace=True)
            data['is_lane_changing'] = data['lane_change'] != 'None'
            data['lc_transition'] = data['is_lane_changing'].diff()
            lc_counter.append(np.count_nonzero(data['lc_transition']) / 2)
        print('LCs from veh records: ', sum(lc_counter))

    # Platoon LC plots ======================================================= #

    def plot_y_vs_vehicle_input(
            self, y: str, vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: List[int],
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            orig_and_dest_lane_speeds: Tuple[int, str]):
        """
        Line plot with vehicle input on the x axis and LC strategies as hue

        :param y: Options: was_lane_change_completed, maneuver_time,
        travel_time, accel_cost, stayed_in_platoon
        """
        self.plot_results_for_platoon_scenario(
            y, 'Main Road Input (vehs/h)', 'Strategy', vehicle_percentages,
            vehicles_per_lane, lane_change_strategies,
            orig_and_dest_lane_speeds, is_bar_plot=True
        )

    def plot_y_vs_platoon_lc_strategy(
            self, y: str, vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: List[int],
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            orig_and_dest_lane_speeds: Tuple[int, str]):
        """
        Line plot with strategies on the x axis and vehicle input as hue
        :param y: Options: was_lane_change_completed, maneuver_time,
        travel_time, accel_cost, stayed_in_platoon
        """
        self.plot_results_for_platoon_scenario(
            y, 'Strategy', 'Main Road Input (vehs/h)', vehicle_percentages,
            vehicles_per_lane, lane_change_strategies, orig_and_dest_lane_speeds
        )

    def plot_results_for_platoon_scenario(
            self, y: str, x: str, hue: str,
            vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: List[int],
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            orig_and_dest_lane_speeds: Tuple[int, str],
            is_bar_plot: bool = False):
        """

        """
        if 'platoon' not in self.scenario_name:
            raise ValueError('Must be scenario with platoon lane changes')

        reader = readers.PlatoonLaneChangeEfficiencyReader(self.scenario_name)
        data = reader.load_platoon_scenario_data(
            [vehicle_percentages], vehicles_per_lane,
            lane_change_strategies, [orig_and_dest_lane_speeds])

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
        st_name_map = {-1: 'humans', 0: 'CAVs', 1: 'SBP', 2: 'Ld First',
                       3: 'LV First', 4: 'Ld First Rev.'}
        data['Strategy'] = data['lane_change_strategy'].apply(
            lambda q: st_name_map[q])
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
            self, y: str,
            vehicle_percentages: Dict[VehicleType, int],
            vehicles_per_lane: int,
            lane_change_strategies: List[PlatoonLaneChangeStrategy],
            dest_lane_speeds: List[str]):
        """

        """
        if 'platoon' not in self.scenario_name:
            raise ValueError('Must be scenario with platoon lane changes')

        orig_and_dest_lane_speeds = [(80, s) for s in dest_lane_speeds]
        reader = readers.PlatoonLaneChangeEfficiencyReader(self.scenario_name)
        data = reader.load_platoon_scenario_data(
            [vehicle_percentages], [vehicles_per_lane],
            lane_change_strategies, orig_and_dest_lane_speeds)

        # Presentation naming
        y_name_map = {
            'was_lane_change_completed': '% Successful Lane Changes',
            'vehicle_maneuver_time': 'Maneuver Time per Vehicle (s)',
            'platoon_maneuver_time': 'Platoon Maneuver Time (s)',
            'travel_time': 'Travel Time (s)',
            'accel_costs': 'Accel Cost (m2/s3)',
            'stayed_in_platoon': '% Stayed in Platoon'
        }
        st_name_map = {-1: 'humans', 0: 'CAVs', 1: 'SBP', 2: 'Ld First',
                       3: 'LV First', 4: 'Ld First Rev.'}
        data['Strategy'] = data['lane_change_strategy'].apply(
            lambda q: st_name_map[q])
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

    # Traffic Light Scenario Plots =========================================== #

    def plot_violations_per_control_percentage(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 10):
        """
        Plots number of violations .

        :param vehicle_percentages: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param vehicles_per_lane: input per lane used to generate the data.
         If this is a list, a single plot with different colors for each
         value is drawn.
        :param warmup_time: must be given in minutes. Samples before
         warmup_time are ignored.
        :return: Nothing, just plots figures
        """
        # if not isinstance(vehicles_per_lane, list):
        #     vehicles_per_lane = [vehicles_per_lane]

        n_simulations = 10
        violations_reader = readers.ViolationsReader(self.scenario_name)
        data = violations_reader.load_data_with_controlled_percentage(
            vehicle_percentages, vehicles_per_lane)

        # TODO: temporary
        data.drop(index=data[data['simulation_number'] == 11].index,
                  inplace=True)

        warmup_time *= 60  # minutes to seconds
        self._prepare_data_for_plotting(data, warmup_time * 60)
        results = data.groupby(['control percentages', 'vehicles_per_lane'],
                               as_index=False)['veh_id'].count().rename(
            columns={'veh_id': 'violations count'})
        results['mean violations'] = results['violations count'] / n_simulations
        print(results)

    def plot_heatmap_for_traffic_light_scenario(
            self, y: str,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 10):
        """
        Plots a heatmap
        :param y:
        :param vehicle_percentages:
        :param vehicles_per_lane:
        :param warmup_time:
        :return:
        """

        title_map = {'vehicle_count': 'Output Flow',
                     'discomfort': 'Discomfort',
                     'barrier_function_risk': 'Risk'}
        n_simulations = 10
        plt.rc('font', size=17)
        for vpl in vehicles_per_lane:
            data = self._load_data(y, vehicle_percentages, [vpl])
            self._prepare_data_for_plotting(data, warmup_time * 60)
            table = pd.pivot_table(
                # data[[y, 'traffic_light_acc_percentage',
                #       'traffic_light_cacc_percentage']],
                data,
                values=y,
                index=['traffic_light_cacc_percentage'],
                columns=['traffic_light_acc_percentage'], aggfunc=np.sum)
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
                self.save_fig(fig, 'heatmap', y, vpl,
                              vehicle_percentages)
            plt.tight_layout()
            plt.show()

    def plot_violations_heatmap(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], warmup_time: int = 10):
        n_simulations = 10
        plt.rc('font', size=17)
        for vpl in vehicles_per_lane:
            data = self._load_data('violations', vehicle_percentages,
                                   [vpl])
            self._prepare_data_for_plotting(data, warmup_time * 60)
            table = pd.pivot_table(
                data[['veh_id', 'traffic_light_acc_percentage',
                      'traffic_light_cacc_percentage']], values='veh_id',
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
                self.save_fig(fig, 'heatmap', 'violations', vpl,
                              vehicle_percentages)
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
        veh_record = reader.load_single_file_from_scenario(
            1, vehicle_percentage, vehicles_per_lane, 0)
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
        veh_record = reader.load_single_file_from_scenario(
            1, vehicle_percentage, vehicles_per_lane, 0)
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
        print('total risk:',  single_veh['risk'].sum() * sampling)

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

    def get_flow_and_risk_plots(self, veh_inputs: List[int],
                                percentages_per_vehicle_types: List[
                                    Dict[VehicleType, int]]):
        """Generates the plots used in the Safe Lane Changes paper."""

        self.box_plot_y_vs_controlled_percentage(
            'flow', veh_inputs, percentages_per_vehicle_types, warmup_time=10
        )
        # self.box_plot_y_vs_controlled_percentage(
        #     'risk', veh_inputs, percentages_per_vehicle_types, warmup_time=10
        # )
        self.plot_risk_histograms('total_risk', percentages_per_vehicle_types,
                                  veh_inputs, [0], min_risk=1)

    # Support methods ======================================================== #
    def _load_data(
            self, y: str, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None,
            lane_change_strategies: List[PlatoonLaneChangeStrategy] = None,
            orig_and_dest_lane_speeds: List[Tuple[int, str]] = None
            ):
        # reader = self._get_data_reader(y)
        reader = self._data_reader_map[y](self.scenario_name)
        data = reader.load_data_from_several_scenarios(
            vehicle_percentages, vehicles_per_lane, accepted_risks,
            lane_change_strategies, orig_and_dest_lane_speeds)
        data['vehicles per hour'] = (
                file_handling.get_scenario_number_of_lanes(self.scenario_name)
                * data['vehicles_per_lane'])
        return data

    def _prepare_data_collection_data(
            self, data: pd.DataFrame, sensor_identifier: Union[int, str] = None,
            warmup_time: int = 600,
            aggregation_period: int = 30) -> pd.DataFrame:
        """
        Keeps only data from one sensor, discards data before warmup time, and
        computes flow for the given aggregation period.
        """
        # Drop early samples
        post_processing.create_time_in_minutes_from_intervals(data)
        data['time'] *= 60
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)
        # Select only one sensor
        if self.scenario_name.startswith('in_and_out'):
            _select_flow_sensors_from_in_and_out_scenario(data,
                                                          sensor_identifier)
        else:
            data.drop(data[data['sensor_number'] != sensor_identifier].index,
                      inplace=True)
        # Aggregate
        aggregated_data = _aggregate_data(data, 'vehicle_count',
                                          aggregation_period, np.sum)
        aggregated_data['flow'] = (3600 / aggregation_period
                                   * aggregated_data['vehicle_count'])
        return aggregated_data

    def _prepare_link_evaluation_data(
            self, data: pd.DataFrame, link: int, segment: int = None,
            lanes: List[int] = None, sensor_name: str = None,
            warmup_time: int = 600,
            aggregation_period: int = 30) -> pd.DataFrame:

        # Drop early samples
        post_processing.create_time_in_minutes_from_intervals(data)
        data['time'] *= 60
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)

        # Select link
        data.drop(index=data[data['link_number'] != link].index,
                  inplace=True)
        # Select segment
        if segment is not None:
            data.drop(index=data[data['link_segment'] != segment].index,
                      inplace=True)
        elif len(data['link_segment'].unique()) > 1:
            print("WARNING: the chosen link has several segments, and we're "
                  "keeping all of them")
        # Select lanes
        if self.scenario_name.startswith('in_and_out'):
            _select_lanes_for_in_and_out_scenario(data, sensor_name)
        elif lanes is not None and _has_per_lane_results(data):
            data.drop(data[~data['lane'].isin(lanes)].index,
                      inplace=True)

        return _aggregate_data(data, 'density', aggregation_period, np.mean)

    @staticmethod
    def _prepare_data_for_plotting(data: pd.DataFrame,
                                   warmup_time: float = 0,
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
         Must be passed in seconds
        :param sensor_name: if plotting flow or density, we can determine
         which sensor/lane is shown
        """
        # TODO: call different methods based on the data type
        if sensor_name is not None and not isinstance(sensor_name, list):
            sensor_name = [sensor_name]
        if 'time' not in data.columns:
            post_processing.create_time_in_minutes_from_intervals(data)
            data['time'] *= 60
        if 'flow' in data.columns:  # data collection data
            if sensor_name is None:
                sensor_name = ['in']
            sensor_number = [ResultAnalyzer._sensor_name_map[name] for name
                             in sensor_name]
            data.drop(data[~data['sensor_number'].isin(sensor_number)].index,
                      inplace=True)
        if (('density' in data.columns) and (sensor_name is not None)
                and (_has_per_lane_results(data))):
            # link evaluation data
            lanes = []
            [lanes.extend(ResultAnalyzer._sensor_lane_map[name])
             for name in sensor_name]
            data.drop(data[~data['lane'].isin(lanes)].index,
                      inplace=True)
        if ('pollutant_id' in data.columns) and pollutant_id is not None:
            data.drop(data[~(data['pollutant_id'] == pollutant_id)].index,
                      inplace=True)
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)

    def _prepare_pollutant_data(self, data: pd.DataFrame,
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
        data.drop(index=data[data[risky_type] < min_risk].index,
                  inplace=True)
        post_processing.drop_warmup_samples(data, warmup_time)

    @staticmethod
    def _ensure_data_source_is_uniform(data: pd.DataFrame,
                                       vehicles_per_lane: List[int]):
        """
        Only keeps the result obtained with a certain random seed if that 
        random seed was used by all controlled vehicle percentages
        :param data:
        :param vehicles_per_lane:
        :return:
        """
        # Get the intersection of random seeds for each control percentage
        random_seeds = np.array([])
        for veh_input in vehicles_per_lane:
            for percent in data['control percentages'].unique():
                current_random_seeds = data.loc[
                    (data['control percentages'] == percent)
                    & (data['vehicles_per_lane'] == veh_input),
                    'random_seed'].unique()
                if random_seeds.size == 0:
                    random_seeds = current_random_seeds
                else:
                    random_seeds = np.intersect1d(random_seeds,
                                                  current_random_seeds)

        # Keep only the random_seeds used by all control percentages
        relevant_data = data.drop(index=data[~data[
            'random_seed'].isin(random_seeds)].index)
        if relevant_data.shape[0] != data.shape[0]:
            print('Some simulations results were dropped because not all '
                  'controlled percentages or vehicle inputs had the same '
                  'amount of simulation results')
        return relevant_data

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

    def save_fig(self, fig: plt.Figure,
                 plot_type: str = None,
                 measurement_name: str = None,
                 vehicles_per_lane: Union[int, List[int]] = None,
                 vehicle_percentages: List[Dict[VehicleType, int]] = None,
                 accepted_risk: List[int] = None,
                 fig_name: str = None
                 ):
        # Making the figure nice for inclusion in documents
        # self.widen_fig(fig, controlled_percentage)
        # plt.rc('font', size=20)
        if not fig_name:
            if not isinstance(vehicles_per_lane, list):
                vehicles_per_lane = [vehicles_per_lane]
            vehicles_types, temp = list_of_dicts_to_1d_list(
                vehicle_percentages)
            veh_penetration_strings = []
            for i in range(len(vehicles_types)):
                veh_penetration_strings.append(str(temp[i]) + '_'
                                               + vehicles_types[i].name.lower())
            fig_name = (
                    plot_type + '_' + measurement_name + '_'
                    + self.scenario_name + '_'
                    + '_'.join(str(v) for v in sorted(vehicles_per_lane)) + '_'
                    + 'vehs_per_lane' + '_'
                    + '_'.join(sorted(veh_penetration_strings))
            )
            if accepted_risk:
                fig_name += '_risks_' + '_'.join(str(ar) for ar
                                                 in accepted_risk)
        fig.set_dpi(400)
        # axes = fig.axes
        plt.tight_layout()
        fig.savefig(os.path.join(self._figure_folder, fig_name), dpi=400)

    def create_figure_name(
            self, plot_type: str, measurement_name: str,
            vehicles_per_lane: Union[int, List[int]],
            vehicle_percentages: List[Dict[VehicleType, int]] = None,
            control_percentage: str = None) -> str:

        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]
        vehicle_input_string = [str(vi) for vi in sorted(vehicles_per_lane)]

        if vehicle_percentages:
            percentage_strings = set()
            vt_strings = set()
            for vp in vehicle_percentages:
                for vt, p in vp.items():
                    percentage_strings.add(str(p))
                    vt_strings.add(vt.name.lower())
            percentage_strings = sorted(list(percentage_strings))
            vt_strings = sorted(list(vt_strings))
            fig_name = '_'.join([plot_type, measurement_name,
                                 self.scenario_name]
                                + vehicle_input_string + ['vehs_per_lane']
                                + percentage_strings + vt_strings)
        else:
            fig_name = '_'.join([plot_type, measurement_name,
                                 self.scenario_name]
                                + vehicle_input_string + ['vehs_per_lane']
                                + [control_percentage])
        # if accepted_risk:
        #     fig_name += '_risks_' + '_'.join(str(ar) for ar in accepted_risk)
        return fig_name

    def widen_fig(self, fig: plt.Figure, n_boxes: int):
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
            self,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicle_inputs):
        """ Checks whether simulations crashed. This is necessary because,
        when doing multiple runs from the COM interface, VISSIM does not
        always indicate that a simulation crashed. """
        reader = readers.DataCollectionReader(self.scenario_name)
        raw_data = reader.load_data_from_several_scenarios(
            vehicle_percentages, vehicle_inputs, accepted_risks=[0])
        data = self._prepare_data_collection_data(raw_data, 'in', 0, 60)
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
        print("Min issue time {} at simulation {}".format(min(issue_time),
                                                          issues.index[
                                                              np.argmin(
                                                                  issue_time)]))


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
        return 'human driven'
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + '% ' + vehicle_type_to_str_map[veh_type])
    return ' '.join(sorted(ret_str))


def _aggregate_data(data: pd.DataFrame, aggregated_variable: str,
                    aggregation_period: int,
                    aggregation_function):
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
    data.drop(data[data['sensor_number'] != sensor_number].index,
              inplace=True)


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


def _box_plot_y_vs_controlled_percentage(relevant_data: pd.DataFrame,
                                         y: str):
    # Plot
    fig = plt.figure()
    plt.rc('font', size=20)
    fig.set_size_inches(9, 6)
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=relevant_data,  # orient='h',
                     x='control percentages', y=y,
                     hue='vehicles per hour')
    # if self.should_save_fig:
    #     self.save_fig(plt.gcf(), 'box_plot', y, vehicles_per_lane,
    #                   vehicle_percentages)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              title='vehicles per hour', ncols=2)
    plt.tight_layout()
    plt.show()
    return fig


def _produce_console_output(
        data: pd.DataFrame, y: str,
        vehicle_percentages: List[Dict[VehicleType, int]],
        accepted_risks: List[int], aggregation_functions,
        show_variation: bool = False,
        normalizer: int = None):
    if normalizer is None:
        try:
            normalizer = len(data['simulation_number'].unique())
        except KeyError:
            normalizer = 1
    group_by_cols = ['vehicles per hour', 'control percentages']
    varied_accepted_risk = len(accepted_risks) > 1
    if varied_accepted_risk:
        group_by_cols += ['accepted_risk']
    mixed_traffic = False
    for vp in vehicle_percentages:
        mixed_traffic = mixed_traffic or any(
            [0 < p < 100 for p in vp.values()])
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
