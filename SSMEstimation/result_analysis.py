from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import List, Dict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import post_processing
import readers
from vehicle import VehicleType, vehicle_type_to_str_map


# # No longer in use [20/10/22]
# def list_of_dicts_to_2d_lists(dict_list: List[Dict]):
#     keys = []
#     values = []
#     for d in dict_list:
#         keys.append(list(d.keys()))
#         values.append(list(d.values()))
#     return keys, values


def list_of_dicts_to_1d_list(dict_list: List[Dict]):
    keys = []
    values = []
    for d in dict_list:
        keys += list(d.keys())
        values += list(d.values())
    return keys, values


def vehicle_percentage_dict_to_string(vp_dict: Dict[VehicleType, int]) -> str:
    if sum(vp_dict.values()) == 0:
        return 'no control'
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(str(p) + '% ' + vehicle_type_to_str_map[veh_type])
    return ' '.join(sorted(ret_str))


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

    def __init__(self, network_name: str, should_save_fig: bool = False):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
            self._figure_folder = ('C:\\Users\\fvall\\Google Drive\\Lane '
                                   'Change\\images')
        else:
            self._figure_folder = ('G:\\My Drive\\Lane Change'
                                   '\\images')
        self.network_name = network_name
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
                          hue='control_percentages', ci='sd')
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
            self, vehicles_per_lane: List[int],
            vehicle_percentages: List[Dict[VehicleType, int]],
            warmup_time: int = 10, accepted_risks: List[int] = None,
            flow_sensor_name: List[str] = None):

        density_data = self._load_data('density',
                                       vehicle_percentages,
                                       vehicles_per_lane, accepted_risks)
        flow_data = self._load_data('flow', vehicle_percentages,
                                    vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(density_data, warmup_time * 60,
                                        flow_sensor_name)
        self._prepare_data_for_plotting(flow_data, warmup_time * 60,
                                        flow_sensor_name)
        intersection_columns = ['vehicles_per_lane', 'control_percentages',
                                'simulation_number', 'time_interval',
                                'random_seed']
        data = flow_data.merge(density_data, on=intersection_columns)

        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        for control_percentage in relevant_data.control_percentages.unique():
            data_to_plot = relevant_data[relevant_data['control_percentages']
                                         == control_percentage]
            ax = sns.scatterplot(data=data_to_plot, x='density', y='flow')
            ax.set_title(control_percentage)
            plt.show()
        sns.scatterplot(data=relevant_data, x='density', y='flow',
                        hue='control_percentages')
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
                               vehicles_per_lane, accepted_risks=[0])
        relevant_data = ResultAnalyzer._prepare_data_collection_data(
            data, flow_sensor_name, warmup_time*60, aggregation_period)
        self._ensure_data_source_is_uniform(relevant_data, vehicles_per_lane)
        self._box_plot_y_vs_controlled_percentage(relevant_data, 'flow')

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
        fig = self._box_plot_y_vs_controlled_percentage(relevant_data, y)
        if self.should_save_fig:
            print('Must double check whether fig is being saved')
            self.save_fig(fig, 'box_plot', y, vehicles_per_lane,
                          vehicle_percentages)


    def _box_plot_y_vs_controlled_percentage(self, relevant_data: pd.DataFrame,
                                             y: str):

        # Plot
        fig = plt.figure()
        plt.rc('font', size=15)
        sns.set_style('whitegrid')
        sns.boxplot(data=relevant_data,  # orient='h',
                    x='control_percentages', y=y,
                    hue='vehicles_per_lane')
        # if self.should_save_fig:
        #     self.save_fig(plt.gcf(), 'box_plot', y, vehicles_per_lane,
        #                   vehicle_percentages)
        plt.tight_layout()
        plt.show()

        # Direct output
        print(relevant_data.groupby(['vehicles_per_lane',
                                     'control_percentages'])[y].median())
        return fig

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

        no_control_idx = (relevant_data['control_percentages']
                          == 'no control')
        relevant_data[['percentage', 'control_type']] = relevant_data[
            'control_percentages'].str.split(' ', expand=True)
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
        print(relevant_data[['vehicles_per_lane', 'control_percentages', y,
                             'Accepted Risk']].groupby(
            ['vehicles_per_lane', 'control_percentages',
             'Accepted Risk']).median())
        # for ct in relevant_data['control_type'].unique():
        #     print('vehs per lane: {}, ct: {}, median {}: {}'.format(
        #         vehicles_per_lane, ct, y,
        #         relevant_data.loc[relevant_data['control_type'] == ct,
        #                           y].median()
        #     ))
        # legend_len = len(relevant_data[hue].unique())
        # n_legend_cols = (legend_len if legend_len < 5
        #                  else legend_len // 2 + 1)
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

    def plot_risky_maneuver_histogram_per_vehicle_type(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 10, min_total_risk: float = 1):
        """
        Plots histograms of risky maneuvers' total risk.

        :param vehicle_percentages: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param vehicles_per_lane: input per lane used to generate the data.
         If this is a list, a single plot with different colors for each
         value is drawn.
        :param warmup_time: must be given in minutes. Samples before
         warmup_time are ignored.
        :param min_total_risk: risky maneuvers with total risk below this
         value are ignored.
        :return: Nothing, just plots figures
        """
        # if not isinstance(vehicles_per_lane, list):
        #     vehicles_per_lane = [vehicles_per_lane]

        risky_maneuver_reader = readers.RiskyManeuverReader(self.network_name)
        data = risky_maneuver_reader.load_data_with_controlled_percentage(
            vehicle_percentages, vehicles_per_lane, accepted_risks=[0])
        warmup_time *= 60  # minutes to seconds
        data.drop(index=data[data['total_risk'] < min_total_risk].index,
                  inplace=True)
        self._prepare_data_for_plotting(data, warmup_time)

        n_simulations = 10
        # for control_percentage in data['control_percentages'].unique():
        for control_percentage, data_to_plot in data.groupby(
                'control_percentages'):
            data_to_plot = data[data['control_percentages']
                                == control_percentage]
            plt.rc('font', size=15)
            min_bin = int(np.floor(min_total_risk))
            max_bin = int(np.ceil(data_to_plot['total_risk'].max()))
            interval = max_bin // 200 + 1
            bins = [i for i in range(min_bin, max_bin, interval)]
            sns.histplot(data=data_to_plot, x='total_risk',
                         stat='count', hue='vehicles_per_lane',
                         palette='tab10', bins=bins)
            plt.tight_layout()
            if self.should_save_fig:
                fig = plt.gcf()
                fig.set_dpi(200)
                fig_name = ('risky_intervals_'
                            + '_'.join(str(v) for v in vehicles_per_lane) + '_'
                            + 'vehs_per_lane' + '_'
                            + str(control_percentage).replace('% ', '_'))
                fig.savefig(os.path.join(self._figure_folder, fig_name))
            plt.show()

        # Direct output:
        print(data.groupby(['vehicles_per_lane', 'control_percentages'])
              ['total_risk'].agg([np.size, np.sum]) / n_simulations)

    def plot_lane_change_risk_histograms(
            self, risk_type: str,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, min_risk: int = 0.1
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
        self._prepare_data_for_plotting(data, warmup_time*60)
        relevant_data = self._ensure_data_source_is_uniform(
            data, vehicles_per_lane)
        relevant_data.drop(
            index=relevant_data[relevant_data[risk_type] < min_risk].index,
            inplace=True)
        plt.rc('font', size=30)
        grouped_data = relevant_data.groupby(['control_percentages',
                                              'accepted_risk'])
        for name, data_to_plot in grouped_data:
            if data_to_plot.empty:
                continue
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat='count', hue='vehicles_per_lane',
                         palette='tab10')
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            plt.title(name)
            plt.tight_layout()
            # if self.should_save_fig:
            #     self.save_fig(fig, 'histogram', risk_type,
            #                   vehicles_per_lane, [item], [ar])
            plt.show()

        # Direct output
        print(relevant_data.groupby(
            ['vehicles_per_lane', 'control_percentages', 'accepted_risk']
            )[risk_type].agg([np.size, np.median]))

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
        self._prepare_data_for_plotting(data, warmup_time*60)
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
            if cp == 'no control':
                continue
            data_to_plot = relevant_data[
                (relevant_data['control_percentages'] == cp)
                ]
            sns.histplot(data=data_to_plot, x=risk_type,
                         stat='count', hue='risk',
                         palette='tab10')
            # plt.legend(title='Risk', labels=['safe', 'low', 'medium'])
            # Direct output
            print('veh penetration: {}'.format(cp))
            print(data_to_plot[['control_percentages', 'veh_type', 'risk',
                                risk_type]].groupby(
                ['control_percentages', 'veh_type', 'risk']).count())
            print(data_to_plot[['control_percentages', 'risk', risk_type]].groupby(
                ['control_percentages', 'risk']).median())
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

        # g = sns.FacetGrid(relevant_data, row='control_percentages',
        #                   col='accepted_risk', hue='vehicles_per_lane',
        #                   margin_titles=True)
        # g.map(sns.histplot, risk_type, stat='count')

        control_percentages = relevant_data['control_percentages'].unique()
        n_ctrl_pctgs = len(control_percentages)
        n_risks = len(accepted_risks)
        fig, axes = plt.subplots(nrows=n_ctrl_pctgs, ncols=n_risks)
        fig.set_size_inches(12, 6)
        for i in range(n_ctrl_pctgs):
            for j in range(n_risks):
                ctrl_percentage = control_percentages[i]
                ar = accepted_risks[j]
                data_to_plot = relevant_data[
                    (relevant_data['control_percentages'] == ctrl_percentage)
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
        lane_change_reader = readers.LaneChangeReader(self.network_name)
        if vehicle_percentages is None and vehicles_per_lane is None:
            data = lane_change_reader.load_test_data()
        else:
            data = lane_change_reader.load_data_with_controlled_percentage(
                vehicle_percentages, vehicles_per_lane)
        warmup_time *= 60  # minutes to seconds
        self._create_single_control_percentage_column(data)
        data.drop(index=data[data['start_time'] < warmup_time].index,
                  inplace=True)

        # TODO: iterate over groupby object
        for control_percentage in data['control_percentages'].unique():
            data_to_plot = data[data['control_percentages']
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
            data.loc[data['control_percentages'] == 'no control',
                     'control_percentages'] = '100% HD'
            n_simulations = len(data['simulation_number'].unique())
            table = data.pivot_table(values=col_in_df,
                                     index=['accepted_risk'],
                                     columns=['control_percentages'],
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
                            'control_percentages'))
            plt.xlabel('', fontsize=22)
            plt.ylabel('accepted initial risk', fontsize=22)
            plt.title(title + ' at ' + str(vpl) + ' vehs/h/lane',
                      fontsize=22)
            plt.tight_layout()
            if self.should_save_fig:
                self.save_fig(plt.gcf(), 'heatmap', y, [vpl],
                              vehicle_percentages, accepted_risks)
            plt.show()

    # def plot_heatmap_input_vs_control(
    #         self, y: str,
    #         vehicle_percentages: List[Dict[VehicleType, int]],
    #         vehicles_per_lane: List[int],
    #         accepted_risks: List[int] = None):
    #
    #     data = self._load_data(y, vehicle_percentages,
    #                            vehicles_per_lane, accepted_risks)
    #     self._prepare_data_for_plotting(data, 600, sensor_name=['out'])
    #     self.plot_heatmap(data, y, 'vehicles_per_lane')

    def plot_fd_discomfort(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None,
            brake_threshold: int = 4):
        y = 'fd_discomfort'
        data = self._load_data(y, vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        return

    def plot_total_output_heatmap(
            self, vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int] = None):
        y = 'vehicle_count'
        data = self._load_data('vehicle_count', vehicle_percentages,
                               vehicles_per_lane, accepted_risks)
        self._prepare_data_for_plotting(data, 600, ['out'])
        normalizer = data.loc[
            (data['control_percentages'] == 'no control')
            & (data['vehicles_per_lane'] == data['vehicles_per_lane'].min()),
            y].sum()
        self.plot_heatmap(data, 'vehicle_count', 'vehicles_per_lane',
                          normalizer, 'Output flow')

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
            (data['control_percentages'] == 'no control')
            & (data['vehicles_per_lane'] == data['vehicles_per_lane'].min()),
            y].sum()
        self.plot_heatmap(data, y, 'vehicles_per_lane', normalizer)

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
        data_to_plot = data.loc[(data['pollutant_id'] == pollutant_id)]
        title = 'Normalized ' + self._pollutant_id_to_string[pollutant_id]
        normalizer = data_to_plot.loc[
            (data_to_plot['control_percentages'] == 'no control')
            & (data_to_plot['vehicles_per_lane']
               == data_to_plot['vehicles_per_lane'].min()),
            y].sum()
        self.plot_heatmap(data_to_plot, 'emission_per_volume',
                          'vehicles_per_lane', normalizer, title)

    def plot_heatmap(self, data, value, rows, normalizer: int = 1,
                     title: str = None, agg_function=np.sum):
        plt.rc('font', size=17)
        if title is None:
            title = value
        table = data.pivot_table(values=value,
                                 index=[rows],
                                 columns=['control_percentages'],
                                 aggfunc=agg_function)
        table /= normalizer
        sns.heatmap(table.sort_index(axis=0, ascending=False),
                    annot=True,
                    xticklabels=table.columns.get_level_values(
                        'control_percentages'))
        plt.title(title, fontsize=22)
        plt.xlabel('Control Percentages')
        if rows == 'vehicles_per_lane':
            plt.ylabel('Input (veh/h)', fontsize=22)
        plt.tight_layout()
        if self.should_save_fig:  # TODO
            pass
        #     self.save_fig(plt.gcf(), 'heatmap',
        #                   self._pollutant_id_to_string[pollutant_id],
        #                   vehicles_per_lane,
        #                   vehicle_percentages, accepted_risks)
        plt.show()

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
        issue_count = data.groupby(['vehicles_per_lane', 'control_percentages',
                                    'accepted_risk', 'issue'])['veh_id'].count()
        issue_count /= n_simulations
        print(issue_count)

    def plot_all_pollutant_heatmaps(
            self,
            vehicle_percentages: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            accepted_risks: List[int] = None):

        for p_id in self._pollutant_id_to_string:
            self.plot_emission_heatmap(vehicle_percentages, vehicles_per_lane,
                                       accepted_risks, p_id)

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
        violations_reader = readers.ViolationsReader(self.network_name)
        data = violations_reader.load_data_with_controlled_percentage(
            vehicle_percentages, vehicles_per_lane)

        # TODO: temporary
        data.drop(index=data[data['simulation_number'] == 11].index,
                  inplace=True)

        warmup_time *= 60  # minutes to seconds
        self._prepare_data_for_plotting(data, warmup_time * 60)
        results = data.groupby(['control_percentages', 'vehicles_per_lane'],
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

    def accel_vs_time_for_different_vehicle_pairs(self):
        """
        Plots acceleration vs time for human following CAV and CAV following
        human. This requires very specific knowledge of the simulation being
        loaded. Currently this function plots a result for the traffic_lights
        scenario with 25% CACC-equipped vehicles.
        :return:
        """
        network_name = 'traffic_lights'
        vehicle_percentage = {VehicleType.TRAFFIC_LIGHT_CACC: 25}
        vehicles_per_lane = 500
        cases = [
            {'follower': 'human', 'leader': 'human', 'id': 202},
            {'follower': 'human', 'leader': 'CAV', 'id': 203},
            {'follower': 'CAV', 'leader': 'human', 'id': 201},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 152},
            {'follower': 'CAV', 'leader': 'CAV', 'id': 196},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 208},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 234},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 239}
        ]
        time = [910, 950]
        reader = readers.VehicleRecordReader(network_name)
        veh_record = reader.load_data_from_scenario(1, vehicle_percentage,
                                                    vehicles_per_lane)
        sns.set_style('whitegrid')
        # fig, axes = plt.subplots(len(cases), 1)
        # fig.set_size_inches(12, 16)
        plt.rc('font', size=20)
        full_data = []
        for i in range(len(cases)):
            case_dict = cases[i]
            veh_id = case_dict['id']
            follower_data = veh_record.loc[veh_record['veh_id'] == veh_id]
            leader_id = follower_data.iloc[0]['leader_id']
            leader_data = veh_record.loc[veh_record['veh_id'] == leader_id]
            data = pd.concat([follower_data, leader_data])
            full_data.append(data)
            sns.lineplot(data=data.loc[(data['time'] > time[0]) & (data[
                                                                       'time'] <
                                                                   time[1])],
                         x='time', y='ax', hue='veh_id',
                         palette='tab10', linewidth=3)
            plt.legend(labels=[case_dict['leader'] + ' leader',
                               case_dict['follower'] + ' follower'],
                       fontsize=22, loc='upper center',
                       bbox_to_anchor=(0.45, 1.3), ncol=2,
                       frameon=False)
            plt.xlabel('t [s]', fontsize=24)
            plt.ylabel('a(t) [m/s^2]', fontsize=24)
            plt.tight_layout()
            if self.should_save_fig:
                fig = plt.gcf()
                fig.set_dpi(600)
                fig_name = ('accel_vs_time_'
                            + case_dict['follower'] + '_follower_'
                            + case_dict['leader'] + '_leader')
                # fig.set_size_inches(12, 4)
                plt.tight_layout()
                fig.savefig(os.path.join(self._figure_folder, fig_name))
            plt.show()

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
        self.plot_risky_maneuver_histogram_per_vehicle_type(
            percentages_per_vehicle_types, veh_inputs, min_total_risk=1
        )

    # Support methods ======================================================== #
    def _load_data(self, y: str,
                   vehicle_percentages: List[Dict[VehicleType, int]],
                   vehicles_per_lane: List[int],
                   accepted_risks: List[int] = None):
        # reader = self._get_data_reader(y)
        reader = self._data_reader_map[y](self.network_name)
        # vehicle_types, percentages = list_of_dicts_to_2d_lists(
        #     percentages_per_vehicle_types)
        data = reader.load_data_with_controlled_percentage(
            vehicle_percentages, vehicles_per_lane, accepted_risks)
        return data

    @staticmethod
    def _prepare_data_collection_data(
            data: pd.DataFrame, sensor_name: str = 'in',
            warmup_time: int = 600,
            aggregation_period: int = 30) -> pd.DataFrame:
        """
        Keeps only data from one sensor, discards data before warmup time, and
        computes flow for the given aggregation period.
        """
        post_processing.create_time_in_minutes_from_intervals(data)
        data['time'] *= 60
        # Drop early samples
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)
        # Select only one sensor
        sensor_number = ResultAnalyzer._sensor_name_map[sensor_name]
        data.drop(data[data['sensor_number'] != sensor_number].index,
                  inplace=True)
        # Aggregate
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
        aggregated_data['vehicle_count'] = data['vehicle_count'].groupby(
            data.index // n).sum().to_numpy()
        aggregated_data['flow'] = (3600 / aggregation_period
                                   * aggregated_data['vehicle_count'])
        return aggregated_data

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
        percentage into a single 'control_percentages' column
        3. [Optional] Removes samples before warm-up time
        4. [Optional] Filter out certain sensor groups
        :param data: data aggregated over time
        :param warmup_time: Samples earlier than warmup_time are dropped.
         Must be passed in seconds
        :param sensor_name: if plotting flow or density, we can determine
         which sensor/lane is shown
        """
        if sensor_name is not None and not isinstance(sensor_name, list):
            sensor_name = [sensor_name]
        if 'time' not in data.columns:
            post_processing.create_time_in_minutes_from_intervals(data)
            data['time'] *= 60
        if 'flow' in data.columns:
            if sensor_name is None:
                sensor_name = ['in']
            sensor_number = [ResultAnalyzer._sensor_name_map[name] for name
                             in sensor_name]
            data.drop(data[~data['sensor_number'].isin(sensor_number)].index,
                      inplace=True)
        if (('density' in data.columns) and (sensor_name is not None)
                and (ResultAnalyzer._has_per_lane_results(data))):
            lanes = []
            [lanes.extend(ResultAnalyzer._sensor_lane_map[name])
             for name in sensor_name]
            data.drop(data[~data['lane'].isin(lanes)].index,
                      inplace=True)
        if ('pollutant_id' in data.columns) and pollutant_id is not None:
            data.drop(data[~data['pollutant_id'] == pollutant_id].index,
                      inplace=True)
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)

    def _prepare_pollutant_data(self, data: pd.DataFrame,
                                pollutant_id: int = None):
        if ('pollutant_id' in data.columns) and pollutant_id is not None:
            data.drop(data[~data['pollutant_id'] == pollutant_id].index,
                      inplace=True)

    @staticmethod
    def _create_single_control_percentage_column(data: pd.DataFrame):
        """
        Create single column with all controlled vehicles' percentages info
        """
        pass
        # percentage_strings = [col for col in data.columns
        #                       if 'percentage' in col]
        # data[percentage_strings] = data[percentage_strings].fillna(0)
        # data['control_percentages'] = ''
        # for ps in percentage_strings:
        #     idx = data[ps] > 0
        #     data[ps] = data[ps].astype('int')
        #     data.loc[idx, 'control_percentages'] += (
        #             + data.loc[idx, ps].apply(str) + '% '
        #             + ps.replace('_percentage', ''))
        # data.loc[data['control_percentages'] == '',
        #          'control_percentages'] = 'no control'
        # data['control_percentages'] = data['control_percentages'].str.replace(
        #     'autonomous', 'AV'
        # )
        # data['control_percentages'] = data['control_percentages'].str.replace(
        #     'connected', 'CAV'
        # )

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
            for percent in data['control_percentages'].unique():
                current_random_seeds = data.loc[
                    (data['control_percentages'] == percent)
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
    def _has_per_lane_results(data) -> bool:
        """
        :param data: Data from link evaluation records
        :return: boolean indicating whether the data has results individualized
        per lane
        """
        if 'lane' in data.columns and data['lane'].iloc[0] != 0:
            return True
        return False

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

    def save_fig(self, fig: plt.Figure, plot_type: str, measurement_name: str,
                 vehicles_per_lane: Union[int, List[int]],
                 vehicle_percentages: List[Dict[VehicleType, int]],
                 accepted_risk: List[int] = None
                 ):
        # Making the figure nice for inclusion in documents
        # self.widen_fig(fig, controlled_percentage)
        # plt.rc('font', size=20)
        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]
        vehicles_types, temp = list_of_dicts_to_1d_list(
            vehicle_percentages)
        veh_penetration_strings = []
        for i in range(len(vehicles_types)):
            veh_penetration_strings.append(str(temp[i]) + '_'
                                           + vehicles_types[i].name.lower())
        # vehicle_type_strings = [vt.name.lower() for vt in set(vehicles_types)]

        fig.set_dpi(200)
        axes = fig.axes
        # if plot_type != 'heatmap':
        #     for ax in axes:
        #         ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3),
        #                             useMathText=True)
        plt.tight_layout()

        fig_name = (
                plot_type + '_' + measurement_name + '_'
                + self.network_name + '_'
                + '_'.join(str(v) for v in sorted(vehicles_per_lane)) + '_'
                + 'vehs_per_lane' + '_'
                + '_'.join(sorted(veh_penetration_strings))
        )
        if accepted_risk:
            fig_name += '_risks_' + '_'.join(str(ar) for ar in accepted_risk)
        # + '_'.join(str(c) for c in controlled_percentage) + '_'
        # + '_'.join(str(vt.name).lower() for vt in
        #            self._vehicle_types))
        # plt.show()
        fig.savefig(os.path.join(self._figure_folder, fig_name))

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
        reader = readers.DataCollectionReader(self.network_name)
        raw_data = reader.load_data_with_controlled_percentage(
            vehicle_percentages, vehicle_inputs, [0])
        data = self._prepare_data_collection_data(raw_data, 'in', 0, 60)
        grouped = data.groupby(
            ['control_percentages', 'vehicles_per_lane', 'accepted_risk',
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
              issues.index[np.argmin(issue_time)]))
