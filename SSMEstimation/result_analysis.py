import os
from typing import List, Dict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import post_processing
import readers
from vehicle import VehicleType


def list_of_dicts_to_2d_lists(dict_list: List[Dict]):
    keys = []
    values = []
    for d in dict_list:
        keys.append(list(d.keys()))
        values.append(list(d.values()))
    return keys, values


def list_of_dicts_to_1d_list(dict_list: List[Dict]):
    # Note: slow solution because of the summing of lists, but shouldn't
    #  impact performance
    keys = []
    values = []
    for d in dict_list:
        keys += list(d.keys())
        values += list(d.values())
    return keys, values


class ResultAnalyzer:
    units_map = {'TTC': 's', 'low_TTC': '# vehicles',
                 'DRAC': 'm/s^2', 'high_DRAC': '# vehicles',
                 'CPI': 'dimensionless', 'DTSG': 'm',
                 'risk': 'm/s', 'estimated_risk': 'm/s',
                 'flow': 'veh/h', 'density': 'veh/km',
                 'time': 'min', 'time_interval': 's'}
    ssm_pretty_name_map = {'low_TTC': 'Low TTC',
                           'high_DRAC': 'High DRAC',
                           'CPI': 'CPI',
                           'risk': 'CRI'}
    data_reader_map = {
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
        'initial_risk_to_fd': readers.LaneChangeReader
    }

    def __init__(self, network_name: str, should_save_fig: bool = False):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-626HHGI':
            self._figure_folder = ('C:\\Users\\fvall\\Google Drive\\Safety in '
                                   'Mixed Traffic\\images')
        else:
            self._figure_folder = ('G:\\My Drive\\Safety in Mixed Traffic'
                                   '\\images')
        self.network_name = network_name
        self.should_save_fig = should_save_fig

    # Plots aggregating results from multiple simulations ==================== #
    def plot_xy(self, x: str, y: str,
                percentages_per_vehicle_types: List[Dict[VehicleType, int]],
                vehicles_per_lane: int, accepted_risks: List[int] = None,
                warmup_time: int = 0):

        if not isinstance(percentages_per_vehicle_types, list):
            percentages_per_vehicle_types = [percentages_per_vehicle_types]
        data = self._load_data(y, percentages_per_vehicle_types,
                               [vehicles_per_lane], accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time)
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=data, x=x, y=y,
                          hue='control_percentages', ci='sd')
        plt.show()

    def plot_y_vs_time(self, y: str,
                       percentages_per_vehicle_types: List[
                           Dict[VehicleType, int]],
                       vehicles_per_lane: int,
                       # controlled_percentage: Union[int, List[int]],
                       warmup_time: int = 0):
        """Plots averaged y over several runs with the same vehicle input 
        versus time.
        
        :param y: name of the variable being plotted.
        :param percentages_per_vehicle_types: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param vehicles_per_lane: input per lane used to generate the data
        :param warmup_time: must be given in minutes. Samples before start_time 
         are ignored.
         """

        data = self._load_data(y, percentages_per_vehicle_types,
                               [vehicles_per_lane])
        self._prepare_data_for_plotting(data, warmup_time)
        relevant_data = self._ensure_data_source_is_uniform(data,
                                                            [vehicles_per_lane])
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=relevant_data, x='time', y=y,
                          hue='control_percentages', ci='sd')
        ax.set_title('Input: ' + str(vehicles_per_lane) + ' vehs per lane')
        if self.should_save_fig:
            self.save_fig(plt.gcf(), 'time_plot', y, [vehicles_per_lane],
                          percentages_per_vehicle_types)
        plt.show()

    def box_plot_y_vs_controlled_percentage(
            self, y: str, vehicles_per_lane: Union[int, List[int]],
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            warmup_time: int = 0, flow_sensor_number: int = 1):
        """Plots averaged y over several runs with the same vehicle input
        versus controlled vehicles percentage as a box plot.

        :param y: name of the variable being plotted.
        :param vehicles_per_lane: input per lane used to generate the data
        :param percentages_per_vehicle_types: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param flow_sensor_number: if plotting flow, we can determine choose
         which data collection measurement will be shown
        """

        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]
        data = self._load_data(y, percentages_per_vehicle_types,
                               vehicles_per_lane)
        self._prepare_data_for_plotting(data, warmup_time, flow_sensor_number)
        relevant_data = self._ensure_data_source_is_uniform(data,
                                                            vehicles_per_lane)

        # Plot
        plt.rc('font', size=15)
        sns.set_style('whitegrid')
        if len(vehicles_per_lane) > 1:
            sns.boxplot(data=relevant_data,  # orient='h',
                        x='control_percentages', y=y,
                        hue='vehicles_per_lane')
        else:
            sns.boxplot(data=relevant_data,  # orient='h',
                        x='control_percentages', y=y)
        if self.should_save_fig:
            self.save_fig(plt.gcf(), 'box_plot', y, vehicles_per_lane,
                          percentages_per_vehicle_types)
        # self.widen_fig(plt.gcf(), controlled_percentage)
        plt.tight_layout()
        plt.show()

        # Direct output
        for veh_input in vehicles_per_lane:
            print('Veh input: ', veh_input)
            veh_input_idx = relevant_data['vehicles_per_lane'] == veh_input
            all_control_percentages = relevant_data[
                'control_percentages'].unique()
            for control_percentage in all_control_percentages:
                control_percentage_idx = (relevant_data['control_percentages']
                                          == control_percentage)
                print('{}, Median {}: {}'.
                      format(control_percentage, y,
                             relevant_data.loc[(veh_input_idx
                                                & control_percentage_idx),
                                               y].median()))

    def box_plot_y_vs_vehicle_type(
            self, y: str, hue: str, vehicles_per_lane: int,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            accepted_risks: List[int] = None,
            warmup_time: int = 10, kind: str = 'box'):
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
        :param kind: set this parameter to 'violin' or 'boxen' for different
         types of plots
        """
        data = self._load_data(y, percentages_per_vehicle_types,
                               [vehicles_per_lane], accepted_risks)
        self._prepare_data_for_plotting(data, warmup_time)
        relevant_data = self._ensure_data_source_is_uniform(
            data, [vehicles_per_lane])

        no_control_idx = (relevant_data['control_percentages']
                          == 'no control')
        relevant_data[['percentage', 'control_type']] = relevant_data[
            'control_percentages'].str.split(' ', expand=True)
        relevant_data.loc[no_control_idx, 'control_type'] = 'human'
        relevant_data.loc[no_control_idx, 'percentage'] = '0%'

        # Plot
        plt.rc('font', size=15)
        sns.set_style('whitegrid')
        sns.boxplot(data=relevant_data, x='control_type', y=y, hue=hue)
                    # kind=kind)
        # legend_len = len(relevant_data[hue].unique())
        # n_legend_cols = (legend_len if legend_len < 5
        #                  else legend_len // 2 + 1)
        plt.legend(title=hue, loc='right', ncol=1,
                   bbox_to_anchor=(1.1, 1))
        plt.title(str(vehicles_per_lane) + ' vehs/h/lane', fontsize=22)
        if self.should_save_fig:
            self.save_fig(plt.gcf(), 'box_plot', y, [vehicles_per_lane],
                          percentages_per_vehicle_types)
        self.widen_fig(plt.gcf(), len(percentages_per_vehicle_types))
        plt.tight_layout()
        plt.show()

    def plot_risky_maneuver_histogram_per_vehicle_type(
            self, percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 10, min_total_risk: float = 1):
        """
        Plots histograms of risky maneuvers' total risk.

        :param percentages_per_vehicle_types: each dictionary in the list must
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

        vehicle_types, percentages = list_of_dicts_to_2d_lists(
            percentages_per_vehicle_types)
        risky_maneuver_reader = readers.RiskyManeuverReader(self.network_name)
        data = risky_maneuver_reader.load_data_with_controlled_percentage(
            vehicle_types, percentages, vehicles_per_lane)
        warmup_time *= 60  # minutes to seconds
        data.drop(index=data[data['total_risk'] < min_total_risk].index,
                  inplace=True)
        self._prepare_data_for_plotting(data, warmup_time)

        # data['duration'] = data['end_time'] - data['time']
        # variables_list = [  # 'duration',
        #     'total_risk',
        #     # 'max_risk'
        # ]
        n_simulations = 10
        for control_percentage in data['control_percentages'].unique():
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
                            + control_percentage.replace('% ', '_'))
                fig.savefig(os.path.join(self._figure_folder, fig_name))
            plt.show()

            # Direct output:
            for veh_input in vehicles_per_lane:
                veh_input_idx = data_to_plot['vehicles_per_lane'] == veh_input
                mean_count = np.count_nonzero(veh_input_idx) / n_simulations
                mean_simulation_risk = (data_to_plot.loc[veh_input_idx,
                                                         'total_risk'].sum()
                                        / n_simulations)
                print('{}, {} vehicles per lane:\n'
                      '\tmean # risky intervals: {}\n'
                      '\tmean simulation risk: {}\n'
                      .format(control_percentage, veh_input,
                              mean_count, mean_simulation_risk))

    def plot_grid_of_risk_histograms(
            self, risk_type: str,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, min_risk: int = 0.01
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
        self._prepare_data_for_plotting(data, warmup_time)
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
                if i == 0:
                    axes[i, j].set_title('accepted risk = ' + str(ar))
        plt.tight_layout()
        plt.show()

    def hist_plot_lane_change_initial_risks(
            self, percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 0):
        """

        :param percentages_per_vehicle_types:
        :param vehicles_per_lane:
        :param warmup_time:
        :return:
        """
        lane_change_reader = readers.LaneChangeReader(self.network_name)
        if percentages_per_vehicle_types is None and vehicles_per_lane is None:
            data = lane_change_reader.load_test_data()
        else:
            vehicle_types, percentages = list_of_dicts_to_2d_lists(
                percentages_per_vehicle_types)
            data = lane_change_reader.load_data_with_controlled_percentage(
                vehicle_types, percentages, vehicles_per_lane)
        warmup_time *= 60  # minutes to seconds
        self._create_single_control_percentage_column(data)
        data.drop(index=data[data['start_time'] < warmup_time].index,
                  inplace=True)

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

    def plot_heatmap(
            self, y: str,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int], accepted_risks: List[int],
            warmup_time: int = 10, normalize: bool = False
    ):
        """

        :param y:
        :param percentages_per_vehicle_types:
        :param vehicles_per_lane:
        :param accepted_risks:
        :param warmup_time:
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
            data = self._load_data(y, percentages_per_vehicle_types, [vpl],
                                   accepted_risks)
            self._prepare_data_for_plotting(data, warmup_time)
            data.loc[data['control_percentages'] == 'no control',
                     'control_percentages'] = '100%HD'
            n_simulations = len(data['simulation_number'].unique())
            table = data.pivot_table(values=col_in_df,
                                     index=['accepted_risk'],
                                     columns=['control_percentages'],
                                     aggfunc=aggregation_function)
            if 'count' in y:
                table /= n_simulations
            no_avs_no_risk_value = table.loc[0, '100%HD']
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
                        # vmin=0, vmax=max_table_value,
                        xticklabels=table.columns.get_level_values(
                            'control_percentages'))
            plt.xlabel('', fontsize=22)
            plt.ylabel('accepted initial risk', fontsize=22)
            plt.title(title + ' at ' + str(vpl) + ' vehs/h/lane',
                      fontsize=22)
            plt.tight_layout()
            plt.show()

    # Traffic Light Scenario Plots =========================================== #
    def plot_violations_per_control_percentage(
            self, percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 10):
        """
        Plots number of violations .

        :param percentages_per_vehicle_types: each dictionary in the list must
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
        vehicle_types, percentages = list_of_dicts_to_2d_lists(
            percentages_per_vehicle_types)
        violations_reader = readers.ViolationsReader(self.network_name)
        data = violations_reader.load_data_with_controlled_percentage(
            vehicle_types, percentages, vehicles_per_lane)

        # TODO: temporary
        data.drop(index=data[data['simulation_number'] == 11].index,
                  inplace=True)

        warmup_time *= 60  # minutes to seconds
        self._prepare_data_for_plotting(data, warmup_time)
        results = data.groupby(['control_percentages', 'vehicles_per_lane'],
                               as_index=False)['veh_id'].count().rename(
            columns={'veh_id': 'violations count'})
        results['mean violations'] = results['violations count'] / n_simulations
        print(results)

    def plot_heatmap_for_traffic_light_scenario(
            self, y: str,
            percentages_per_vehicle_types: List[Dict[VehicleType, int]],
            vehicles_per_lane: List[int],
            warmup_time: int = 10):
        """
        Plots a heatmap
        :param y:
        :param percentages_per_vehicle_types:
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
            data = self._load_data(y, percentages_per_vehicle_types, [vpl])
            self._prepare_data_for_plotting(data, warmup_time)
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
                              percentages_per_vehicle_types)
            plt.tight_layout()
            plt.show()

    def plot_violations_heatmap(self,
                                percentages_per_vehicle_types: List[
                                    Dict[VehicleType, int]],
                                vehicles_per_lane: List[int],
                                warmup_time: int = 10):
        n_simulations = 10
        plt.rc('font', size=17)
        for vpl in vehicles_per_lane:
            data = self._load_data('violations', percentages_per_vehicle_types,
                                   [vpl])
            self._prepare_data_for_plotting(data, warmup_time)
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
                              percentages_per_vehicle_types)
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
        vehicle_type = [VehicleType.TRAFFIC_LIGHT_CACC]
        percentage = [25]
        vehicles_per_lane = 500
        cases = [
            {'follower': 'human', 'leader': 'human', 'id': 202},
            {'follower': 'human', 'leader': 'CAV', 'id': 203},
            {'follower': 'CAV', 'leader': 'human', 'id': 201},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 152},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 196},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 208},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 234},
            # {'follower': 'CAV', 'leader': 'CAV', 'id': 239}
        ]
        time = [910, 950]
        reader = readers.VehicleRecordReader(network_name)
        veh_record = reader.load_data(1, vehicle_type, percentage,
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
        # full_data = pd.concat(full_data)

    def plot_double_y_axes(self,  # controlled_percentage: int,
                           percentages_per_vehicle_types: List[Dict[
                               VehicleType, int]],
                           x: str,
                           y: List[str], warmup_time=0):
        """Loads data from the simulation with the indicated controlled vehicles
        percentage and plots two variables against the same x axis

        :param percentages_per_vehicle_types: each dictionary in the list must
         define the percentage of different vehicles type in the simulation
        :param x: Options: flow, density, or any of the surrogate safety
         measures, namely, risk, low_TTC, high_DRAC
        :param y: Must be a two-element list. Options: flow, density, or any of
         the surrogate safety measures, namely, risk, low_TTC, high_DRAC
        :param warmup_time: must be given in minutes. Samples before start_time
         are ignored.
         """
        raise NotImplementedError
        # if len(percentages_per_vehicle_types) > 1:
        #     raise ValueError('This function does not work when class member '
        #                      '_vehicle_type has more than one element.')
        # if len(y) != 2:
        #     raise ValueError('Parameter y should be a list with two strings.')
        #
        # data = self._load_all_merged_data(percentages_per_vehicle_types)
        # self._prepare_data_for_plotting(data, warmup_time)
        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        #
        # sns.scatterplot(data=data, ax=ax1, x=x, y=y[0], color='b')
        # ax1.yaxis.label.set_color('b')
        # sns.scatterplot(data=data, ax=ax2, x=x, y=y[1], color='r')
        # ax2.yaxis.label.set_color('r')
        #
        # ax1.set_title(
        #     str(controlled_percentage) + '% '
        #     + self._vehicle_types[0].name.lower())
        # fig.tight_layout()
        # plt.show()
        #
        # return ax1, ax2

    def speed_color_map(self, vehicles_per_lane: int,
                        percentages_per_vehicle_types: List[
                            Dict[VehicleType, int]]):
        # controlled_percentage: Union[int, List[int]]):
        """

        :param vehicles_per_lane:
       :param percentages_per_vehicle_types: each dictionary in the list must
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
        #         # TODO: for now just one file. We'll see later if aggregating
        #         #  makes sense
        #         veh_record = veh_record_reader.load_data(max_file,
        #                                                  vehicle_type, p,
        #                                                  vehicles_per_lane)
        #         # Get only main segment.
        #         # TODO: this must change for other scenarios
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
        self.box_plot_y_vs_controlled_percentage(
            'risk', veh_inputs, percentages_per_vehicle_types, warmup_time=10
        )
        self.plot_risky_maneuver_histogram_per_vehicle_type(
            percentages_per_vehicle_types, veh_inputs, min_total_risk=1
        )

    # Support methods ======================================================== #
    def _load_data(self, y: str,
                   percentages_per_vehicle_types: List[Dict[VehicleType, int]],
                   vehicles_per_lane: List[int],
                   accepted_risks: List[int] = None):
        # reader = self._get_data_reader(y)
        reader = self.data_reader_map[y](self.network_name)
        vehicle_types, percentages = list_of_dicts_to_2d_lists(
            percentages_per_vehicle_types)
        data = reader.load_data_with_controlled_percentage(
            vehicle_types, percentages, vehicles_per_lane, accepted_risks)
        return data

    def _prepare_data_for_plotting(self, data: pd.DataFrame,
                                   warmup_time: float = 0,
                                   flow_sensor_number: int = 1):
        """Performs several operations to make the data proper for plotting:
        1. Fill NaN entries in columns describing controlled vehicle
        percentage
        2. Aggregates data from all columns describing controlled vehicle
        percentage into a single 'control_percentages' column
        3. [Optional] Removes samples before warm-up time
        4. [Optional] Filter out certain sensor groups

        :param data: data aggregated over time
        :param warmup_time: Samples earlier than warmup_time are dropped
        :param flow_sensor_number: if plotting flow, we can determine choose
         which data collection measurement will be shown
        """
        self._create_single_control_percentage_column(data)
        if 'time' not in data.columns:
            post_processing.create_time_in_minutes_from_intervals(data)
        if 'flow' in data.columns:
            data.drop(data[data['sensor_number'] != flow_sensor_number].index,
                      inplace=True)
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)

    def _create_single_control_percentage_column(self, data: pd.DataFrame):
        """
        Create single column with all controlled vehicles' percentages info
        """
        percentage_strings = [col for col in data.columns
                              if 'percentage' in col]
        data[percentage_strings] = data[percentage_strings].fillna(0)
        data['control_percentages'] = ''
        for ps in percentage_strings:
            idx = data[ps] > 0
            data[ps] = data[ps].astype('int')
            data.loc[idx, 'control_percentages'] += (
                    + data.loc[idx, ps].apply(str) + '% '
                    + ps.replace('_percentage', ''))
        data.loc[data['control_percentages'] == '',
                 'control_percentages'] = 'no control'
        data['control_percentages'] = data['control_percentages'].str.replace(
            'autonomous', 'AV'
        )
        data['control_percentages'] = data['control_percentages'].str.replace(
            'connected', 'CAV'
        )

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
                 percentages_per_vehicle_types: List[
                     Dict[VehicleType, int]],
                 # controlled_percentage: Union[str, List[str]]
                 ):
        # Making the figure nice for inclusion in documents
        # self.widen_fig(fig, controlled_percentage)
        # plt.rc('font', size=20)
        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]
        # if not isinstance(controlled_percentage, list):
        #     controlled_percentage = [controlled_percentage]
        vehicles_types, _ = list_of_dicts_to_1d_list(
            percentages_per_vehicle_types)

        fig.set_dpi(200)
        axes = fig.axes
        if plot_type != 'heatmap':
            for ax in axes:
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3),
                                    useMathText=True)
        plt.tight_layout()
        vehicle_type_strings = [vt.name.lower() for vt in set(vehicles_types)]
        fig_name = (
                plot_type + '_' + measurement_name + '_'
                + self.network_name + '_'
                + '_'.join(str(v) for v in sorted(vehicles_per_lane)) + '_'
                + 'vehs_per_lane' + '_'
                + '_'.join(sorted(vehicle_type_strings))
        )
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
            if ssm in self.ssm_pretty_name_map:
                ssm_plot_name = self.ssm_pretty_name_map[ssm]
            else:
                ssm_plot_name = ssm
            ax.plot(aggregated_data[ssm].index, aggregated_data[ssm][ssm],
                    label=ssm_plot_name + ' (' + self.units_map[ssm] + ')')
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
        if ssm_name in self.ssm_pretty_name_map:
            ssm_plot_name = self.ssm_pretty_name_map[ssm_name]
        else:
            ssm_plot_name = ssm_name
        ax.set_ylabel(ssm_plot_name + ' (' + self.units_map[ssm_name] + ')',
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

    def find_unfinished_simulations(self,
                                    vehicle_types: List[List[VehicleType]],
                                    percentage: List[List[int]],
                                    vehicle_inputs):
        """ Checks whether simulations crashed. This is necessary because,
        when doing multiple runs from the COM interface, VISSIM does not
        always indicate that a simulation crashed. """

        # We must check either SSM results
        reader = readers.SSMDataReader(self.network_name)
        data_no_control = (reader.load_data_with_controlled_percentage(
            vehicle_types, [[0]], vehicle_inputs))
        end_time = data_no_control.iloc[-1]['time_interval']
        for vt in vehicle_types:  # reader in data_collection_readers:
            # print(vt.name)
            data = reader.load_data_with_controlled_percentage(
                [vt], percentage, vehicle_inputs)
            all_random_seeds = data['random_seed'].unique()
            all_inputs = data['vehicles_per_lane'].unique()
            for veh_input in all_inputs:
                # print('veh input=', veh_input)
                for random_seed in all_random_seeds:
                    # print('random seed=', random_seed)
                    sim_data = data.loc[
                        (data['random_seed'] == random_seed)
                        & (data['vehicles_per_lane'] == veh_input)]
                    if sim_data.iloc[-1]['time_interval'] != end_time:
                        last_time = sim_data.iloc[-1]['time_interval']
                        print('Simulation with input {} random seed {} '
                              'stopped at {}'.format(veh_input, random_seed,
                                                     last_time))
