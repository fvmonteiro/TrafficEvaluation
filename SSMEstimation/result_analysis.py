import os
from typing import List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import post_processing
import readers
from vehicle import VehicleType


class ResultAnalyzer:
    _figure_folder = 'G:\\My Drive\\Safety in Mixed Traffic\\images'
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

    def __init__(self, network_name: str,
                 vehicle_types: Union[VehicleType, List[VehicleType]]):
        if not isinstance(vehicle_types, list):
            vehicle_types = [vehicle_types]
        self.network_name = network_name
        self._vehicle_types = vehicle_types
        # link_evaluation_reader = [readers.LinkEvaluationReader(
        #     network_name, vt) for vt in vehicle_types]
        # data_collections_reader = [readers.DataCollectionReader(
        #     network_name, vt) for vt in vehicle_types]
        # ssm_data_reader = [readers.SSMDataReader(
        #     network_name, vt) for vt in vehicle_types]
        # self._data_readers = [
        #     link_evaluation_reader,
        #     data_collections_reader,
        #     ssm_data_reader
        # ]

    # Plots aggregating results from multiple simulations =====================#
    def plot_xy(self, x: str, y: str,
                controlled_vehicles_percentage: Union[int, List[int]],
                warmup_time: int = 0):
        if not isinstance(controlled_vehicles_percentage, list):
            controlled_vehicles_percentage = [controlled_vehicles_percentage]
        data = self._load_all_merged_data(controlled_vehicles_percentage)
        self._prepare_data_for_plotting(data, warmup_time)
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=data, x=x, y=y,
                          hue='control_percentages', ci='sd')
        plt.show()

    def plot_y_vs_time(self, y: str, vehicles_per_lane: int,
                       controlled_vehicles_percentage:
                       Union[int, List[int]],
                       warmup_time: int = 0, should_save_fig: bool = False):
        """Plots averaged y over several runs with the same vehicle input 
        versus time.
        
        :param y: name of the variable being plotted.
        :param vehicles_per_lane: input per lane used to generate the data
        :param controlled_vehicles_percentage: Percentage of controlled vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn. [No more] We expect an 
         int, but, for debugging purposes, a string with the folder name is also 
         accepted.
        :param warmup_time: must be given in minutes. Samples before start_time 
         are ignored.
        :param should_save_fig: determines whether to save the resulting
         figure to a file
         """

        data = self._load_all_merged_data(controlled_vehicles_percentage)
        self._prepare_data_for_plotting(data, warmup_time)
        relevant_data = self._select_relevant_data(data, vehicles_per_lane)
        # relevant_data = data.loc[
        #     data['vehicles_per_lane'] == vehicles_per_lane]
        # self.remove_deadlock_simulations(relevant_data)

        # Plot
        sns.set_style('whitegrid')
        ax = sns.lineplot(data=relevant_data, x='time', y=y,
                          hue='control_percentages', ci='sd')
        ax.set_title('Input: ' + str(vehicles_per_lane) + ' vehs per lane')
        if should_save_fig:
            self.save_fig(plt.gcf(), 'time_plot', y, [vehicles_per_lane],
                          controlled_vehicles_percentage)
        plt.show()

    def box_plot_y_vs_controlled_percentage(
            self, y: str, vehicles_per_lane: Union[int, List[int]],
            controlled_percentage: Union[int, List[int]],
            warmup_time: int = 0, should_save_fig: bool = False):
        """Plots averaged y over several runs with the same vehicle input
        versus controlled vehicles percentage as a box plot.

        :param y: name of the variable being plotted.
        :param vehicles_per_lane: input per lane used to generate the data
        :param controlled_percentage: Percentage of controlled vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn. [No more] We expect an
         int,but, for debugging purposes, a string with the folder name is also
         accepted.
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param should_save_fig: determines whether to save the resulting
         figure to a file
        """

        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]

        data = self._load_all_merged_data(controlled_percentage)
        self._prepare_data_for_plotting(data, warmup_time)
        relevant_data = self._select_relevant_data(data, vehicles_per_lane)

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
        if should_save_fig:
            self.save_fig(plt.gcf(), 'box_plot', y, vehicles_per_lane,
                          controlled_percentage)
        self.widen_fig(plt.gcf(), controlled_percentage)
        plt.tight_layout()
        plt.show()

    def box_plot_y_vs_vehicle_type(
            self, y: str, vehicles_per_lane: int,
            controlled_percentage: Union[int, List[int]],
            warmup_time: int = 0, should_save_fig: bool = False):
        """
        Plots averaged y over several runs with the same vehicle input
        versus vehicles type as a box plot. The control percentages are used
        as the boxplot hue parameter.

        :param y: name of the variable being plotted.
        :param vehicles_per_lane: input per lane used to generate the data
        :param controlled_percentage: Percentage of controlled vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn.
        :param warmup_time: must be given in minutes. Samples before
         start_time are ignored.
        :param should_save_fig: determines whether to save the resulting
         figure to a file
        """

        data = self._load_all_merged_data(controlled_percentage)
        self._prepare_data_for_plotting(data, warmup_time)
        # Adjust names for nicer looking plot
        # data['control_percentages'] = data[
        #     'control_percentages'].str.replace('% ', '%\n')
        relevant_data = self._select_relevant_data(data, vehicles_per_lane)

        # For this plot, we want the control type to be on the x axis and the
        # percentages to be the hue
        relevant_data[['percentage', 'control_type']] = relevant_data[
            'control_percentages'].str.split(' ', expand=True)
        no_control_idx = relevant_data['control_percentages'] == 'no control'
        relevant_data.loc[no_control_idx, 'control_type'] = 'autonomous'
        relevant_data.loc[no_control_idx, 'percentage'] = '0%'

        # Plot
        plt.rc('font', size=15)
        sns.set_style('whitegrid')
        sns.boxplot(data=relevant_data,  # orient='h',
                    x='control_type', y=y,
                    hue='percentage')
        if should_save_fig:
            self.save_fig(plt.gcf(), 'box_plot', y, [vehicles_per_lane],
                          controlled_percentage)
        self.widen_fig(plt.gcf(), controlled_percentage)
        plt.tight_layout()
        plt.show()

    def plot_risky_maneuver_histogram(
            self, controlled_percentage: Union[int, List[int]],
            vehicles_per_lane: Union[int, List[int]],
            warmup_time: int = 10,
            min_total_risk: float = 0):
        """
        Plots histograms of risky maneuvers' duration, total risk and max risk.

        :param controlled_percentage: Percentage of controlled vehicles
         in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn.
        :param vehicles_per_lane: input per lane used to generate the data.
         If this is a list, generates one plot per input.
        :param warmup_time: must be given in minutes. Samples before
         warmup_time are ignored.
        :param min_total_risk: risky maneuvers with total risk below this
         value are ignored.
        :return: Nothing, just plots figures
        """
        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]
        data = self._load_all_risky_maneuver_data(controlled_percentage,
                                                  vehicles_per_lane)
        warmup_time *= 60  # minutes to seconds
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)
        data.drop(index=data[data['total_risk'] < min_total_risk].index,
                  inplace=True)

        data['duration'] = data['end_time'] - data['time']
        variables_list = [#'duration',
                          'total_risk',
                          #'max_risk'
                          ]
        for veh_input in vehicles_per_lane:
            data_to_plot = self._select_relevant_data(data, veh_input)
            for variable in variables_list:
                sns.histplot(data=data_to_plot, x=variable,
                             stat='count', hue='control_percentages')
                plt.show()

    def plot_fundamental_diagram(self,
                                 controlled_percentage: Union[int, List[int]],
                                 warmup_time=0):
        """Loads all measures of flow and density of a network to plot the
        fundamental diagram

        :param controlled_percentage: Percentage of controlled vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn.
        :param warmup_time: must be given in minutes. Samples before start_time
         are ignored."""
        self.scatter_plot('density', 'flow', controlled_percentage,
                          warmup_time)

    # TODO: find better name. scatter_plot_by_controlled_percentage?
    def scatter_plot(self, x: str, y: str,
                     controlled_percentage: Union[int, list], warmup_time=0):
        """Loads data from the simulation(s) with the indicated controlled
        vehicles percentage and plots y against x.

        :param x: Options: flow, density, or any of the surrogate safety
         measures, namely, risk, low_TTC, high_DRAC
        :param y: Options: flow, density, or any of
         the surrogate safety measures, namely, risk, low_TTC, high_DRAC
        :param controlled_percentage: Percentage of controlled vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn.
        :param warmup_time: must be given in minutes. Samples before start_time
         are ignored.
         """

        data = self._load_all_merged_data(controlled_percentage)
        self._prepare_data_for_plotting(data, warmup_time)
        self.remove_deadlock_simulations(data)
        sns.scatterplot(data=data, x=x, y=y,
                        hue='control_percentages')
        plt.show()

    def plot_double_y_axes(self, controlled_percentage: int, x: str,
                           y: List[str], warmup_time=0):
        """Loads data from the simulation with the indicated controlled vehicles
        percentage and plots two variables against the same x axis

        :param controlled_percentage: Percentage of controlled vehicles
         present in the simulation.
        :param x: Options: flow, density, or any of the surrogate safety
         measures, namely, risk, low_TTC, high_DRAC
        :param y: Must be a two-element list. Options: flow, density, or any of
         the surrogate safety measures, namely, risk, low_TTC, high_DRAC
        :param warmup_time: must be given in minutes. Samples before start_time
         are ignored.
         """

        if len(self._vehicle_types):
            raise ValueError('This function does not work when class member '
                             '_vehicle_type has more than one element.')
        if len(y) != 2:
            raise ValueError('Parameter y should be a list with two strings.')

        data = self._load_all_merged_data(controlled_percentage)
        self._prepare_data_for_plotting(data, warmup_time)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        sns.scatterplot(data=data, ax=ax1, x=x, y=y[0], color='b')
        ax1.yaxis.label.set_color('b')
        sns.scatterplot(data=data, ax=ax2, x=x, y=y[1], color='r')
        ax2.yaxis.label.set_color('r')

        ax1.set_title(
            str(controlled_percentage) + '% '
            + self._vehicle_types[0].name.lower())
        fig.tight_layout()
        plt.show()

        return ax1, ax2

    def speed_color_map(self, vehicles_per_lane: int,
                        controlled_percentage: Union[int, List[int]]):
        """

        :param vehicles_per_lane:
        :param controlled_percentage:
        :return:
        """
        if isinstance(controlled_percentage, list):
            percentages = np.sort(controlled_percentage[:])
        else:
            percentages = np.sort([controlled_percentage])

        for vehicle_type in self._vehicle_types:
            for p in percentages:
                veh_record_reader = readers.VehicleRecordReader(
                    self.network_name, vehicle_type)
                min_file, max_file = veh_record_reader.find_min_max_file_number(
                    p, vehicles_per_lane)
                # TODO: for now just one file. We'll see later if aggregating
                #  makes sense
                veh_record = veh_record_reader.load_data(max_file, p,
                                                         vehicles_per_lane)
                # Get only main segment. TODO: this must change for other scenarios
                veh_record.drop(
                    index=veh_record[(veh_record['link'] != 3)
                                     | (veh_record['time'] < 300)].index,
                    inplace=True)
                veh_record['time [s]'] = veh_record['time'] // 10
                space_bins = [i for i in range(0, int(veh_record['x'].max()), 25)]
                veh_record['x [m]'] = pd.cut(veh_record['x'], bins=space_bins,
                                                  labels=space_bins[:-1])
                plotted_data = veh_record.groupby(['time [s]', 'x [m]'],
                                                  as_index=False)['vx'].mean()
                plotted_data = plotted_data.pivot('time [s]', 'x [m]', 'vx')
                ax = sns.heatmap(plotted_data)
                ax.invert_yaxis()
                plt.show()
            # We only plot 0 percentage once
            if 0 in percentages:
                percentages = np.delete(percentages, 0)

    # Support methods =========================================================#

    def _load_all_merged_data(
            self,
            controlled_vehicles_percentage: Union[int, List[int]]):
        """Loads the necessary data, merges it all under a single dataframe,
        computes the flow and returns the dataframe

        :param controlled_vehicles_percentage: Percentage of controlled
         vehicles in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn. [No more] We expect an
         int, but, for debugging purposes, a string with the folder name is also
         accepted.
        :return: Merged dataframe with link evaluation, data collection
        results and surrogate safety measurement data"""

        if isinstance(controlled_vehicles_percentage, list):
            percentage_copy = controlled_vehicles_percentage[:]
        else:
            percentage_copy = [controlled_vehicles_percentage]
        data = pd.DataFrame()
        for vt in self._vehicle_types:
            data_reader = readers.MergedDataReader(self.network_name, vt)
            new_data = data_reader.load_multiple_data(
                percentage_copy)
            data = new_data if data.empty else data.append(new_data)
            # We only need to load data without any controlled vehicles once
            if 0 in percentage_copy:
                percentage_copy.remove(0)
        return data.reset_index(drop=True)

    def _prepare_data_for_plotting(self, data: pd.DataFrame,
                                   warmup_time: float = 0,
                                   sensor_numbers: Union[int, List[int]] = 1):
        """Performs several operations to make the data proper for plotting:
        1. Fill NaN entries in columns describing controlled vehicle
        percentage
        2. Aggregates data from all columns describing controlled vehicle
        percentage into a single 'control_percentages' column
        3. [Optional] Removes samples before warm-up time
        4. [Optional] Filter out certain sensor groups

        :param data: Dataframe with data from all sources (link evaluation,
         data collection, and ssm)
        :param warmup_time: Samples earlier than warmup_time are dropped
        :param sensor_numbers: We only keep data collection measurements from
         the listed sensor numbers"""

        if sensor_numbers:
            if not isinstance(sensor_numbers, list):
                sensor_numbers = [sensor_numbers]
            data.drop(index=data[~data['sensor_number'].isin(
                sensor_numbers)].index,
                      inplace=True)

        # Fill NaNs in cols of *_percentage with zero
        percentage_strings = [col for col in data.columns
                              if 'percentage' in col]
        data[percentage_strings] = data[percentage_strings].fillna(0)

        # Create single columns with all controlled vehicles' percentages info
        data['control_percentages'] = ''
        for vt in self._vehicle_types:
            vt_str = vt.name.lower() + '_percentage'
            idx = data[vt_str] > 0
            data.loc[idx, 'control_percentages'] += (
                    data.loc[idx, vt_str].apply(int).apply(str) + '% ' +
                    vt.name.lower())
        data.loc[data['control_percentages'] == '',
                 'control_percentages'] = 'no control'

        # Optional warm-up time.
        data.drop(index=data[data['time'] < warmup_time].index, inplace=True)

    def _load_all_risky_maneuver_data(
            self, controlled_percentage: Union[int, List[int]],
            vehicles_per_lane: Union[int, List[int]]):
        """
        Goes into the folder of each selected vehicle type, loads and merges
        all the requested data.

        :param controlled_percentage: Percentage of controlled vehicles
         in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn.
        :param vehicles_per_lane: input per lane used to generate the data.
         If this is a list, generates one plot per input. TODO: not necessary?
        :return:
        """

        if not isinstance(controlled_percentage, list):
            controlled_percentage = [controlled_percentage]
        data_per_veh_type = []
        for vt in self._vehicle_types:
            reader = readers.RiskyManeuverReader(self.network_name, vt)
            data_per_percentage = []
            for percentage in controlled_percentage:
                data_per_percentage.append(
                    reader.load_data_with_controlled_vehicles_percentage(
                        percentage))
            data_per_veh_type.append(pd.concat(data_per_percentage,
                                               ignore_index=True))
        data = pd.concat(data_per_veh_type)

        # Create single columns with all controlled vehicles' percentages info
        data['control_percentages'] = ''
        for vt in self._vehicle_types:
            vt_str = vt.name.lower() + '_percentage'
            idx = data[vt_str] > 0
            data.loc[idx, 'control_percentages'] += (
                    data.loc[idx, vt_str].apply(int).apply(str) + '% ' +
                    vt.name.lower())
        data.loc[data['control_percentages'] == '',
                 'control_percentages'] = 'no control'

        return data

    @staticmethod
    def _select_relevant_data(data: pd.DataFrame,
                              vehicles_per_lane: Union[int, List[int]]):
        """
        Selects only the desired vehicle inputs and uniformity of data from
        each different controlled vehicle percentage. In other words,
        only keeps the result of a certain random seed if that random seed
        was used by all controlled vehicle percentages
        :param data:
        :param vehicles_per_lane:
        :return:
        """
        if not isinstance(vehicles_per_lane, list):
            vehicles_per_lane = [vehicles_per_lane]

        filtered_data = data.loc[data['vehicles_per_lane'].isin(
            vehicles_per_lane)]

        # Get the intersection of random seeds for each control percentage
        random_seeds = np.array([])
        for veh_input in vehicles_per_lane:
            for percent in data['control_percentages'].unique():
                current_random_seeds = filtered_data.loc[
                        (filtered_data['control_percentages'] == percent)
                        & (filtered_data['vehicles_per_lane'] == veh_input),
                        'random_seed'].unique()
                if random_seeds.size == 0:
                    random_seeds = current_random_seeds
                else:
                    random_seeds = np.intersect1d(random_seeds,
                                                  current_random_seeds)

        # Keep only the random_seeds used by all control percentages
        relevant_data = filtered_data.drop(index=filtered_data[~filtered_data[
            'random_seed'].isin(random_seeds)].index)
        if relevant_data.shape[0] != filtered_data.shape[0]:
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
                 vehicles_per_lane: List[int],
                 controlled_percentage: List[int]):
        # Making the figure nice for inclusion in documents
        self.widen_fig(fig, controlled_percentage)
        fig.set_dpi(200)
        axes = fig.axes
        for ax in axes:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3),
                                useMathText=True)
        #     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        #                  ax.get_yticklabels() + ax.get_legend().get_texts()):
        #         item.set_fontsize(15)
        #     for item in ax.get_xticklabels():
        #         item.set_fontsize(15)
        # plt.rcParams.update({'font.size': 13})
        # plt.rcParams.update({'xtick.labelsize': 12})
        plt.tight_layout()
        fig_name = (plot_type + '_' + measurement_name + '_'
                    + '_'.join(str(v) for v in vehicles_per_lane) + '_'
                    + 'vehs_per_lane' + '_'
                    + '_'.join(str(c) for c in controlled_percentage) + '_'
                    + '_'.join(str(vt.name).lower() for vt in
                               self._vehicle_types))
        # plt.show()
        fig.savefig(os.path.join(self._figure_folder, fig_name))

    def widen_fig(self, fig: plt.Figure, controlled_percentage: List[int]):
        if (len(self._vehicle_types) >= 3) and (len(controlled_percentage) > 1):
            fig.set_size_inches(6.4*2, 4.8)

    # Plots for a single simulation - OUTDATED: might not work ================#

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

    # Data integrity checks ===================================================#
    def find_removed_vehicles(self, vehicle_type: VehicleType,
                              controlled_vehicle_percentage: int):
        """Checks whether VISSIM removed any vehicles for standing still too
        long.
        Results obtained so far:
        - in_and_out with no controlled vehicles and highest inputs (2000 and
        2500) had no vehicles removed. """
        veh_rec_reader = readers.VehicleRecordReader(self.network_name,
                                                     vehicle_type)
        _, max_file_number = veh_rec_reader.find_min_max_file_number(
            controlled_vehicle_percentage)
        exit_links = [5, 6]
        for file_number in range(max_file_number - 10 + 1, max_file_number + 1):
            print('file number=', file_number)
            vehicle_record = veh_rec_reader.load_data(file_number, 0)
            max_time = vehicle_record.iloc[-1]['time']
            all_ids = vehicle_record['veh_id'].unique()
            removed_vehs = []
            for veh_id in all_ids:
                veh_data = vehicle_record.loc[vehicle_record['veh_id']
                                              == veh_id]
                veh_last_link = veh_data['link'].iloc[-1]
                veh_last_time = veh_data['time'].iloc[-1]
                if veh_last_link not in exit_links and veh_last_time < max_time:
                    removed_vehs.append(veh_id)
            if len(removed_vehs) < 10:
                print('Removed vehs: ', removed_vehs)
            else:
                print(len(removed_vehs), ' removed vehs.')

    def find_unfinished_simulations(self, percentage):
        """ Checks whether simulations crashed. This is necessary because,
        when doing multiple runs from the COM interface, VISSIM does not
        always indicate that a simulation crashed. """

        # We must check either SSM or merged results
        data_collection_readers = [readers.SSMDataReader(
            self.network_name, vt) for vt in self._vehicle_types]
        data_no_control = (data_collection_readers[0].
                           load_data_with_controlled_vehicles_percentage(0))
        end_time = data_no_control.iloc[-1]['time_interval']
        for reader in data_collection_readers:
            print(reader.vehicle_type)
            data = reader.load_data_with_controlled_vehicles_percentage(
                percentage)
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
                        print('Simulation with input {} random seed {} '
                              'stopped at {}'.format(
                                   veh_input, random_seed,
                                   sim_data.iloc[-1]['time_interval']))
