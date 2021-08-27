from typing import List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import readers
from data_writer import SSMDataWriter
from post_processing import VehicleRecordPostProcessor, SSMEstimator


class ResultAnalyzer:
    units_map = {'TTC': 's', 'low_TTC': '# vehicles',
                 'DRAC': 'm/s^2', 'high_DRAC': '# vehicles',
                 'CPI': 'dimensionless', 'DTSG': 'm',
                 'exact_risk': 'm/s', 'estimated_risk': 'm/s',
                 'flow': 'veh/h', 'density': 'veh/km',
                 'time': 'min', 'time_interval': 's'}
    ssm_pretty_name_map = {'low_TTC': 'Low TTC',
                           'high_DRAC': 'High DRAC',
                           'CPI': 'CPI',
                           'exact_risk': 'CRI'}

    def __init__(self, network_name):
        self._link_evaluation_reader = readers.LinkEvaluationReader(
            network_name)
        self._data_collections_reader = readers.DataCollectionReader(
            network_name)
        self._vehicle_record_reader = readers.VehicleRecordReader(
            network_name)
        self._ssm_data_reader = readers.SSMDataReader(network_name)
        self._writer = SSMDataWriter(network_name)

    # Not sure about the title here yet: maybe move this to post processor?
    def vehicle_record_to_ssm_summary(self, autonomous_percentage: int = 0):
        """Reads multiple vehicle record data files, postprocesses them,
        computes SSMs, averages the SSMs within a time period, and produces a
        single dataframe that can be merged to results from link evaluation
        and data collection measurements.

        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation.
        :return: Dataframe with columns simulation_number, time_interval and
         one for each SSM"""

        final_file_number = (self._vehicle_record_reader.
                             find_highest_file_number(autonomous_percentage))
        if self._check_if_ssm_file_exists(final_file_number,
                                          autonomous_percentage):
            print('SSM file already exists. Returning the loaded file.\n'
                  'If you want to create a new file, delete the old one first.')
            return self._ssm_data_reader.load_data(final_file_number,
                                                   autonomous_percentage)

        column_titles = ['simulation_number', 'time_interval', 'low_TTC',
                         'high_DRAC', 'exact_risk', 'exact_risk_no_lane_change']
        result_df = pd.DataFrame(columns=column_titles)
        aggregation_period = 30  # [s] TODO: read from somewhere?
        initial_file_number = 2  # the code to run multiple scenarios
        # discards the file number 1

        for file_number in range(initial_file_number, final_file_number + 1):
            print('Working on file number {} / {}'.format(
                file_number, final_file_number + 1))

            vehicle_records = (self._vehicle_record_reader.
                               load_data(file_number, autonomous_percentage))
            post_processor = VehicleRecordPostProcessor('vissim',
                                                        vehicle_records)
            post_processor.post_process_data()
            post_processor.create_time_bins_and_labels(aggregation_period)

            # Adding SSMs
            ssm_estimator = SSMEstimator(vehicle_records)
            ssm_estimator.include_ttc()
            ssm_estimator.include_drac()
            ssm_estimator.include_collision_free_gap(consider_lane_change=False)
            ssm_estimator.include_collision_free_gap(consider_lane_change=True)
            ssm_estimator.include_exact_risk(consider_lane_change=False)
            ssm_estimator.include_exact_risk(consider_lane_change=True)

            # Aggregate
            aggregated_data = vehicle_records[column_titles[1:]].groupby(
                'time_interval').sum()
            aggregated_data.reset_index(inplace=True)
            aggregated_data.insert(0, 'simulation_number', file_number)
            result_df = result_df.append(aggregated_data)
            print('-' * 79)

        print('Dataframe built. Saving to file...')
        self._writer.save_as_csv(result_df, autonomous_percentage)
        return result_df

    # Plots aggregating results from multiple simulations =====================#
    def plot_y_vs_time(self, y: str, input_per_lane: int,
                       autonomous_percentage:
                       Union[int, List[int], str, List[str]],
                       start_time: int = None):
        """Plots averaged y over several runs with the same vehicle input 
        versus time.
        
        :param y: name of the variable being plotted.
        :param input_per_lane: input per lane used to generate the data
        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn. We expect an int, 
         but, for debugging purposes, a string with the folder name is also 
         accepted.
        :param start_time: must be given in minutes. Samples before start_time 
        are ignored."""

        if not isinstance(autonomous_percentage, list):
            autonomous_percentage = [autonomous_percentage]

        # TODO: check if input_per_lane exists in
        #  self._autonomous_percentage_simulated_inputs_map

        all_data = self._load_all_data(autonomous_percentage)
        # Create time in minutes for better display
        seconds_in_minute = 60
        all_data['time'] = all_data['time_interval'].apply(
            lambda x: int(x.split('-')[0]) / seconds_in_minute)
        relevant_data = all_data.loc[
            all_data['input_per_lane'] == input_per_lane]
        # Optional warm-up time
        if start_time is not None:
            relevant_data = relevant_data.loc[relevant_data['time']
                                              >= start_time]
        self.remove_deadlock_simulations(relevant_data)
        # Plot
        sns.set_style('whitegrid')
        n_lines = len(autonomous_percentage)
        ax = sns.lineplot(data=relevant_data, x='time', y=y,
                          hue='autonomous_percentage', ci='sd',
                          palette=sns.color_palette('deep', n_colors=n_lines)
                          )  # as_cmap = True
        ax.set_title('Input: ' + str(input_per_lane) + ' vehs per lane')
        plt.show()

    def plot_y_vs_autonomous_percentage(
            self, y: str, input_per_lane: Union[int, List[int]],
            autonomous_percentage: Union[int, List[int], str, List[str]],
            start_time: int = None):
        """Plots averaged y over several runs with the same vehicle input
        versus autonomous percentage as a box plot.

        :param y: name of the variable being plotted.
        :param input_per_lane: input per lane used to generate the data
        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn. We expect an int,
         but, for debugging purposes, a string with the folder name is also
         accepted.
        :param start_time: must be given in minutes. Samples before
         start_time are ignored."""

        if not isinstance(autonomous_percentage, list):
            autonomous_percentage = [autonomous_percentage]
        if not isinstance(input_per_lane, list):
            input_per_lane = [input_per_lane]

        # TODO: check if input_per_lane exists in
        #  self._autonomous_percentage_simulated_inputs_map

        all_data = self._load_all_data(autonomous_percentage)
        # Create time in minutes for better display
        seconds_in_minute = 60
        all_data['time'] = all_data['time_interval'].apply(
            lambda x: int(x.split('-')[0]) / seconds_in_minute)
        relevant_data = all_data.loc[
            all_data['input_per_lane'].isin(input_per_lane)]
        # Optional warm-up time
        if start_time is not None:
            relevant_data = relevant_data.loc[relevant_data['time']
                                              >= start_time]
        self.remove_deadlock_simulations(relevant_data)
        # Plot
        sns.set_style('whitegrid')
        n_lines = len(input_per_lane)
        ax = sns.boxplot(data=relevant_data, x='autonomous_percentage', y=y,
                         hue='input_per_lane',
                         palette=sns.color_palette('deep', n_colors=n_lines))
        plt.show()

    def plot_fundamental_diagram(self, autonomous_percentage: Union[int, list]):
        """Loads all measures of flow and density of a network to plot the
        fundamental diagram

        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn."""
        self.plot_with_labels(autonomous_percentage, 'density', 'flow')

    # TODO: find better name. scatter_plot_by_autonomous_percentage?
    def plot_with_labels(self, autonomous_percentage: Union[int, list],
                         x: str, y: str):
        """Loads data from the simulation(s) with the indicated autonomous
        percentage and plots y against x.

        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn.
        :param x: Options: flow, density, or any of the surrogate safety
         measures, namely, exact_risk, low_TTC, high_DRAC
        :param y: Options: flow, density, or any of
         the surrogate safety measures, namely, exact_risk, low_TTC, high_DRAC
         """

        if not isinstance(autonomous_percentage, list):
            autonomous_percentage = [autonomous_percentage]

        data = self._load_all_data(autonomous_percentage)
        self.remove_deadlock_simulations(data)
        sns.scatterplot(data=data, x=x, y=y, hue='autonomous_percentage',
                        palette=sns.color_palette(
                            'deep', n_colors=len(autonomous_percentage)))
        plt.show()

    def plot_double_y_axes(self, autonomous_percentage: int, x: str,
                           y: List[str]):
        """Loads data from the simulation with the indicated autonomous
        percentage and plots two variables against the same x axis

        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation.
        :param x: Options: flow, density, or any of the surrogate safety
         measures, namely, exact_risk, low_TTC, high_DRAC
        :param y: Must be a two-element list. Options: flow, density, or any of
         the surrogate safety measures, namely, exact_risk, low_TTC, high_DRAC
         """

        if len(y) != 2:
            print('Parameter y should be a list with two strings.')
            return

        data = self._load_all_data(autonomous_percentage)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        sns.scatterplot(data=data, ax=ax1, x=x, y=y[0], color='b')
        ax1.yaxis.label.set_color('b')
        sns.scatterplot(data=data, ax=ax2, x=x, y=y[1], color='r')
        ax2.yaxis.label.set_color('r')

        ax1.set_title(str(autonomous_percentage) + '% autonomous')
        fig.tight_layout()
        plt.show()

        return ax1, ax2

    # Support methods =========================================================#

    def _load_all_data(self, autonomous_percentage: Union[int, List[int],
                                                          str, List[str]]):
        """Loads the necessary data, merges it all under a single dataframe,
        computes the flow and returns the dataframe

        :param autonomous_percentage: Percentage of autonomous vehicles
         in the simulation. If this is a list, a single plot with
         different colors for each percentage is drawn. We expect an int,
         but, for debugging purposes, a string with the folder name is also
         accepted.
        :return: Merged dataframe with link evaluation, data collection
        results and surrogate safety measurement data"""
        seconds_in_hour = 3600
        link_evaluation_data = (
            self._link_evaluation_reader.load_data_with_autonomous_percentage(
                autonomous_percentage))
        data_collections_data = (
            self._data_collections_reader.load_data_with_autonomous_percentage(
                autonomous_percentage))
        ssm_data = self._ssm_data_reader.load_data_with_autonomous_percentage(
            autonomous_percentage)
        # We merge to be sure we're properly matching data collection results
        # and link evaluation data (same simulation, same time interval)
        full_data = link_evaluation_data.merge(
            right=data_collections_data, how='inner',
            on=['autonomous_percentage', 'input_per_lane', 'time_interval',
                'random_seed'])
        time_interval = full_data['time_interval'].iloc[0]
        interval_start, _, interval_end = time_interval.partition('-')
        measurement_period = int(interval_end) - int(interval_start)
        full_data['flow'] = (seconds_in_hour / measurement_period
                             * full_data['vehicle_count(ALL)'])
        full_data = full_data.merge(
            right=ssm_data, how='inner',
            on=['autonomous_percentage', 'input_per_lane', 'time_interval',
                'random_seed'])

        # Some column names contain (ALL). We can remove that information
        column_names = [name.split('(')[0] for name in full_data.columns]
        full_data.columns = column_names

        return full_data

    def _check_if_ssm_file_exists(self, file_identifier: int,
                                  autonomous_percentage: int = 0) -> bool:
        """We use this function to avoid repeating computations or
        overwriting files.

        :param file_identifier: File number, usually equal to the number of
         simulations run with a given autonomous percentage.
        :param autonomous_percentage: Percentage of autonomous vehicles
         present in the simulation.
        :return: True if file already exists, False otherwise"""
        try:
            df = self._ssm_data_reader.load_data(file_identifier,
                                                 autonomous_percentage)
        except ValueError:  # couldn't load file
            return False
        return not df.empty

    @staticmethod
    def remove_deadlock_simulations(data):
        deadlock_entries = (
            data.loc[data['flow'] == 0, ['input_per_lane', 'random_seed']].
                drop_duplicates().values
        )
        for element in deadlock_entries:
            idx = data.loc[(data['input_per_lane'] == element[0]) & (data[
                                                                         'random_seed'] ==
                                                                     element[
                                                                         1])].index
            data.drop(idx, inplace=True)
            print('Removed results from simulation with input {}, random '
                  'seed {} due to deadlock'.
                  format(element[0], element[1]))

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

    def plot_risk_counter(self, ssm_name, vehicle_record_data):
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
