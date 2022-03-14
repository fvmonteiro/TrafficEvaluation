import warnings
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import data_writer
import readers
from vehicle import Vehicle, VehicleType


def create_time_in_minutes(data: pd.DataFrame):
    """ Creates a 'time' column in minutes

    :param data: Dataframe with data from all sources (link evaluation,
     data collection, and ssm)
    """
    # Create time in minutes for better display
    seconds_in_minute = 60
    data['time'] = data['time_interval'].apply(
        lambda x: int(x.split('-')[0]) / seconds_in_minute)


def compute_flow(data: pd.DataFrame, sensor_numbers: Union[int, List[int]] = 1):
    """ Computes flow in vehicles/hour and includes it in the dataframe

    :param data: Dataframe containing a vehicle_count column
    :param sensor_numbers: We only keep data collection measurements from
     the listed sensor numbers
    """
    if sensor_numbers:
        if not isinstance(sensor_numbers, list):
            sensor_numbers = [sensor_numbers]
        data.drop(index=data[~data['sensor_number'].isin(
            sensor_numbers)].index,
                  inplace=True)
    # Compute flow
    time_interval = data['time_interval'].iloc[0]
    interval_start, _, interval_end = time_interval.partition('-')
    measurement_period = int(interval_end) - int(interval_start)
    seconds_in_hour = 3600
    data['flow'] = (seconds_in_hour / measurement_period
                    * data['vehicle_count'])


def create_ssm_dataframe(vehicle_record: pd.DataFrame,
                         ssm_names: List[str]):
    # Compute
    ssm_estimator = SSMEstimator(vehicle_record)
    ssm_estimator.include_ssms(ssm_names)
    # Aggregate
    aggregation_period = 30  # [s]
    columns = ['time_interval'] + ssm_names
    create_time_bins_and_labels(aggregation_period, vehicle_record)
    aggregated_data = vehicle_record[columns].groupby(
        'time_interval').sum()
    aggregated_data.reset_index(inplace=True)
    return aggregated_data


def extract_risky_maneuvers(vehicle_record: pd.DataFrame,
                            risk_name: str):
    """
    Find risky maneuvers and write their information on a dataframe
    A risky maneuver is defined as a time interval during which risk is
    greater than zero. We save the vehicles involved, start and end
    times, total (sum over duration) risk and pointwise max risk of each
    risky maneuver.

    :param vehicle_record: dataframe with step by step vehicle data
     including some risk measurement
    :param risk_name: columns of the dataframe with the relevant risk for
     this scenario
    :return: dataframe where each row represents one risky maneuver
    """

    # Note: some vehicles enter simulation with positive risk. This
    # makes it hard to deal with the whole dataframe at once.
    # Thus, we deal with one vehicle at a time. There should be no more
    # than 10k vehicles per simulation, so I'm hoping computations won't
    # take too long.
    # Spoiler alert: it takes some minutes to run over 10 files

    risk_margin = 0.1  # ignore total risks below this margin
    vehicle_record['is_risk_positive'] = vehicle_record[risk_name] > 0
    all_ids = vehicle_record['veh_id'].unique()
    risky_data_list = [pd.DataFrame(
        columns=['veh_id', 'leader_id', 'time', 'end_time',
                 'total_risk', 'max_risk'])]
    for veh_id in all_ids:
        single_veh_record = vehicle_record[vehicle_record['veh_id']
                                           == veh_id]
        if not any(single_veh_record['is_risk_positive']):
            continue
        delta_t = round(single_veh_record['time'].iloc[1]
                        - single_veh_record['time'].iloc[0], 2)

        # TODO: compare computation speeds
        risk_transition_idx = (single_veh_record['is_risk_positive'].diff()
                               != 0)
        # risk_transition_idx = (
        #   single_veh_record['is_risk_positive'].shift()
        #   != single_veh_record['is_risk_positive'])

        # Dealing with cases where the vehicle enters simulation with
        # positive risk:
        risk_transition_idx.iloc[0] = (
            single_veh_record['is_risk_positive'].iloc[0])
        risk_transition_idx = risk_transition_idx[risk_transition_idx]
        start_indices = risk_transition_idx.iloc[::2].index
        end_indices = risk_transition_idx.iloc[1::2].index
        # Dealing with cases where the vehicle leaves simulation with
        # positive risk (should never occur with full simulation data):
        if len(end_indices) == len(start_indices) - 1:
            end_indices = end_indices.append(
                pd.Index([single_veh_record.index[-1]]))

        # Build the df with data for each risky maneuver for this vehicle
        temp_df = single_veh_record[['veh_id', 'leader_id', 'time']].loc[
            start_indices]
        try:
            temp_df['end_time'] = single_veh_record['time'].loc[
                end_indices].values
        except ValueError:
            end_times = single_veh_record['time'].loc[
                end_indices].values
            temp_df['end_time'] = 0
            temp_df['end_time'].iloc[:len(end_times)] = end_times
        risk = single_veh_record[risk_name]
        risk_grouped = risk.groupby((risk == 0).cumsum())
        cumulative_risk = risk_grouped.cumsum() * delta_t
        cumulative_max_risk = risk_grouped.cummax()
        temp_df['total_risk'] = cumulative_risk.shift().loc[
            end_indices].values
        temp_df['max_risk'] = cumulative_max_risk.shift().loc[
            end_indices].values
        temp_df.drop(index=temp_df[temp_df < risk_margin].index,
                     inplace=True)
        risky_data_list.append(temp_df)
    risky_data = pd.concat(risky_data_list)
    if risky_data.empty:
        risky_data.loc[0] = 0
    risky_data['simulation_number'] = vehicle_record[
        'simulation_number'].iloc[0]
    return risky_data


def merge_data(network_name: str, vehicle_type: List[VehicleType],
               controlled_vehicle_percentage: List[int]):
    """Loads data collections, link evaluation and SSM data, merges them
    and saves the merged dataframe to a file
    If there is more than one link evaluation segment or more than one
    data collection measurement, the resulting dataset will have a lot of
    redundant information"""

    print('Started merging data')

    data_readers = (
        readers.LinkEvaluationReader(network_name),
        readers.DataCollectionReader(network_name),
        readers.SSMDataReader(network_name)
    )
    percentage_columns = [vt.name.lower() + '_percentage' for vt in
                          vehicle_type]
    shared_cols = ['vehicles_per_lane', 'time_interval',
                   'random_seed', 'simulation_number'] + percentage_columns
    merged_data = pd.DataFrame()
    for reader in data_readers:
        data = reader.load_data_with_controlled_percentage(
            [vehicle_type], [controlled_vehicle_percentage])
        if merged_data.empty:
            merged_data = data
        else:
            merged_data = merged_data.merge(right=data, how='inner',
                                            on=shared_cols)
    # DataPostProcessor.clean_headers(merged_data)
    compute_flow(merged_data)
    create_time_in_minutes(merged_data)
    writer = data_writer.MergedDataWriter(network_name, vehicle_type)
    writer.save_as_csv(merged_data, controlled_vehicle_percentage)

    print('Merged data saved')


def save_safety_files(vehicle_input: int,
                      writers: List[data_writer.SSMDataWriter],
                      data: List[List[pd.DataFrame]],
                      controlled_percentage: List[int]):
    """

    :param vehicle_input:
    :param writers:
    :param data:
    :param controlled_percentage:
    :return:
    """
    print('Files with input ', vehicle_input, ' done. Saving to file...')
    for i in range(len(writers)):
        writers[i].save_as_csv(pd.concat(data[i]), controlled_percentage,
                               vehicle_input)
    print('Successfully saved.')


def check_human_take_over(network_name: str,
                          vehicle_type: List[VehicleType],
                          controlled_vehicle_percentage: List[int],
                          vehicle_inputs: List[int] = None):
    """Reads multiple vehicle record data files to check how often the
    autonomous vehicles gave control back to VISSIM

    :param network_name: Network name. Either the actual file name or the
     network nickname. Currently available: in_and_out, in_and_merge, i710,
     us101
    :param vehicle_type: Enum to indicate the vehicle (controller) type
    :param controlled_vehicle_percentage: Percentage of controlled vehicles
     present in the simulation.
    :param vehicle_inputs: simulation vehicle inputs to be checked
    :return: Nothing.
    """

    vehicle_record_reader = readers.VehicleRecordReader(network_name)
    data_generator = vehicle_record_reader.generate_data(
        vehicle_type, controlled_vehicle_percentage, vehicle_inputs)
    n_blocked_vehs = []
    for (vehicle_records, file_number) in data_generator:
        blocked_vehs = vehicle_records.loc[
            (vehicle_records['vissim_control'] == 1)
            & (vehicle_records['veh_type'] != Vehicle.VISSIM_CAR_ID),
            'veh_id'].unique()
        n_blocked_vehs.append(blocked_vehs.shape[0])
        print('Total blocked vehicles: ', blocked_vehs.shape[0])
    print('Mean blocked vehicles:', np.mean(n_blocked_vehs))


def find_traffic_light_violations_all(network_name: str,
                                      vehicle_type: List[VehicleType],
                                      controlled_vehicle_percentage: List[int],
                                      vehicle_inputs: List[int] = None,
                                      debugging: bool = False):
    """
    Reads multiple vehicle record data files, looks for cases of
    traffic light violation, and records them in a new csv file

    :param network_name: only network with traffic lights is traffic_lights
    :param vehicle_type: Vehicle type enum. Choose between
     TRAFFIC_LIGHT_ACC or TRAFFIC_LIGHT_CACC
    :param controlled_vehicle_percentage: Percentage of controlled vehicles
     present in the simulation.
    :param vehicle_inputs: Vehicle inputs for which we want SSMs
     computed. If None (default), computes SSMs for all simulated vehicle
     inputs.
    :param debugging: If true, we load only 10^5 samples from the vehicle
     records and do not save results.
    :return: Nothing. Violation results are saved to as csv files"""

    traffic_light_reader = readers.TrafficLightSourceReader(network_name)
    try:
        traffic_light_data = traffic_light_reader.load_data()
    except FileNotFoundError:
        print('No traffic light source file, so no violations.')
        return

    if 'starts_red' not in traffic_light_data.columns:
        traffic_light_data['starts_red'] = True
    traffic_light_data['cycle_time'] = (
            traffic_light_data['red duration']
            + traffic_light_data['green duration']
            + traffic_light_data['amber duration'])

    violations_list = []
    vehicle_record_reader = readers.VehicleRecordReader(network_name)
    n_rows = 10 ** 6 if debugging else None
    data_generator = vehicle_record_reader.generate_data(
        vehicle_type, controlled_vehicle_percentage, vehicle_inputs,
        n_rows)
    for (vehicle_records, file_number) in data_generator:
        violations_list.append(
            find_traffic_light_violations(vehicle_records, traffic_light_data))
    violations = pd.concat(violations_list)
    tf_violation_writer = data_writer.TrafficLightViolationWriter(
        network_name, vehicle_type)
    tf_violation_writer.save_as_csv(violations,
                                    controlled_vehicle_percentage, 0)


def find_traffic_light_violations(vehicle_record: pd.DataFrame,
                                  traffic_light_data: pd.DataFrame):
    """
    Finds red light running violations in a single simulation.

    :param vehicle_record: dataframe with step by step vehicle data
    :param traffic_light_data: dataframe where each row describes a
     traffic light
    :return: dataframe where each row represents one traffic light violation
    """

    violations_list = []
    warmup_time = vehicle_record['time'].iloc[0]
    for _, tf in traffic_light_data.iterrows():
        vehicle_record['dist_to_tf'] = (vehicle_record['x']
                                        - tf['position'])
        after_tf = vehicle_record[vehicle_record['dist_to_tf'] > 0]
        crossing_time = after_tf.loc[
            after_tf.groupby('veh_id').dist_to_tf.idxmin(),
            'time']
        crossing_time = crossing_time[crossing_time > warmup_time]
        tf_cycle = tf['cycle_time']
        violation_idx = crossing_time[(crossing_time % tf_cycle) <
                                      tf['red duration']].index
        violations_per_tf = vehicle_record.loc[
            violation_idx, ['simulation_number', 'veh_id', 'time',
                            'vehicles_per_lane']]
        violations_per_tf['traffic_light'] = tf['id']
        violations_list.append(violations_per_tf)
    return pd.concat(violations_list)


def add_main_ssms(vehicle_records: pd.DataFrame, ssm_names: List[str]):
    ssm_estimator = SSMEstimator(vehicle_records)
    for ssm in ssm_names:
        ssm_estimator.include_ssm_by_name(ssm)
    # if network_name in ['in_and_out', 'i710', 'us101']:
    #     ssm_estimator.include_ttc()
    #     ssm_estimator.include_drac()
    #     for choice in [False, True]:
    #         ssm_estimator.include_collision_free_gap(
    #             consider_lane_change=choice)
    #         ssm_estimator.include_risk(consider_lane_change=choice)
    # elif network_name in ['traffic_lights']:
    #     ssm_estimator.include_barrier_function_risk()


def compute_discomfort(vehicle_records: pd.DataFrame):
    """
    Computes discomfort as the integral of |a(t) - a_comf| if |a(t)| > a_comf.

    :param vehicle_records: vehicle records data loaded from VISSIM
    :return: Dataframe with values of discomfort for all vehicles every
    'aggregation period' seconds
    """
    comfortable_brake = -4
    discomfort_idx = vehicle_records['ax'] < comfortable_brake
    vehicle_records['discomfort'] = 0
    vehicle_records.loc[discomfort_idx, 'discomfort'] = (
        comfortable_brake - vehicle_records.loc[discomfort_idx, 'ax'])
    try:
        aggregated_acceleration_data = vehicle_records[[
            'time_interval', 'discomfort']].groupby('time_interval').sum()
    except KeyError:
        aggregation_period = 30
        create_time_bins_and_labels(aggregation_period, vehicle_records)
        aggregated_acceleration_data = vehicle_records[[
            'time_interval', 'discomfort']].groupby('time_interval').sum()
    aggregated_acceleration_data.reset_index(inplace=True)
    return aggregated_acceleration_data


def create_time_bins_and_labels(period, vehicle_records):
    """Creates equally spaced time intervals, generates labels
    that go with them and includes a time_interval column to the
    dataframe.

    :param vehicle_records: vehicle records data loaded from VISSIM
    :param period: time interval length
    :return: None; alters the data in place"""
    final_time = int(vehicle_records['time'].iloc[-1])
    interval_limits = []
    interval_labels = []
    for i in range(period, final_time + period,
                   period):
        interval_limits.append(i)
        interval_labels.append(str(i) + '-'
                               + str(i + period))
    vehicle_records['time_interval'] = pd.cut(
        x=vehicle_records['time'], bins=interval_limits,
        labels=interval_labels[:-1])


def check_already_processed_vehicle_inputs(
        network_name: str, vehicle_type: List[VehicleType],
        controlled_vehicle_percentage: List[int], vehicle_inputs: List[int]):
    """
    Checks if the scenario being post processed was processed before.
    Prints a message if yes.
    :param network_name: Currently available: in_and_out, in_and_merge,
     i710, us101, traffic_lights
    :param vehicle_type: Enum to indicate the vehicle (controller) type
    :param controlled_vehicle_percentage: Percentage of controlled vehicles
     present in the simulation
    :param vehicle_inputs: number of vehicles entering the simulation per
     hour
    :return: nothing. Just prints a message on the console.
    """
    ssm_reader = readers.SSMDataReader(network_name)
    try:
        ssm_data = ssm_reader.load_data_with_controlled_percentage(
            [vehicle_type], [controlled_vehicle_percentage], vehicle_inputs)
    except OSError:
        return
    # if not ssm_data.empty:
    processed_vehicle_inputs = ssm_data['vehicles_per_lane'].unique()
    for v_i in (set(vehicle_inputs) & set(processed_vehicle_inputs)):
        print('FYI: SSM results for network {}, vehicle type {}, '
              'percentage {}, and input {} already exist. They are '
              'being recomputed.'.
              format(network_name, [vt.name for vt in vehicle_type],
                     controlled_vehicle_percentage, v_i))


def compute_distance_to_leader(data_source: str, veh_data: pd.DataFrame,
                               adjusted_idx: np.array,
                               adjusted_leader_idx: np.array):
    """
    Computes the longitudinal distance between a vehicle's front
    bumper to the rear bumper of the leading vehicle
    :param data_source: vissim, ngsim or synthetic
    :param veh_data: vehicle data during a single time step
    :param adjusted_idx: vehicle indices starting from zero
    :param adjusted_leader_idx: leader indices starting from zero
    """
    n = np.max(adjusted_idx) + 1
    if data_source == DataPostProcessor.VISSIM:

        front_x_vector = np.zeros(n)
        front_y_vector = np.zeros(n)
        rear_x_vector = np.zeros(n)
        rear_y_vector = np.zeros(n)

        front_x_vector[adjusted_idx] = veh_data['front_x']
        rear_x_vector[adjusted_idx] = veh_data['rear_x']
        front_y_vector[adjusted_idx] = veh_data['front_y']
        rear_y_vector[adjusted_idx] = veh_data['rear_y']
        distance = np.sqrt((rear_x_vector[adjusted_leader_idx]
                            - front_x_vector[adjusted_idx]) ** 2
                           + (rear_y_vector[adjusted_leader_idx]
                              - front_y_vector[adjusted_idx]) ** 2)
        # Set gap to zero when there's no leader
        distance[adjusted_idx == adjusted_leader_idx] = 0

    elif data_source == DataPostProcessor.NGSIM:
        length = np.zeros(n)
        length[adjusted_idx] = veh_data['length']
        leader_length = length[adjusted_leader_idx]
        distance = veh_data['delta_x'] - leader_length
    else:
        distance = veh_data['delta_x']

    return distance


class DataPostProcessor:
    """Class grouping different post processing methods."""

    VISSIM = 'vissim'
    NGSIM = 'ngsim'
    SYNTHETIC = 'synthetic'
    integer_columns = {'veh_id': int, 'leader_id': int,
                       'veh_type': int, 'lane': int}

    highway_scenarios = {'in_and_out', 'in_and_merge', 'i710', 'us101'}
    traffic_light_scenarios = {'traffic_lights'}

    # def __init__(self, data_source):
    #     """
    #     :param data_source: string 'vissim' or 'ngsim'
    #     """
    #     self.data_source = data_source.lower()
    #    # self.veh_records = data

    def post_process_data(self, data_source: str,
                          vehicle_records: pd.DataFrame):
        """Post processing includes:
         - Computing relative velocity and bumper to bumper distance to leader,
         - Adding leader type
         - Converting all data to SI units
        :param data_source: vissim, ngsim or synthetic
        :param vehicle_records: detailed vehicle states over time
        :return: nothing, it alters the data in place"""

        if data_source.lower() == DataPostProcessor.VISSIM:
            DataPostProcessor.post_process_vissim_vehicle_record(
                vehicle_records)
        elif data_source.lower() == DataPostProcessor.NGSIM:
            self.post_process_ngsim_data(vehicle_records)
        elif data_source == DataPostProcessor.SYNTHETIC:
            DataPostProcessor.post_process_synthetic_data(vehicle_records)
        else:
            print('[DataPostProcessor] Trying to process data from unknown '
                  'data source.')
            return

    @staticmethod
    def post_process_vissim_vehicle_record(vehicle_records):
        """
        Process fzp vehicle record data file generated by VISSIM
        :return: None
        """
        # veh_data = self.veh_records

        kph_to_mps = 1 / 3.6
        vehicle_records['vx'] = vehicle_records['vx'] * kph_to_mps

        # warm_up_time = 60
        # DataPostProcessor.remove_early_samples(vehicle_records, warm_up_time)

        # When the vehicle is stopping due to a traffic light, we consider it
        # has no leader
        if 'leader_type' in vehicle_records.columns:
            signal_ahead_idx = vehicle_records['leader_type'] == 'Signal head'
            vehicle_records.loc[signal_ahead_idx, 'leader_id'] = np.nan
        # By convention, if vehicle has no leader, we set it as its own leader
        vehicle_records['leader_id'].fillna(vehicle_records['veh_id'],
                                            inplace=True, downcast='infer')
        # Compute relative velocity to the vehicle's leader (own vel minus
        # leader vel) Note: we need this function because VISSIM output
        # 'SpeedDiff' is not always correct. It has been observed to equal
        # the vehicle's own speed at the previous time step.
        DataPostProcessor.compute_values_relative_to_leader(
            DataPostProcessor.VISSIM, vehicle_records)

    def post_process_ngsim_data(self, vehicle_records):
        """
        Process csv data file generated from NGSIM
        :return: None
        """
        # vehicle_record = self.veh_records
        vehicle_records.astype(self.integer_columns, copy=False)

        columns_in_feet = ['x', 'y', 'vx', 'length', 'delta_x']
        foot_to_meter = 0.3048
        vehicle_records[columns_in_feet] *= foot_to_meter

        base_time = min(vehicle_records['time'])
        # From milliseconds to deciseconds
        vehicle_records['time'] = (vehicle_records['time'] - base_time) // 100
        vehicle_records.sort_values(by=['time', 'veh_id'], inplace=True)
        # Define warm up time as the first moment some vehicle has a leader
        warm_up_time = min(vehicle_records.loc[vehicle_records['leader_id'] > 0,
                                               'time'])
        DataPostProcessor.remove_early_samples(warm_up_time, vehicle_records)
        # By convention, if vehicle has no leader, we set it as its own leader
        no_leader_idx = vehicle_records['leader_id'] == 0
        vehicle_records.loc[no_leader_idx,
                            'leader_id'] = vehicle_records['veh_id']

        vehicle_records['delta_x_old'] = vehicle_records[
            'delta_x']  # for checks

        DataPostProcessor.compute_values_relative_to_leader(
            DataPostProcessor.NGSIM, vehicle_records)

    @staticmethod
    def post_process_synthetic_data(vehicle_records):
        """
        Process csv file synthetically generated
        :return: None
        """
        DataPostProcessor.compute_values_relative_to_leader(
            DataPostProcessor.SYNTHETIC, vehicle_records)

    @staticmethod
    def remove_early_samples(vehicle_records: pd.DataFrame,
                             warm_up_time: float):
        """Remove samples with time below some warm up time

        :param vehicle_records: vehicle records loaded from VISSIM
        :param warm_up_time: time below which samples are removed"""

        veh_data = vehicle_records
        below_warmup = veh_data.loc[veh_data['time'] <= warm_up_time].index
        print('Removing {} warm up time samples'.format(len(below_warmup)))
        veh_data.drop(index=below_warmup, inplace=True)

    @staticmethod
    def compute_values_relative_to_leader(data_source: str,
                                          vehicle_records: pd.DataFrame):
        """Computes bumper to bumper distance and relative speed to preceding
        vehicle, and adds the preceding vehicle type."""

        # vehicle_records = self.veh_records
        total_samples = vehicle_records.shape[0]
        print('Adding distance, relative speed and leader type for {} '
              'samples'.format(total_samples))

        percent = 0.1
        out_of_bounds_idx = []
        grouped_by_time = vehicle_records.groupby('time')
        delta_v = np.zeros(total_samples)
        leader_type = np.zeros(total_samples)
        distance = np.zeros(total_samples)
        counter = 0
        for _, current_data in grouped_by_time:
            veh_idx = current_data['veh_id'].to_numpy()
            leader_idx = current_data['leader_id'].to_numpy()
            min_idx = min(veh_idx)
            max_idx = max(veh_idx)
            n_vehicles = max_idx - min_idx + 1
            vel_vector = np.zeros(n_vehicles)
            type_vector = np.zeros(n_vehicles)

            # If leader is not in the current time, we proceed as if there
            # was no leader (happens more often with NGSIM data)
            out_of_bounds_check = current_data['leader_id'] > max_idx
            if np.any(out_of_bounds_check):
                # Save the indices with issues to correct the original
                # dataframe after the loop
                out_of_bounds_idx.extend(list(out_of_bounds_check.
                                              index[out_of_bounds_check]))
                # Then correct the indices for this loop iteration
                leader_idx[out_of_bounds_check] = veh_idx[out_of_bounds_check]

            adjusted_idx = veh_idx - min_idx
            adjusted_leader_idx = leader_idx - min_idx
            vel_vector[adjusted_idx] = current_data['vx']
            delta_v[counter:counter + current_data.shape[0]] = (
                    vel_vector[adjusted_idx] - vel_vector[adjusted_leader_idx])

            type_vector[adjusted_idx] = current_data['veh_type']
            leader_type[counter:counter + current_data.shape[0]] = (
                type_vector[adjusted_leader_idx])

            distance[counter:counter + current_data.shape[0]] = (
                compute_distance_to_leader(data_source, current_data,
                                           adjusted_idx, adjusted_leader_idx))
            counter += current_data.shape[0]

            if counter >= percent * total_samples:
                print('{:.0f}%'.format(counter / total_samples * 100), end=',')
                percent += 0.1
        print()  # skip a line
        vehicle_records['delta_v'] = delta_v
        vehicle_records['leader_type'] = leader_type
        vehicle_records['delta_x'] = distance

        if len(out_of_bounds_idx) > 0:
            vehicle_records.loc[out_of_bounds_idx, 'leader_id'] = (
                vehicle_records.loc[out_of_bounds_idx, 'veh_id'])
            print('Found {} instances of leaders outside the simulation'.
                  format(len(out_of_bounds_idx)))

    def create_simulation_summary(self, network_name: str,
                                  vehicle_type: List[VehicleType],
                                  controlled_percentage: List[int],
                                  vehicle_inputs: List[int] = None,
                                  debugging: bool = False):
        """Reads multiple vehicle record data files, postprocesses them,
        computes and aggregates SSMs results, extracts risky maneuvers, and
        find traffic light violations. SSMs, risky maneuvers and
        violations are saved as csv files per vehicle input.

        :param network_name: Currently available: in_and_out, in_and_merge,
         i710, us101, traffic_lights
        :param vehicle_type: Enum to indicate the vehicle (controller) type
        :param controlled_percentage: Percentage of controlled vehicles
         present in the simulation.
        :param vehicle_inputs: Vehicle inputs for which we want SSMs
         computed. If None (default), computes SSMs for all simulated vehicle
         inputs.
        :param debugging: If true, we load only 10^5 samples from the vehicle
         records and do not save results.
        :return: Nothing. SSM results are saved to as csv files"""

        check_already_processed_vehicle_inputs(
            network_name, vehicle_type, controlled_percentage,
            vehicle_inputs)

        if network_name in DataPostProcessor.highway_scenarios:
            ssm_names = ['low_TTC', 'high_DRAC', 'risk',
                         'risk_no_lane_change']
            risk_name = 'risk'
        elif network_name in DataPostProcessor.traffic_light_scenarios:
            ssm_names = ['barrier_function_risk']
            risk_name = 'barrier_function_risk'
        else:
            raise ValueError('Unknown network name.')

        traffic_light_reader = readers.TrafficLightSourceReader(
            network_name)
        try:
            traffic_light_data = traffic_light_reader.load_data()
        except FileNotFoundError:
            traffic_light_data = pd.DataFrame()
            print('No traffic light data found -> no violations')

        ssm_writer = data_writer.SSMDataWriter(network_name, vehicle_type)
        risky_maneuver_writer = data_writer.RiskyManeuverWriter(network_name,
                                                                vehicle_type)
        violation_writer = data_writer.TrafficLightViolationWriter(
            network_name, vehicle_type)
        discomfort_writer = data_writer.DiscomfortWriter(network_name,
                                                         vehicle_type)
        vehicle_record_reader = readers.VehicleRecordReader(network_name)
        n_rows = 10 ** 6 if debugging else None
        for vi in vehicle_inputs:
            print('Start of safety summary creation for network {}, vehicle '
                  'type {}, percentage {}, and input {}'.format(
                    network_name, [vt.name.lower() for vt in vehicle_type],
                    controlled_percentage, vi))
            data_generator = vehicle_record_reader.generate_data(
                vehicle_type, controlled_percentage, [vi],
                n_rows)
            ssm_data = []
            risky_maneuvers = []
            violations_data = []
            discomfort_data = []
            for (vehicle_records, file_number) in data_generator:
                # self.post_process_data(DataPostProcessor.VISSIM, vehicle_records)
                # print('Computing SSMs')
                # aggregated_data = create_ssm_dataframe(vehicle_records,
                #                                        ssm_names)
                # aggregated_data.insert(0, 'simulation_number', file_number)
                # aggregated_data['vehicles_per_lane'] = vi
                # ssm_data.append(aggregated_data)
                # print('Extracting risky maneuvers')
                # risky_maneuvers.append(extract_risky_maneuvers(
                #     vehicle_records, risk_name))
                if not traffic_light_data.empty:
                    print('Measuring discomfort')  # so far only used in
                    # traffic light simulations
                    discomfort_data.append(compute_discomfort(vehicle_records))
                    # print('Looking for traffic light violations')
                    # violations_data.append(find_traffic_light_violations(
                    #     vehicle_records, traffic_light_data))
                print('-' * 79)

            writers = [ssm_writer, risky_maneuver_writer]
            data = [ssm_data, risky_maneuvers]
            if not traffic_light_data.empty:
                writers += [violation_writer, discomfort_writer]
                data += [violations_data, discomfort_data]
            save_safety_files(vi, [discomfort_writer], [discomfort_data],
                              controlled_percentage)

    # @staticmethod
    # def clean_headers(data: pd.DataFrame):
    #     """ Deletes '(ALL)' from some columns names
    #
    #     :param data: Dataframe with data from all sources (link evaluation,
    #      data collection, and ssm)
    #     """
    #     # Some column names contain (ALL). We can remove that information
    #     column_names = [name.split('(')[0] for name in data.columns]
    #     data.columns = column_names


class SSMEstimator:
    coded_ssms = {'TTC', 'DRAC', 'CPI', 'collision_free_gap', 'DTSG', 'vf_gap',
                  'risk', 'estimated_risk', 'barrier_function_safe_gap'}
    ttc_threshold = 1.5  # [s]
    drac_threshold = 3.5  # [m/s2]

    def __init__(self, veh_data):
        self.veh_data = veh_data

    def include_ssms(self, ssm_names: List[str]):
        for ssm in ssm_names:
            self.include_ssm_by_name(ssm)

    def include_ssm_by_name(self, ssm_name: str):
        ssm_name = ssm_name.lower()
        if 'ttc' in ssm_name:
            self.include_ttc()
        elif 'drac' in ssm_name:
            self.include_drac()
        elif ssm_name == 'risk':
            self.include_collision_free_gap()
            self.include_risk()
        elif ssm_name == 'risk_no_lane_change':
            self.include_collision_free_gap(consider_lane_change=False)
            self.include_risk(consider_lane_change=False)
        elif ssm_name == 'barrier_function_risk':
            self.include_barrier_function_risk()
        else:
            raise ValueError('ssm ', ssm_name, ' cannot be called by the '
                                               'include_ssm_by_name method')

    def include_ttc(self, safe_threshold: float = ttc_threshold):
        """
        Includes Time To Collision (TTC) and a flag indicating if the TTC is
        below a threshold to the the dataframe.
        TTC = deltaX/deltaV if follower is faster; otherwise infinity

        :param safe_threshold: [Optional] Threshold against which TTC values
         are compared.
        :return: None
        """
        veh_data = self.veh_data
        veh_data['TTC'] = float('nan')
        valid_ttc_idx = veh_data['delta_v'] > 0
        ttc = (veh_data['delta_x'].loc[valid_ttc_idx]
               / veh_data['delta_v'].loc[valid_ttc_idx])
        veh_data.loc[valid_ttc_idx, 'TTC'] = ttc
        veh_data['low_TTC'] = veh_data['TTC'] < safe_threshold

    def include_drac(self, safe_threshold: float = drac_threshold):
        """
        Includes Deceleration Rate to Avoid Collision (DRAC) and a flag
        indicating if the DRAC is above a threshold to the dataframe.
        DRAC = deltaV^2/(2.deltaX), if follower is faster; otherwise zero

        :param safe_threshold: [Optional] Threshold against which DRAC values
         are compared.
        :return: None
        """
        veh_data = self.veh_data
        veh_data['DRAC'] = 0
        valid_drac_idx = veh_data['delta_v'] > 0
        drac = (veh_data['delta_v'].loc[valid_drac_idx] ** 2
                / 2 / veh_data['delta_x'].loc[valid_drac_idx])
        veh_data.loc[valid_drac_idx, 'DRAC'] = drac
        veh_data['high_DRAC'] = veh_data['DRAC'] > safe_threshold

    def include_cpi(self, max_decel_data, is_default_vissim=True):
        """
        Includes Crash Probability Index (CPI) to the dataframe

        CPI = Prob(DRAC > MADR), where MADR is the maximum available
        deceleration rate. Formally, we should check the truncated Gaussian
        parameters for each velocity. However, the default VISSIM max
        decel is a linear function of the velocity and the other three
        parameters are constant. We make use of this to speed up this
        function.

        :param max_decel_data: dataframe with maximum deceleration per
        vehicle type per speed
        :param is_default_vissim: boolean to identify if data was generated
        using default VISSIM deceleration parameters
        :return: None
        """

        veh_types = np.unique(
            max_decel_data.index.get_level_values('veh_type'))
        df = self.veh_data
        if 'DRAC' not in df.columns:
            self.include_drac()
        df['CPI'] = 0
        # veh_types = np.unique(df['veh type'])
        for veh_type in veh_types:
            idx = (df['veh_type'] == veh_type) & (df['DRAC'] > 0)
            if is_default_vissim:
                first_row = max_decel_data.loc[veh_type, 0]
                possible_vel = max_decel_data.loc[veh_type].index
                min_vel = 0
                max_vel = max(possible_vel)
                decel_min_vel = max_decel_data.loc[veh_type, min_vel]['mean']
                decel_max_vel = max_decel_data.loc[veh_type, max_vel]['mean']
                madr_array = (decel_min_vel + (decel_max_vel - decel_min_vel)
                              / max_vel * df.loc[idx, 'vx'])
                df.loc[idx, 'CPI'] = truncnorm.cdf(df.loc[idx, 'DRAC'],
                                                   a=first_row['norm_min'],
                                                   b=first_row['norm_max'],
                                                   loc=(-1) * madr_array,
                                                   scale=first_row['std'])
            else:
                a_array = []
                b_array = []
                madr_array = []
                std_array = []
                for vel in df.loc[idx, 'vx']:
                    row = max_decel_data.loc[veh_type, round(vel, -1)]
                    a_array.append(row['norm_min'])
                    b_array.append(row['norm_max'])
                    madr_array.append(-1 * row['mean'])
                    std_array.append(row['std'])
                df.loc[idx, 'CPI'] = truncnorm.cdf(df.loc[idx, 'DRAC'],
                                                   a=a_array, b=b_array,
                                                   loc=madr_array,
                                                   scale=std_array)

    def include_collision_free_gap(self, same_type_gamma=1,
                                   consider_lane_change: bool = True):
        """
        Computes the collision free (safe) gap and adds it to the dataframe.
        If the vehicle violates the safe gap, the absolute value of the
        distance to the safe gap (DTSG) is also added to the dataframe. The
        DTSG column is padded with zeros.
        :param same_type_gamma: factor multiplying standard value of maximum
         braking of the leader when both leader and follower are of the same
         type. Values greater than 1 indicate more conservative assumptions
        :param consider_lane_change: if set to false, we don't consider the
         effects of lane change, i.e., we treat all situations as simple
         vehicle following. If set to true, we overestimate the risk by
         assuming a reduced max brake during lane changes.
        """

        ssm_1 = 'collision_free_gap'
        ssm_2 = 'DTSG'
        self.include_distance_based_ssm(
            ssm_1, same_type_gamma, consider_lane_change=consider_lane_change)

        if not consider_lane_change:
            ssm_1 += '_no_lane_change'
            ssm_2 += '_no_lane_change'
        self.veh_data[ssm_2] = self.veh_data['delta_x'] - self.veh_data[ssm_1]
        self.veh_data.loc[self.veh_data[ssm_2] > 0, ssm_2] = 0
        self.veh_data[ssm_2] = np.abs(self.veh_data[ssm_2])

    def include_vehicle_following_gap(self, same_type_gamma=1, rho=0.2,
                                      free_flow_velocity=None):
        """
        Includes time headway based desired vehicle following gap to the
        dataframe
        The vehicle following gap is an overestimation of the collision free
        gap which assumes:
        . (1-rho)vE(t) <= vL(t) <= vE(t), for all t
        . vE(t) <= Vf, for all t.

        :param same_type_gamma: factor multiplying standard value of maximum
         braking of the leader when both leader and follower are of the same
         type. Values greater than 1 indicate more conservative assumptions
        :param rho: defines the lower bound on the leader velocity following
         (1-rho)vE(t) <= vL(t). Must be in the interval [0, 1]
        :param free_flow_velocity: (optional) must be given in m/s
        :return:
        """
        self.include_distance_based_ssm('vf_gap', same_type_gamma, rho,
                                        free_flow_velocity)

    def include_risk(self, same_type_gamma: float = 1,
                     consider_lane_change: bool = True):
        """
        Includes exact risk, computed as the relative velocity at
        collision time under the worst case scenario, to the dataframe

        :param consider_lane_change: if set to false, we don't consider the
         effects of lane change, i.e., we treat all situations as simple
         vehicle following. If set to true, we overestimate the risk by
         assuming a reduced max brake during lane changes.
        :param same_type_gamma: factor multiplying standard value of maximum
         braking of the leader when both leader and follower are of the same
         type. Values greater than 1 indicate more conservative assumptions
        """
        self.include_distance_based_ssm(
            'risk', same_type_gamma,
            consider_lane_change=consider_lane_change)

    def include_estimated_risk(self, same_type_gamma=1, rho=0.2,
                               free_flow_velocity=None):
        """
        Includes estimated risk, which is an overestimation of the exact risk
        under certain assumptions, to the dataframe
        Assumptions:
        . (1-rho)vE(t) <= vL(t) <= vE(t), for all t
        . vE(t) <= Vf, for all t.

        :param same_type_gamma: factor multiplying standard value of maximum
         braking of the leader when both leader and follower are of the same
         type. Values greater than 1 indicate more conservative assumptions
        :param rho: defines the lower bound on the leader velocity following
         (1-rho)vE(t) <= vL(t). Must be in the interval [0, 1]
        :param free_flow_velocity: (optional) must be given in m/s
        :return:
        """
        self.include_distance_based_ssm('estimated_risk', same_type_gamma, rho,
                                        free_flow_velocity)

    def include_barrier_function_risk(self):
        """
        Computes the safe gap as defined by the control barrier function and
        compares it to the current gap. We only include values if they are
        negative.
        :return: nothing; the method changes the dataframe.
        """
        time_headway = 1
        standstill_distance = 3
        comfortable_braking = 4

        gap = self.veh_data['delta_x'].values
        follower_vel = self.veh_data['vx'].values
        delta_vel = self.veh_data['delta_v'].values
        leader_vel = follower_vel - delta_vel
        safe_gap = (time_headway * follower_vel + standstill_distance
                    + (follower_vel ** 2 - leader_vel ** 2)
                    / 2 / comfortable_braking)
        diff_to_safe_gap = safe_gap - gap
        self.veh_data['barrier_function_risk'] = diff_to_safe_gap
        self.veh_data.loc[diff_to_safe_gap < 0, 'barrier_function_risk'] = 0
        # no leader cases
        self.veh_data.loc[self.veh_data['veh_id'] == self.veh_data['leader_id'],
                          'barrier_function_risk'] = 0

    def include_distance_based_ssm(self, ssm_name: str,
                                   same_type_gamma: float = 1,
                                   rho: float = 0.2,
                                   free_flow_velocity: float = None,
                                   consider_lane_change: bool = True):
        """
        Generic method to include one out of a set of distance based surrogate
        safety measures.
        :param ssm_name: {collision_free_gap, vf_gap, risk, estimated_risk}
        :param same_type_gamma: factor multiplying standard value of maximum
         braking of the leader when both leader and follower are of the same
         type. Values greater than 1 indicate more conservative assumptions
        :param rho: defines the lower bound on the leader velocity following
         (1-rho)vE(t) <= vL(t). Must be in the interval [0, 1]
        :param consider_lane_change: if set to false, we don't consider the
         effects of lane change, i.e., we treat all situations as simple
         vehicle following. If set to true, we overestimate the risk by
         assuming a reduced max brake during lane changes.
        :param free_flow_velocity: (optional) must be given in m/s
        """
        veh_types = np.unique(self.veh_data['veh_type'])
        df = self.veh_data
        has_leader = df['veh_id'] != df['leader_id']
        if not consider_lane_change:
            ssm_name += "_no_lane_change"
        df[ssm_name] = 0

        for follower_type in veh_types:
            try:
                follower = Vehicle(follower_type, gamma=1)
            except KeyError:
                print('Follower of type {} not found. Skipping it.'.
                      format(follower_type))
                continue
            if not follower.is_relevant:
                print('Skipping follower of type ', follower.type)
                continue

            if free_flow_velocity is not None:
                follower.free_flow_velocity = free_flow_velocity
            follower_idx = (df['veh_type'] == follower_type) & has_leader

            for leader_type in veh_types:
                try:
                    if follower_type == leader_type:
                        gamma = same_type_gamma
                    else:
                        gamma = 1
                    leader = Vehicle(leader_type, gamma=gamma)
                except KeyError:
                    print('Leader of type {} not found. Skipping it.'.
                          format(leader_type))
                    continue
                if not leader.is_relevant:
                    print('Skipping leader of type ', leader.type)
                    continue

                veh_idx = follower_idx & (df['leader_type'] == leader_type)
                if ssm_name.startswith('safe_gap'):
                    df.loc[veh_idx, ssm_name] = (
                        self._compute_collision_free_gap(
                            veh_idx, follower, leader, consider_lane_change))
                elif ssm_name.startswith('vf_gap'):
                    df.loc[veh_idx, ssm_name] = (
                        self._compute_vehicle_following_gap(
                            veh_idx, follower, leader, rho))
                elif ssm_name.startswith('risk'):
                    df.loc[veh_idx, ssm_name] = self._compute_risk(
                        veh_idx, follower, leader, consider_lane_change)
                elif ssm_name.startswith('estimated_risk'):
                    df.loc[veh_idx, ssm_name] = (
                        self._compute_estimated_risk(veh_idx, follower,
                                                     leader, rho))
                else:
                    print('Unknown distance based SSM requested. Skipping...')
                    pass

    def _compute_collision_free_gap(self, veh_idx: pd.Series,
                                    follower: Vehicle, leader: Vehicle,
                                    consider_lane_change: bool = False):
        """
        The collision free is computed such that, under the worst case
        braking scenario, both vehicles achieve full stop without colliding

        :param veh_idx: boolean array indicating which vehicles of the
         dataset are being considered
        :param follower: following vehicle - object of the Vehicle class
        :param leader: leading vehicle - object of the Vehicle class
        :param consider_lane_change: if set to false, we don't consider the
         effects of lane change, i.e., we treat all situations as simple
         vehicle following. If set to true, we overestimate the risk by
         assuming a reduced max brake during lane changes.
        :return: safe gaps
        """
        follower_vel = self.veh_data.loc[veh_idx, 'vx'].values
        delta_vel = self.veh_data.loc[veh_idx, 'delta_v'].values
        leader_vel = follower_vel - delta_vel
        safe_gap = np.zeros(len(follower_vel))

        (follower_effective_max_brake,
         follower_effective_lambda1, _) = (
            self.get_braking_parameters_over_time(
                self.veh_data.loc[veh_idx],
                follower,
                consider_lane_change)
        )

        gamma = leader.max_brake / follower_effective_max_brake
        gamma_threshold = leader_vel / (follower_vel
                                        + follower_effective_lambda1)
        is_above = gamma >= gamma_threshold

        safe_gap[is_above] = (
            (follower_vel[is_above] ** 2 /
             (2 * follower_effective_max_brake[is_above])
             - leader_vel[is_above] ** 2 / (2 * leader.max_brake)
             + (follower_effective_lambda1[is_above] * follower_vel[is_above]
                / follower_effective_max_brake[is_above])
             + follower_effective_lambda1[is_above] ** 2 /
             (2 * follower_effective_max_brake[is_above])
             + follower.lambda0))
        if any(gamma) < 1:
            is_below = ~is_above
            brake_difference = (follower_effective_max_brake[is_below] -
                                leader.max_brake)
            safe_gap[is_below] = (
                (delta_vel[is_below] ** 2 / 2 / brake_difference
                 + (follower_effective_lambda1[is_below] * delta_vel[is_below]
                    / brake_difference)
                 + follower_effective_lambda1[is_below] ** 2 /
                 (2 * brake_difference)
                 + follower.lambda0))
        return safe_gap

    def _compute_vehicle_following_gap(self, veh_idx: pd.Series,
                                       follower: Vehicle, leader: Vehicle,
                                       rho: float = 0.2):
        """
        Computes time headway based vehicle following gap

        The vehicle following gap is an overestimation of the collision free
        gap which assumes:
        . (1-rho)vE(t) <= vL(t) <= vE(t), for all t
        . vE(t) <= Vf, for all t.

        :param veh_idx: boolean array indicating which vehicles of the
         dataset are being considered
        :param follower: following vehicle - object of the Vehicle class
        :param leader: leading vehicle - object of the Vehicle class
        :param rho: defines the lower bound on the leader velocity following
         (1-rho)vE(t) <= vL(t). Must be in the interval [0, 1]
        :return: vehicle following gaps
        """""
        h, d = follower.compute_vehicle_following_parameters(
            leader.max_brake, rho)
        return h * self.veh_data.loc[veh_idx, 'vx'] + d

    def _compute_risk(self, veh_idx: pd.Series,
                      follower: Vehicle, leader: Vehicle,
                      consider_lane_change: bool = True):
        """
        Computes the exact risk, which is the relative velocity at collision
        time under the worst case braking scenario

        :param veh_idx: boolean array indicating which vehicles of the
         dataset are being considered
        :param follower: following vehicle - object of the Vehicle class
        :param leader: leading vehicle - object of the Vehicle class
        :param consider_lane_change: if set to false, we don't consider the
         effects of lane change, i.e., we treat all situations as simple
         vehicle following. If set to true, we overestimate the risk by
         assuming a reduced max brake during lane changes.
        :return: exact risks
        """
        gap = self.veh_data.loc[veh_idx, 'delta_x'].values
        follower_vel = self.veh_data.loc[veh_idx, 'vx'].values
        delta_vel = self.veh_data.loc[veh_idx, 'delta_v'].values
        leader_vel = follower_vel - delta_vel
        risk_squared = np.zeros(len(follower_vel))
        safe_gap_col_name = 'safe_gap'
        if not consider_lane_change:
            safe_gap_col_name += '_no_lane_change'
        safe_gap = self.veh_data.loc[veh_idx, safe_gap_col_name].values

        (follower_effective_max_brake,
         follower_effective_lambda1,
         follower_effective_tau_j) = (
            self.get_braking_parameters_over_time(
                self.veh_data.loc[veh_idx],
                follower,
                consider_lane_change))

        gamma = leader.max_brake / follower_effective_max_brake
        gamma_threshold = leader_vel / (follower_vel +
                                        follower_effective_lambda1)
        is_gamma_above = gamma >= gamma_threshold

        # Gap thresholds
        # (note that delta_vel is follower_vel - leader_vel)
        gap_thresholds = [0] * 3
        gap_thresholds[0] = (
                follower.brake_delay
                * (follower.brake_delay / 2 * (follower.accel_t0
                                               + leader.max_brake)
                   + delta_vel))
        # For when the leader is at low speeds
        leader_low_vel_idx = (leader_vel
                              < follower.brake_delay * leader.max_brake)
        gap_thresholds[0][leader_low_vel_idx] = (
                follower.brake_delay
                * (follower.brake_delay / 2 * follower.accel_t0
                   + follower_vel[leader_low_vel_idx])
                - leader_vel[leader_low_vel_idx] ** 2 / 2 / leader.max_brake
        )
        gap_thresholds[1] = (
                (follower.brake_delay + follower_effective_tau_j)
                * (follower_effective_lambda1 + delta_vel
                   - (follower.brake_delay + follower_effective_tau_j) / 2
                   * (follower_effective_max_brake - leader.max_brake))
                + follower.lambda0)
        gap_thresholds[2] = (
                leader_vel / leader.max_brake
                * (follower_effective_lambda1 + follower_vel
                   - (follower_effective_max_brake / leader.max_brake + 1)
                   * leader_vel / 2)
                + follower.lambda0)

        idx_case_1 = gap <= gap_thresholds[0]
        idx_case_2 = ((gap > gap_thresholds[1])
                      & (gap <= gap_thresholds[1]))
        idx_case_3 = ((gap > gap_thresholds[1])
                      & ((is_gamma_above
                          & (gap <= gap_thresholds[2]))
                         | (~is_gamma_above
                            & (gap <= safe_gap))))
        idx_case_4 = (is_gamma_above
                      & (gap > gap_thresholds[2])
                      & (gap <= safe_gap))

        risk_squared[idx_case_1] = (
                delta_vel[idx_case_1] ** 2
                + 2 * gap[idx_case_1] * (follower.accel_t0
                                         + leader.max_brake)
        )
        risk_squared[idx_case_2] = 0
        # In the code, delta_v = v_f - v_l (in the written documents it's
        # usually v_l - v_f)
        risk_squared[idx_case_3] = (
                (-delta_vel[idx_case_3]
                 - follower_effective_lambda1[idx_case_3]) ** 2
                - 2 * (follower_effective_max_brake[idx_case_3]
                       - leader.max_brake)
                * (gap[idx_case_3] - follower.lambda0)
        )
        risk_squared[idx_case_4] = (
                (follower_vel[idx_case_4]
                 + follower_effective_lambda1[idx_case_4]) ** 2
                - 2 * follower_effective_max_brake[idx_case_4] *
                (gap[idx_case_4] - follower.lambda0
                 + leader_vel[idx_case_4] ** 2 / 2 / leader.max_brake)
        )
        # Couple of sanity checks
        if any(idx_case_2):
            print('Collisions during jerk phase:', np.count_nonzero(idx_case_2))
            print('I guess it''s time to code severity for this case...')
        idx_issue = risk_squared < 0
        if np.any(idx_issue):
            print('{} negative risk samples'.format(np.count_nonzero(
                idx_issue)))
            risk_squared[idx_issue] = 0

        return np.sqrt(risk_squared)

    def _compute_estimated_risk(self, veh_idx: pd.Series,
                                follower: Vehicle, leader: Vehicle,
                                rho: float = 0.2):
        """
        Compute estimated risk, which is an overestimation of the exact risk
        under the following assumptions.
        . (1-rho)vE(t) <= vL(t) <= vE(t), for all t
        . vE(t) <= Vf, for all t.

        :param veh_idx: boolean array indicating which vehicles of the
         dataset are being considered
        :param follower: following vehicle - object of the Vehicle class
        :param leader: leading vehicle - object of the Vehicle class
        :param rho: defines the lower bound on the leader velocity following
         (1-rho)vE(t) <= vL(t). Must be in the interval [0, 1]
        :return: estimated risks
        """
        gamma = leader.max_brake / follower.max_brake
        gamma_threshold = ((1 - rho) * follower.free_flow_velocity
                           / (follower.free_flow_velocity + follower.lambda1))
        gap = self.veh_data.loc[veh_idx, 'delta_x'].values
        follower_vel = self.veh_data.loc[veh_idx, 'vx'].values

        if gamma >= gamma_threshold:
            estimated_risk_squared = (
                    ((1 - (1 - rho) ** 2 / gamma) * follower.free_flow_velocity
                     + 2 * follower.lambda1) * follower_vel
                    + follower.lambda1 ** 2
                    - 2 * follower.max_brake * (gap - follower.lambda0)
            )
        else:
            estimated_risk_squared = (
                    (rho ** 2 * follower.free_flow_velocity
                     + 2 * rho * follower.lambda1) * follower_vel
                    + follower.lambda1 ** 2
                    - 2 * follower.max_brake * (1 - gamma)
                    * (gap - follower.lambda0)
            )

        estimated_risk_squared[estimated_risk_squared < 0] = 0
        return np.sqrt(estimated_risk_squared)

    @staticmethod
    def get_braking_parameters_over_time(vehicle_data: pd.DataFrame,
                                         vehicle: Vehicle,
                                         consider_lane_change: bool = True) \
            -> (np.array, np.array, np.array):
        """The vehicle maximum braking is reduced during lane change. This
        function determines when the vehicle is lane changing and returns
        arrays with values of maximum brake and lambda1 at each simulation
        step.

        :param vehicle_data: Must contain either a column 'lane_change' or a
         column 'y'. If only column 'y' exists, sends a warning.
        :param vehicle: object containing the vehicle's parameters
        :param consider_lane_change: if set to false, we don't consider the
         effects of lane change, i.e., we treat all situations as simple
         vehicle following. If set to true, we overestimate the risk by
         assuming a reduced max brake during lane changes.
        :return: Tuple of numpy arrays"""

        # In VISSIM's indication of lane change direction, elements are
        # strings, where 'None' means no lane change.
        # We mimic this by checking the lateral coordinates if the lane
        # change columns is not present.
        if 'lane_change' in vehicle_data.columns:
            lane_change_indicator = vehicle_data['lane_change']
        else:
            warnings.warn('This vehicle record does not contain lane change '
                          'data. It will be estimated by lateral position.\n'
                          'Consider rerunning simulations.')
            lane_change_indicator = (np.abs(
                vehicle_data['y'] - 0.5) < 0.1)
            lane_change_indicator[lane_change_indicator.index[
                lane_change_indicator]] = 'None'

        vehicle_effective_max_brake = (np.ones(len(lane_change_indicator))
                                       * vehicle.max_brake)
        vehicle_effective_lambda1 = (np.ones(len(lane_change_indicator))
                                     * vehicle.lambda1)
        vehicle_effective_tau_j = (np.ones(len(lane_change_indicator))
                                   * vehicle.tau_j)
        if consider_lane_change:
            lane_change_idx = (lane_change_indicator != 'None').values
            vehicle_effective_max_brake[lane_change_idx] = (
                vehicle.max_brake_lane_change)
            vehicle_effective_lambda1[lane_change_idx] = (
                vehicle.lambda1_lane_change)
            vehicle_effective_tau_j[lane_change_idx] = (
                vehicle.tau_j_lane_change)

        return (vehicle_effective_max_brake,
                vehicle_effective_lambda1,
                vehicle_effective_tau_j)
