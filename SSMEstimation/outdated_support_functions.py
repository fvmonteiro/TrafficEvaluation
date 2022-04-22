import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import post_processing
import readers
from vehicle import VehicleType
from vissim_interface import VissimInterface

def generate_us_101_reduced_speed_table(first_file_number=2):
    """Get the link mean speeds for simulations with different speed limits
    and organize them in a nice looking dataframe"""
    reader = readers.ReducedSpeedAreaReader('US_101')
    all_rsa = reader.read_all_reduced_speed_area_files(
        first_file_number=first_file_number)
    last_sim_number = len(all_rsa) + first_file_number - 1
    # lsr = reader.read_link_segment_results(last_sim_number)

    sim_number = []
    speed_limits = []
    mean_speeds = []
    seen_speed_limit_combination = set()
    for i in range(first_file_number, last_sim_number + 1):
        speed_limit_combination = tuple(all_rsa[i - first_file_number][
                                            'speed(10)'])
        lsr = reader.read_link_segment_results(i)
        if (speed_limit_combination not in seen_speed_limit_combination
                and not lsr.empty):
            seen_speed_limit_combination.add(speed_limit_combination)
            sim_number.append(i)
            speed_limits.append(list(speed_limit_combination))
            # all_rsa[i - first_file_number]['speed(10)'].to_numpy())
            mean_speeds.append(lsr.loc[lsr['sim_number'].astype(str) == str(i),
                                       'speed'][:5].values)
        # if len(mean_speeds[-1]) == 0:
        #     print(i)

    col_names = []
    [col_names.append('speed lim lane ' + str(i)) for i in range(1, 6)]
    [col_names.append('mean speed seg. ' + str(i)) for i in range(1, 4)]
    col_names.append('mean speed in ramp')
    col_names.append('mean speed out ramp')
    df = pd.DataFrame(index=sim_number,
                      data=np.hstack((speed_limits,
                                      mean_speeds)),
                      columns=col_names)
    df['mean speed lim'] = df.iloc[:, 0:5].mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(df['mean speed lim'], df['mean speed seg. 1'], 'rx')
    ax.plot(df['mean speed lim'], df['mean speed seg. 2'], 'bo')
    ax.plot(df['mean speed lim'], df['mean speed seg. 3'], 'k*')
    ax.legend(['1', '2', '3'])
    return df


def save_post_processed_data(data, data_source, simulation_name):
    """
    Save data csv file in the post-processed directory
    :param data: pandas dataframe with vehicle data
    :param data_source:
    :param simulation_name: string with file name (no .csv needed)
    """
    # TODO: shouldn't the PostProcessedDataReader contain a saving function?
    post_processed_dir = readers.PostProcessedDataReader_OLD.post_processed_dir
    data_dir = os.path.join(post_processed_dir, data_source)
    data.to_csv(os.path.join(data_dir, simulation_name + '.csv'), index=False)
    print('Data saved at: \n\tFolder: {:s}\n\tName:: {:s}'.
          format(data_dir, simulation_name))


def post_process_and_save(data_source, network_name, vehicle_type):
    if data_source.upper() == 'VISSIM':
        data_reader = readers.VehicleRecordReader(network_name, vehicle_type)
    elif data_source.upper() == 'NGSIM':
        data_reader = readers.NGSIMDataReader(network_name)
    elif data_source == 'synthetic_data':
        data_reader = readers.SyntheticDataReader()
    else:
        print('Trying to process data from unknown data source')
        return

    print('Loading data from {}, network: {}'.
          format(data_source, network_name))
    data = data_reader.load_data()
    print('Raw data shape: ', data.shape)
    data_pp = post_processing
    data_pp.post_process_data(data)
    print('Post processed data shape: ', data.shape)
    save_post_processed_data(data, data_source,
                             data_reader.network_name)


def create_ssm(data_source, network_name, ssm_names):
    if isinstance(ssm_names, str):
        ssm_names = [ssm_names]

    data_reader = readers.PostProcessedDataReader_OLD(data_source, network_name)
    data = data_reader.load_data()
    ssm_estimator = post_processing.SSMEstimator(data)

    for ssm in ssm_names:
        try:
            ssm_method = getattr(ssm_estimator, 'include_' + ssm.lower())
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement `{}`".format(
                    ssm_estimator.__class__.__name__, ssm))

        print('Computing ' + ssm)
        data_rows_before = data.shape[0]
        if ssm == 'CPI':
            if data_source.lower() == 'vissim':
                max_decel_data = (
                    readers.VissimDataReader.load_max_deceleration_data())
                ssm_method(max_decel_data)
            else:
                print('Can only compute CPI for VISSIM simulations')
        else:
            ssm_method()
        data_rows_after = data.shape[0]

        # Temporary check
        if any(data['veh_id'] == 0):
            n_zero_samples = sum(data['veh_id'] == 0)
            print('It looks like {:d} samples were overwritten'.
                  format(n_zero_samples))
        if data_rows_before != data_rows_after:
            print('{:d} extra rows'.format(data_rows_after - data_rows_before))

    print('Saving ' + ', '.join(ssm_names))
    save_post_processed_data(ssm_estimator.veh_data,
                             data_reader.data_source,
                             data_reader.network_name)


def explore_issues():
    """Explore the source of difference between trj and veh record data"""
    vissim_reader = readers.VehicleRecordReader('highway_in_and_out_lanes',
                                                vehicle_type=
                                                VehicleType.CONNECTED)
    veh_record = vissim_reader.load_data(2)
    pp = post_processing
    pp.post_process_data(veh_record)
    ssm_estimator = post_processing.SSMEstimator(veh_record)
    ssm_estimator.include_ttc()

    conflicts = pd.read_csv(VissimInterface.get_networks_folder()
                            + '\\highway_in_and_out_lanes_001.csv')
    ttc_threshold = 1.5

    for i, row in conflicts.iterrows():
        if row['ConflictType'] != 'rear end':
            pass
        conflict_time = row['tMinTTC']
        conflict_ttc = row['TTC']
        # leader_idx = row['FirstVID']
        follower_idx = row['SecondVID']
        veh_record_ttc = veh_record.loc[
            (veh_record['time'] == conflict_time)
            & (veh_record['veh_id'] == follower_idx),
            'TTC2'].iloc[0]
        if veh_record_ttc < ttc_threshold:
            print('Conflict {} found in veh records\n'
                  '\tDelta TTC = {:.2f}'.
                  format(i + 1, conflict_ttc - veh_record_ttc))
        else:
            conflict_times_vr = veh_record[
                (veh_record['TTC2'] <= 1.5)
                & (veh_record['veh_id'] == follower_idx)]['time'].values
            if len(conflict_times_vr) > 0:
                if (conflict_times_vr[0] <= conflict_time
                        <= conflict_times_vr[-1]):
                    print('Conflict {} found nearby in veh records'.
                          format(i + 1))
                else:
                    print('Conflict {} at time {} not found,'
                          ' but veh has a conflict in the interval [{}, {}]'.
                          format(i + 1, conflict_time, conflict_times_vr[0],
                                 conflict_times_vr[-1]))
            else:
                print('Conflict {}: veh {} never has any conflicts'.
                      format(i + 1, follower_idx))

    """Conclusions after running on the toy scenario:
        1. Not all rear end conflicts found by the SSAM software can be found
        by our code that checks the vehicle record file.
        2. Even after looking at the trj file (the CSV equivalent of it), I
        could not figure out why SSAM considers those moments as conflicts.
        In other words, the TTC computed from the trj file still seems to be
        above the conflict threshold"""
    return veh_record
