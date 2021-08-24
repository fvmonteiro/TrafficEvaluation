import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import result_analysis
from data_writer import SyntheticDataWriter
import readers
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
    post_processed_dir = readers.PostProcessedDataReader.post_processed_dir
    data_dir = os.path.join(post_processed_dir, data_source)
    data.to_csv(os.path.join(data_dir, simulation_name + '.csv'), index=False)
    print('Data saved at: \n\tFolder: {:s}\n\tName:: {:s}'.
          format(data_dir, simulation_name))


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    sim_params = {'SimPeriod': 200, 'RandSeed': 1}
    vi = VissimInterface()
    if not vi.load_simulation(network_file):
        return
    vi.set_evaluation_outputs(False, False, False, False)
    vi.set_simulation_parameters(sim_params)
    vi.run_toy_scenario()
    vi.close_vissim()


def run_i170_scenario(save_results=False):
    network_file = "I710-MultiSec-3mi"
    sim_resolution = 5  # No. of simulation steps per second
    simulation_time = 3600  # 5400  # 4000
    vi = VissimInterface()
    if not vi.load_simulation(network_file):
        return
    vi.set_evaluation_outputs(save_results, save_results,
                              save_results, save_results)
    idx_scenario = 2  # 0: No block, 1: All time block, 2: Temporary block
    demands = [5500, ]
    # Vehicle Composition ID
    # 1: 10% Trucks
    # demandComposition = 2

    for demand in demands:
        random_seed = 1
        sim_params = {'SimPeriod': simulation_time, 'SimRes': sim_resolution,
                      'UseMaxSimSpeed': True, 'RandSeed': random_seed}
        vi.set_simulation_parameters(sim_params)
        vi.run_i710_simulation(idx_scenario, demand)


def post_process_and_save(data_source, network_name):
    if data_source.upper() == 'VISSIM':
        data_reader = readers.VehicleRecordReader(network_name)
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
    data_pp = result_analysis.VehicleRecordPostProcessor(data_source, data)
    data_pp.post_process_data()
    print('Post processed data shape: ', data.shape)
    save_post_processed_data(data, data_source,
                             data_reader.file_name)


def create_ssm(data_source, network_name, ssm_names):
    if isinstance(ssm_names, str):
        ssm_names = [ssm_names]

    data_reader = readers.PostProcessedDataReader(data_source, network_name)
    data = data_reader.load_data()
    ssm_estimator = result_analysis.SSMEstimator(data)

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


def create_and_save_synthetic_data():
    vx = 20
    delta_v = 0
    writer = SyntheticDataWriter()
    data = writer.create_data(vx, delta_v)
    writer.write_data(data)


def test_safe_gap_computation():
    # create_and_save_synthetic_data()
    data_source = 'synthetic'
    # post_process_and_save('synthetic_data', network_file)
    data_reader = readers.PostProcessedDataReader(data_source)
    data = data_reader.load_data()

    gamma = 1 / 0.8
    rho = 0.2
    ssm_estimator = result_analysis.SSMEstimator(data)
    ssm_estimator.include_exact_risk(same_type_gamma=gamma)
    ssm_estimator.include_estimated_risk(rho=rho, same_type_gamma=gamma)
    risk_gap = data.loc[data['delta_x'] <= (data['safe_gap'] + 0.02),
                        'delta_x']
    risk = data.loc[data['delta_x'] <= (data['safe_gap'] + 0.02), 'exact_risk']
    estimated_risk_gap = data.loc[data['delta_x'] <= (data['vf_gap'] + 0.02),
                                  'delta_x']
    estimated_risk = data.loc[data['delta_x'] <= (data['vf_gap'] + 0.02),
                              'estimated_risk']

    # Plot
    fig, ax = plt.subplots()
    ax.plot(risk_gap, risk)
    ax.plot(estimated_risk_gap, estimated_risk)
    plt.show()
    # save_post_processed_data(ssm_estimator.veh_data,
    #                          data_reader.data_source, data_reader.file_name)


def explore_issues():
    """Explore the source of difference between trj and veh record data"""
    vissim_reader = readers.VehicleRecordReader('highway_in_and_out_lanes')
    veh_record = vissim_reader.load_data()
    pp = result_analysis.VehicleRecordPostProcessor('vissim', veh_record)
    pp.post_process_data()
    ssm_estimator = result_analysis.SSMEstimator(veh_record)
    ssm_estimator.include_ttc()

    conflicts = pd.read_csv(VissimInterface.networks_folder
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


def main():
    # image_folder = "G:\\My Drive\\Safety in Mixed Traffic\\images"

    # Define data source #
    network_file = 'toy'  # Options: i710, us-101, toy

    # Running #
    # run_i170_scenario(True)
    # run_toy_example()
    # generate_us_101_reduced_speed_table(1)
    # vi = VissimInterface()
    # vi.load_simulation('toy')
    # vi.run_with_increasing_autonomous_penetration(
    #     autonomous_percentage_increase=100,
    #     initial_autonomous_percentage=0,
    #     final_autonomous_percentage=100)
    # result_analyzer = result_analysis.ResultAnalyzer(network_file)
    # for autonomous_percentage in [0, 100]:
    #     result_analyzer.vehicle_record_to_ssm_summary(autonomous_percentage)

    # SSM computation check #
    # veh_rec_reader = readers.VehicleRecordReader('toy')
    # veh_rec = veh_rec_reader.load_data(51, 100)
    # pp = result_analysis.VehicleRecordPostProcessor('vissim', veh_rec)
    # pp.post_process_data()
    #
    # ssm_estimator = result_analysis.SSMEstimator(veh_rec)
    # ssm_estimator.include_collision_free_gap()

    # Post processing #
    # post_process_and_save(data_source, network_file)

    # Save all SSMs #
    # ssm_names = ['TTC', 'DRAC', 'collision_free_gap',
    #              'vehicle_following_gap', 'CPI',
    #              'exact_risk', 'estimated_risk']
    # ssm_names = ['CPI']
    # create_ssm(data_source, network_file, ssm_names)

    # Check results graphically #
    result_analyzer = result_analysis.ResultAnalyzer(network_file)
    all_percentages = [i for i in range(0, 101, 25)]
    # '100_percent_autonomous_only_longitudinal_control'
    # for veh_input in range(2000, 2001, 500):
    #     result_analyzer.plot_variable_vs_time(
    #         'flow', veh_input,
    #         all_percentages,
    #         start_time=5)
    #     result_analyzer.plot_variable_vs_time(
    #         'exact_risk', veh_input,
    #         all_percentages,
    #         start_time=5)
    # result_analyzer.plot_double_y_axes(0, x='density', y=['flow',
    #                                                       'exact_risk'])
    result_analyzer.plot_with_labels([0, 100], x='density',
                                     y='flow')
    # reader = readers.PostProcessedDataReader(data_source, network_file)
    # data = reader.load_data()
    # ssm_estimator = SSMEstimator(data)
    # # ssm_estimator.plot_ssm('low_TTC')
    # # ssm_estimator.plot_ssm('exact_risk')
    # # ssm_estimator.plot_ssm('estimated_risk')
    # # ssm_estimator.plot_ssm('vx')
    # image_path = os.path.join(image_folder, network_file)
    # ssm_estimator.plot_ssm_moving_average('low_TTC', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('high_DRAC', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('CPI', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('exact_risk', 500, image_path)
    # ssm_estimator.plot_ssm_moving_average('estimated_risk')
    # ssm_estimator.plot_ssm_moving_average('vx')


if __name__ == '__main__':
    main()
