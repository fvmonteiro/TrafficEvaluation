import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import post_processing
import result_analysis
from data_writer import SyntheticDataWriter
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
    vi.run_in_and_out_scenario()
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
    data_pp = post_processing.DataPostProcessor(data_source)
    data_pp.post_process_data(data)
    print('Post processed data shape: ', data.shape)
    save_post_processed_data(data, data_source,
                             data_reader.network_name)


def create_ssm(data_source, network_name, ssm_names):
    if isinstance(ssm_names, str):
        ssm_names = [ssm_names]

    data_reader = readers.PostProcessedDataReader(data_source, network_name)
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
    ssm_estimator = post_processing.SSMEstimator(data)
    ssm_estimator.include_risk(same_type_gamma=gamma)
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
    vissim_reader = readers.VehicleRecordReader('highway_in_and_out_lanes',
                                                vehicle_type=
                                                VehicleType.CONNECTED)
    veh_record = vissim_reader.load_data(2)
    pp = post_processing.DataPostProcessor('vissim')
    pp.post_process_data(veh_record)
    ssm_estimator = post_processing.SSMEstimator(veh_record)
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

    # =============== Define data source =============== #
    # Options: i710, us-101, in_and_out, in_and_merge
    network_file = VissimInterface.network_names_map['in_and_out']
    vehicle_type = VehicleType.AUTONOMOUS

    # =============== Temporary tests  =============== #
    # ra = result_analysis.ResultAnalyzer(network_file, vehicle_type)
    # ra.find_unfinished_simulations(100)
    # =============== Running =============== #

    # percentage_increase = 150
    # initial_percentage = 0
    # final_percentage = 100
    # vi = VissimInterface()
    # vi.load_simulation(network_file)
    # vi.run_with_increasing_controlled_vehicle_percentage(
    #     vehicle_type,
    #     percentage_increase=100,
    #     initial_percentage=100,
    #     final_percentage=100,
    #     input_increase_per_lane=1000,
    #     initial_input_per_lane=1000,
    #     max_input_per_lane=2000)
    # vi.run_with_increasing_controlled_vehicle_percentage(
    #     vehicle_type,
    #     percentage_increase=25,
    #     initial_percentage=25,
    #     final_percentage=75,
    #     input_increase_per_lane=500,
    #     initial_input_per_lane=2000,
    #     max_input_per_lane=2000)
    # vi.close_vissim()

    # =============== Post processing =============== #

    # post_processor = post_processing.DataPostProcessor()
    # for percentage in range(100, 100+1, 100):
    #     post_processor.create_ssm_summary(network_file,
    #                                       vehicle_type,
    #                                       percentage)
    #     post_processor.merge_data(network_file, vehicle_type, percentage)

    # =============== Check results graphically =============== #

    vehicle_types = [VehicleType.ACC,
                     VehicleType.AUTONOMOUS,
                     VehicleType.CONNECTED]
    result_analyzer = result_analysis.ResultAnalyzer('in_and_out',
                                                     vehicle_types)

    percentages = [0, 100]
    for veh_input in [2000]:
        result_analyzer.plot_y_vs_time('flow', veh_input, percentages,
                                       warmup_time=5)
        result_analyzer.plot_y_vs_time('risk', veh_input, [100],
                                       warmup_time=5)
        # result_analyzer.plot_y_vs_time('risk_no_lane_change', veh_input,
        #                                percentages, warmup_time=5)
    veh_inputs = [i for i in range(1000, 2001, 1000)]
    result_analyzer.box_plot_y_vs_controlled_percentage('flow', veh_inputs,
                                                        percentages,
                                                        warmup_time=10)
    result_analyzer.box_plot_y_vs_controlled_percentage('risk', veh_inputs,
                                                        [100],
                                                        warmup_time=10)
    # result_analyzer.plot_y_vs_controlled_percentage(
    #     'risk_no_lane_change', 2000, percentages, warmup_time=10)

    # result_analyzer.plot_double_y_axes(0, x='density', y=['flow',
    #                                                       'exact_risk'])
    # result_analyzer.scatter_plot(x='average_speed', y='risk',
    #                              controlled_percentage=100,
    #                              warmup_time=10)

    # =============== SSM computation check =============== #
    # veh_rec_reader = readers.VehicleRecordReader('toy')
    # veh_rec = veh_rec_reader.load_data(51, 100)
    # pp = result_analysis.VehicleRecordPostProcessor('vissim', veh_rec)
    # pp.post_process_data()
    #
    # ssm_estimator = result_analysis.SSMEstimator(veh_rec)
    # ssm_estimator.include_collision_free_gap()

    # Save all SSMs #
    # ssm_names = ['TTC', 'DRAC', 'collision_free_gap',
    #              'vehicle_following_gap', 'CPI',
    #              'exact_risk', 'estimated_risk']
    # ssm_names = ['CPI']
    # create_ssm(data_source, network_file, ssm_names)

    # =============== SSM tests =============== #
    # reader = readers.VehicleRecordReader('toy')
    # veh_record = reader.load_data(1, 'test')
    # pp = result_analysis.VehicleRecordPostProcessor('vissim', veh_record)
    # pp.post_process_data()
    # ssm_estimator = result_analysis.SSMEstimator(veh_record)
    # ssm_estimator.include_collision_free_gap()
    # ssm_estimator.include_exact_risk()
    # print('done')
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
