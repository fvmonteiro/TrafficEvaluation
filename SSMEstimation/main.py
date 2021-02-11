import pandas as pd
import os
from classesForSSM import VissimInterface, DataReader, DataAnalyzer, OnLineDataReader, NGSIMDataReader


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    layout_file = 'highway_in_and_out_lanes'
    sim_params = {'SimPeriod': 200, 'RandSeed': 1}
    vi = VissimInterface(network_file, layout_file)
    vi.open_simulation()
    vi.run_toy_scenario(sim_params)


def run_i170_scenario(save_veh_record=False):
    network_file = "I710 - MultiSec - 3mi"
    layout_file = "I710 - MultiSec - 3mi"
    sim_resolution = 5  # No. of simulation steps per second
    simulation_time = 3600  # 5400  # 4000
    vi = VissimInterface(network_file, layout_file)
    vi.open_simulation()

    idx_scenario = 2  # 0: No block, 1: All time block, 2: Temporary block
    demands = [5500, ]
    # Vehicle Composition ID
    # 1: 10% Trucks
    # demandComposition = 2

    for demand in demands:
        random_seed = 1
        sim_params = {'SimPeriod': simulation_time, 'SimRes': sim_resolution, 'UseMaxSimSpeed': True,
                      'RandSeed': random_seed}
        vi.run_i710_simulation(sim_params, idx_scenario, demand, save_veh_record)


def post_process_and_save(network_file):
    data_analyzer = DataAnalyzer(VissimInterface.networks_folder, network_file, raw=True)
    data_analyzer.post_process_vissim_output()
    data_analyzer.save_to_csv(VissimInterface.networks_folder)


def create_all_ssm(network_file):
    data_analyzer = DataAnalyzer(VissimInterface.networks_folder, network_file, raw=False)

    print('Computing TTC')
    data_analyzer.include_ttc()
    for df in data_analyzer.veh_records.values():
        valid_ttc = df.loc[df['TTC'] < float('inf'), 'TTC']
        print('Mean TTC: {} for {} samples'.format(valid_ttc.mean(), valid_ttc.count()))
    print('Saving TTC')
    data_analyzer.save_to_csv(VissimInterface.networks_folder)

    print('Computing DRAC')
    data_analyzer.include_drac()
    for df in data_analyzer.veh_records.values():
        valid_drac = df.loc[df['DRAC'] > 0, 'DRAC']
        print('Mean DRAC: {} for {} samples'.format(valid_drac.mean(), valid_drac.count()))
    print('Saving DRAC')
    data_analyzer.save_to_csv(VissimInterface.networks_folder)

    print('Computing CPI')
    data_analyzer.include_cpi()
    for df in data_analyzer.veh_records.values():
        valid_cpi = df.loc[df['CPI'] > 0, 'CPI']
        print('Mean CPI: {} for {} samples'.format(valid_cpi.mean(), valid_cpi.count()))
    data_analyzer.save_to_csv(VissimInterface.networks_folder)
    print('Saving CPI')

    print('Computing Safe Gaps')
    data_analyzer.include_safe_gaps()
    for df in data_analyzer.veh_records.values():
        valid_safe_gap = df.loc[df['safe gap'] > 0, 'safe gap']
        print('Mean safe gap: {} for {} samples'.format(valid_safe_gap.mean(), valid_safe_gap.count()))
        risky_gaps = df.loc[df['DTSG'] < 0, 'DTSG']
        print('Mean unsafe gap: {} for {} samples'.format(risky_gaps.mean(), risky_gaps.count()))
    print('Saving Safe Gaps')
    data_analyzer.save_to_csv(VissimInterface.networks_folder)


def main():
    # run_i170_scenario(True)
    # network_file = "I710 - MultiSec - 3mi"
    # network_file = 'highway_in_and_out_lanes'
    # post_process_and_save(network_file)
    # create_all_ssm(network_file)


    return 0


if __name__ == '__main__':
    main()
