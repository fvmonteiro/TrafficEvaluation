import pandas as pd
import os
from classesForSSM import DataAnalyzer
import readers
from vissim_interface import VissimInterface


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


def post_process_and_save(data_source, network_name):
    if data_source.lower() == 'vissim':
        data_reader = readers.VissimDataReader(network_name)
    elif data_source.lower() == 'ngsim':
        data_reader = readers.NGSIMDataReader(network_name)
    else:
        print('Trying to process data from unknown data source')
        return

    print('Loading data from {}, network: {}'.format(data_source, network_name))
    data = data_reader.load_data()
    data_analyzer = DataAnalyzer(data_source, data)
    data_analyzer.post_process_data()
    data_analyzer.save_to_csv(network_name)


def create_all_ssm(data_source, network_name):
    if data_source.lower() == 'vissim':
        data_reader = readers.VissimDataReader(network_name)
    elif data_source.lower() == 'ngsim':
        data_reader = readers.NGSIMDataReader(network_name)
    else:
        print('Trying to process data from unknown data source')
        return

    data_analyzer = DataAnalyzer(data_reader.load_data())

    print('Computing TTC')
    data_analyzer.include_ttc()
    for df in data_analyzer.veh_records.values():
        valid_ttc = df.loc[df['TTC'] < float('inf'), 'TTC']
        print('Mean TTC: {} for {} samples'.format(valid_ttc.mean(), valid_ttc.count()))
    print('Saving TTC')
    data_analyzer.save_to_csv(network_name)

    print('Computing DRAC')
    data_analyzer.include_drac()
    for df in data_analyzer.veh_records.values():
        valid_drac = df.loc[df['DRAC'] > 0, 'DRAC']
        print('Mean DRAC: {} for {} samples'.format(valid_drac.mean(), valid_drac.count()))
    print('Saving DRAC')
    data_analyzer.save_to_csv(network_name)

    print('Computing CPI')
    data_analyzer.include_cpi()
    for df in data_analyzer.veh_records.values():
        valid_cpi = df.loc[df['CPI'] > 0, 'CPI']
        print('Mean CPI: {} for {} samples'.format(valid_cpi.mean(), valid_cpi.count()))
    data_analyzer.save_to_csv(network_name)
    print('Saving CPI')

    print('Computing Safe Gaps')
    data_analyzer.include_safe_gaps()
    for df in data_analyzer.veh_records.values():
        valid_safe_gap = df.loc[df['safe gap'] > 0, 'safe gap']
        print('Mean safe gap: {} for {} samples'.format(valid_safe_gap.mean(), valid_safe_gap.count()))
        risky_gaps = df.loc[df['DTSG'] < 0, 'DTSG']
        print('Mean unsafe gap: {} for {} samples'.format(risky_gaps.mean(), risky_gaps.count()))
    print('Saving Safe Gaps')
    data_analyzer.save_to_csv(network_name)


def main():
    # run_i170_scenario(True)
    # network_file = "I710 - MultiSec - 3mi"
    # network_file = 'highway_in_and_out_lanes'
    post_process_and_save('ngsim', 'us-101')
    # create_all_ssm(network_file)
    # da = DataAnalyzer('vissim', [])





if __name__ == '__main__':
    main()
