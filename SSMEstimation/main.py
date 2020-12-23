from classesForSSM import VissimInterface, SSMAnalyzer, DataReader
import pandas as pd
import os


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    layout_file = 'highway_in_and_out_lanes'
    sim_params = {'SimPeriod': 200, 'RandSeed': 1}
    vi = VissimInterface(network_file, layout_file)
    vi.open_simulation()
    vi.run_toy_scenario(sim_params)
    # data_reader = DataReader(VissimInterface.networks_folder, network_file)
    # data_reader.load_data_from_vissim()
    # data_reader.post_process_output()
    # data_reader.get_single_dataframe()
    # data_reader.save_to_csv()


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


def create_all_ssm():
    network_file = 'highway_in_and_out_lanes'
    data_reader = DataReader(VissimInterface.networks_folder, network_file)
    data_reader.load_data_from_csv()
    data_reader.load_max_decel_data()
    # SSMAnalyzer.include_ttc(data_reader.sim_output)
    # SSMAnalyzer.include_drac(data_reader.sim_output)
    # SSMAnalyzer.include_cpi(data_reader.sim_output, data_reader.max_decel)
    # SSMAnalyzer.include_safe_gaps(data_reader.sim_output, data_reader.max_decel)

    # TODO: Check these cases in VISSIM
    # df = data_reader.get_single_dataframe()
    # print(df[df['CPI'] > 0])


def main():
    # network_file = "I710 - MultiSec - 3mi"
    network_file = 'highway_in_and_out_lanes'
    data_reader = DataReader(VissimInterface.networks_folder, network_file)
    data_reader.load_data_from_vissim()
    SSMAnalyzer.include_drac(data_reader.sim_output)
    df = data_reader.get_single_dataframe()
    df.info()
    print('DRAC' in df.columns)


if __name__ == '__main__':
    main()
