from classesForSSM import VissimInterface, SSMAnalyzer, DataReader
from vslControl import mkdir
import pandas as pd
import os


def run_toy_example():
    network_file = 'highway_in_and_out_lanes'
    vi = VissimInterface(network_file)
    vi.generate_data()
    data_reader = DataReader(VissimInterface.networks_folder, network_file)
    data_reader.load_data_from_vissim()
    data_reader.post_process_output()
    data_reader.get_single_dataframe()
    data_reader.save_to_csv()


def run_i170_scenario():
    network_file = "I710 - MultiSec - 3mi"
    layout_file = "I710 - MultiSec - 3mi"
    vi = VissimInterface(network_file, layout_file)

    idx_scenario = 1  # 0: No block 1: All time block
    # idxController and kLaneClosure
    ctrl = [(3, 1)]  # [(1, 0), (1, 2)]
    demands = [5500, ]
    simulation_time_sec = 300  # 5400  # 4000
    # Vehicle Composition ID
    # 1: 10% Trucks
    # demandComposition = 2

    for demand in demands:
        for jController, kLaneClosure in ctrl:
            folder_dir = os.path.join(VissimInterface.networks_folder, 'MicroResults')
            mkdir(folder_dir)
            random_seed = 1
            vi.run_simulation(simulation_time_sec, idx_scenario, kLaneClosure, random_seed, folder_dir, demand)


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
    run_i170_scenario()


if __name__ == '__main__':
    main()
