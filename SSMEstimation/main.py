from classesForSSM import VissimInterface, SSMAnalyzer, DataReader
import pandas as pd
import os


def main():
    vi = VissimInterface()
    # TODO: rerun simulation and do not save SpeedDiff
    # vi.generate_data()

    data_reader = DataReader(vi.vissim_networks_folder, vi.network_file)
    data_reader.load_data_from_csv()
    data_reader.load_max_decel_data()

    SSMAnalyzer.include_ttc(data_reader.sim_output)
    SSMAnalyzer.include_drac(data_reader.sim_output)
    SSMAnalyzer.include_cpi(data_reader.sim_output, data_reader.max_decel)

    # TODO: Check these cases in VISSIM
    # df = data_reader.get_single_dataframe()
    # print(df[df['CPI'] > 0])

    # vi.close_vissim()


if __name__ == '__main__':
    main()
