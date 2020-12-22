# attempt to use VISSIM COM interface through Pytyhon win32com ref : Mark Hammond, Andy Robinson (2000, Jan). Python
# Programming on Win32, [Online]. Available: http://www.icodeguru.com/WebServer/Python-Programming-on-Win32/ ref :
# VISSIM 5.30-04 - COM Interface Manual, PTV group, Karlsrueh, Germany, 2011.

import ctypes.wintypes
# import email_interface
import inspect
import os
import pywintypes
import win32api
import win32com.client as com
import shutil
import numpy as np
import random


# import win_interface

def get_program_name():
    # ref : In Python, how do I get the path and name of the file that is currently executing?, 2008,
    # [Online] Available at : http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of
    # -the-file-that-is-currently-executin
    program_full_name = inspect.getfile(inspect.currentframe())
    program_name = os.path.split(program_full_name)[-1]
    return program_name


def mkdir(path):
    """ recursively create given path """
    upper, name = os.path.split(path)
    if not os.path.exists(upper):
        mkdir(upper)
    # if path already exists, do not create. Otherwise, error will occur
    if not os.path.exists(path):
        os.mkdir(path)


def calc_random_seeds(n_monte_carlo):
    # list comprehension
    # same as {3i + 1 | 0 <= i < nMonteCarlo}
    return [3 * i + 1 for i in range(n_monte_carlo)]


def write_header(file_name, title, n_sec, first_column_header="t/s", column_header="Sec"):
    """write header line of a data file
    Parameters
    ----------
    file_name : string
        file name
    title : string
        first line to be written
    n_sec : int
        number of columns
    first_column_header : string
        header of the first column
    column_header : string
        header of each column. Repeated Nsec times with headers
        example : Nsec = 2, columnHeader = "Sec"
        -> "Sec1\tSec2\t"
    """
    log_file = open(file_name, 'w')
    # first line
    log_file.write(title)
    log_file.write('\n')
    # column headers
    #   first column header
    log_file.write(first_column_header)
    log_file.write('\t')

    column_header_template = column_header + "%d\t"
    # each column
    for i in range(n_sec):  # xrange
        log_file.write(column_header_template % (i + 1))
    log_file.write('\n')
    log_file.close()


def lc_control(scenario, links_obj, LC_distance):
    """Apply Lane Change control for specific scenario
    Parameters
    -------------
        scenario: dict
            the scenario dictionary
        links_obj: links object of VISSIM
            the links object
        LC_distance: int
        the distance of LC controlled section (m)

    """

    # link_id = scenario['link']
    link_id = 10010
    lane = scenario['lane']
    link_obj = links_obj.ItemByKey(link_id)
    for Lane in link_obj.Lanes:
        if Lane.AttValue('Index') == lane:
            Lane.SetAttValue('BlockedVehClasses', '10,20')


def vsl_feedback_linearization(density, speed, rho_star, vsl, section_length, link_groups, err_sum, perturbation):
    """
    density: (list[Nsec]) list of densities in all sections
    vsl: (list[Nsec]) current VSL command in each sections
    section_length: (list[Nsec]) length of each section
    """
    start_section = 0  # start_section: (int) the first section controlled with VSL
    end_section = 5
    # end_section: (int) the last section upstream the discharging section.
    # First downstream section of end_section is the discharging section.
    n_sec_controlled = end_section - start_section + 1
    # n_sec_controlled is the number of sections under control,
    # including the discharging section! The discharging section is considered as one single section

    wbar = 15 * (1 + perturbation)
    w = 30 * (1 + perturbation)
    C = 7200
    C_d = 4800
    vf = 100
    epsilon = 0.15
    rho_jbar = C / vf + C / wbar
    rho_j = C / vf + C / w
    vsl_min = 20
    vsl_max = 100
    rho_e = [rho_star] * (n_sec_controlled + 1)

    # incorporate uncertainties
    density1 = [(1 + 0) * x for x in density]

    rho = density1[start_section:(end_section + 2)]
    flow = np.multiply(density1, speed)
    q = flow[start_section:(end_section + 2)]
    x = np.subtract(rho, rho_e)
    for i in range(len(x)):
        err_sum[i] = err_sum[i] + x[i]

    # we need N+1 measurements to produce N VSL commands
    lambda1 = [200.0] * n_sec_controlled
    lambda2 = [2.0] * n_sec_controlled
    c = [-4000.0] * n_sec_controlled
    v = [0.0] * n_sec_controlled
    qv = [0.0] * n_sec_controlled
    for i in range(n_sec_controlled):
        qv[i] = lambda1[i] * x[i + 1] + lambda2[i] * (err_sum[i + 1] + c[i])

    for i in range(n_sec_controlled):
        if i == 0:
            v[i] = (q[i + 1] - qv[i]) * w / (w * rho_j - q[i + 1] + qv[i])
        elif rho[i] == 0:
            v[i] = vf
        else:
            v[i] = (q[i + 1] - qv[i]) / rho[i]

    for i in range(n_sec_controlled):
        v[i] = round(v[i] * 0.1) * 10
        if v[i] <= vsl[i + start_section]:
            v[i] = max(v[i], vsl[i + start_section] - 10, vsl_min)
        else:
            v[i] = min(v[i], vsl[i + start_section] + 10, vsl_max)
        if v[i] == 10:
            v[i] = 12
        if v[i] == 22:
            v[i] = 20
        vsl[i + start_section] = v[i]

    return vsl, err_sum


def run_simulation(simulation_time_sec, idx_scenario, idx_controller, idx_lane_closure, random_seed, folder_dir,
                   network_dir, demand, perturbation, sec1_len):
    """run PTV VISSIM simulation using given arguments
    Parameters
    ----------
        simulation_time_sec : int
            Simulation time (s)
        demand : int
            vehicle input (veh/hr)
        idx_scenario : int
            Index of simulation scenarios
        idx_controller : int
            Controller Index
        idx_lane_closure : int
            Whether Lane Change control is added
        random_seed : int
            seed for random numbers
        folder_dir : string, path
            location of data files
        network_dir : string, path
            location of network file
        sec1_len:
        perturbation:
    """

    # Link Groups
    link_groups = [
        # {'MAINLINE': (1,2,3,4,5,6),   'ONRAMP': (),      'OFFRAMP': (),     'DC': 1, 'VSL': (1,2,3,4,5,6,7,8,9,10,
        # 11,12,13,14,15,16,17,18)},
        {'MAINLINE': (1, 5, 6), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 1, 'VSL': (4, 5, 6, 13, 14, 15, 16, 17, 18)},
        {'MAINLINE': (7, 8), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 4, 'VSL': (19, 20, 21, 22, 23, 24)},
        {'MAINLINE': (10, 11), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 5, 'VSL': (25, 26, 27, 28, 29, 30)},
        {'MAINLINE': (12, 34), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 6, 'VSL': (31, 32, 33, 34, 35, 36)},
        {'MAINLINE': (17, 30), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 7, 'VSL': (37, 38, 39, 40, 41, 42)},
        {'MAINLINE': (16, 24), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 8, 'VSL': (43, 44, 45, 46, 47, 48)},
        {'MAINLINE': (13, 10010), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 9, 'VSL': (49, 50, 51)},
        {'MAINLINE': (9,), 'ONRAMP': (), 'OFFRAMP': (), 'DC': 10, 'VSL': (52, 53, 54)},
    ]

    '''setting of scenarios'''
    scenarios = [{'group': 8, 'link': 9, 'lane': 2, 'coordinate': 10, 'start_time_sec': 30, 'end_time_sec': 30},
                 {'group': 8, 'link': 9, 'lane': 2, 'coordinate': 10, 'start_time_sec': 30, 'end_time_sec': 3570},
                 {'group': 8, 'link': 9, 'lane': 2, 'coordinate': 10, 'start_time_sec': 600, 'end_time_sec': 4800}]

    n_sec = len(link_groups)  # number of sections
    sim_resolution = 5  # No. of simulation steps per second
    t_data_sec = 30.0  # Data sampling period
    t_ctrl_sec = 30.0  # Control time interval

    #   Incident time period
    start_time_sec = scenarios[idx_scenario]['start_time_sec']
    end_time_sec = scenarios[idx_scenario]['end_time_sec']

    vf = 100  # The free flow speed of highway in km/h
    # err_sum = 0
    rho_star = min(demand / vf, 45)

    # speed[]: average speed of each section
    # density[]: vehicle density of each section
    # flow_section[]: flow rate of each section
    # flowLane[]: flow rate of each lane at bottleneck section
    # vsl[]: vsl command of each section at current control period, a list of dicts
    # vslOld[]: vsl command of each section at previous control period
    speed = [0.0] * n_sec
    density = [0.0] * n_sec
    flow_section = [0.0] * n_sec
    err_sum = [0.0] * n_sec
    vsl = [vf] * n_sec

    '''Define log file names'''
    flow_file_name = os.path.join(folder_dir, "flow_Log.txt")
    den_file_name = os.path.join(folder_dir, "den_Log.txt")
    vsl_file_name = os.path.join(folder_dir, "vsl_Log.txt")
    vtt_file_name = os.path.join(folder_dir, "vtt_Log.txt")

    prog_id = "VISSIM.Vissim.1000"

    '''file paths'''
    network_file_name = "I710 - MultiSec - 3mi.inpx"
    layout_file_name = "I710 - MultiSec - 3mi.layx"

    ''' Start VISSIM simulation '''
    ''' COM lines'''

    try:
        print("Client: Creating a Vissim instance")
        vs = com.Dispatch(prog_id)

        print("Client: read network and layout")
        vs.LoadNet(os.path.join(network_dir, network_file_name), 0)
        vs.LoadLayout(os.path.join(network_dir, layout_file_name))

        ''' initialize simulations '''
        ''' setting random seed, simulation time, simulation resolution, simulation speed '''
        print("Client: Setting simulations")
        vs.Simulation.SetAttValue('SimRes', sim_resolution)
        vs.Simulation.SetAttValue('UseMaxSimSpeed', True)
        vs.Simulation.SetAttValue('RandSeed', random_seed)

        # Reference to road network
        net = vs.Net

        ''' Set vehicle input '''
        veh_inputs = net.VehicleInputs
        veh_input = veh_inputs.ItemByKey(1)
        veh_input.SetAttValue('Volume(1)', demand)

        ''' Set default speed limit and 1st sign location'''
        VSLs = net.DesSpeedDecisions

        for VSL in VSLs:
            VSL.SetAttValue("DesSpeedDistr(10)", vf)  # car (km/h)
            VSL.SetAttValue("DesSpeedDistr(20)", vf)  # HGV (km/h)

        pos = 3203 - (sec1_len - 1) * 1600
        VSLs.ItemByKey(4).SetAttValue("Pos", pos)
        VSLs.ItemByKey(5).SetAttValue("Pos", pos)
        VSLs.ItemByKey(6).SetAttValue("Pos", pos)

        ''' Get Data collection and link objects '''
        data_collections = net.DataCollectionMeasurements
        links = net.Links
        vtt_container = net.VehicleTravelTimeMeasurements

        # Make sure that all lanes are open
        link_n_lane = {}
        link_length = {}

        for link in links:
            id = link.AttValue('No')
            link_n_lane[id] = link.AttValue('NumLanes')
            link_length[id] = link.AttValue('Length2D')  # meter
            # Open all lanes
            for lane in link.Lanes:
                lane.SetAttValue('BlockedVehClasses', '')

        section_length = [0.0] * n_sec  # length of each section
        for iSec in range(len(link_groups)):  # xrange
            for linkID in link_groups[iSec]['MAINLINE']:
                link = links.ItemByKey(linkID)
                section_length[iSec] += link.AttValue('Length2D')  # meter

        bus_no = 2  # No. of buses used to block the lane
        bus_array = [0] * bus_no

        vehs = net.Vehicles

        # start simulation
        print("Client: Starting simulation")
        print("Scenario:", idx_scenario)
        print("Controller:", idx_controller)
        print("Lane Closure:", idx_lane_closure, "section")
        print("Random Seed:", random_seed)

        current_time = 0

        # vs.Simulation.SetAttValue("SimBreakAt", 200)
        # vs.Simulation.RunContinuous()

        '''Link Segment Results'''

        while current_time <= simulation_time_sec:
            # run sigle step of simulation
            vs.Simulation.RunSingleStep()
            current_time = vs.Simulation.AttValue('SimSec')

            if 0 == current_time % t_data_sec:
                # Get density, flow, speed of each section
                for iSection in range(n_sec):  # xrange
                    # Flow volume in one sampling period
                    data_collection = data_collections.ItemByKey(link_groups[iSection]['DC'])
                    flow_section[iSection] = data_collection.AttValue('Vehs(Current,Last,All)')

                    # Density and Speed in each section
                    n_veh = 0
                    dist = 0
                    denominator = 0
                    for linkID in link_groups[iSection]['MAINLINE']:
                        link = links.ItemByKey(linkID)
                        link_vehs = link.vehs
                        num_vehs = link_vehs.count
                        sum_speed = 0
                        if num_vehs == 0:
                            v = 0
                        else:
                            for link_veh in link_vehs:
                                sum_speed += link_veh.AttValue('Speed')
                            v = sum_speed / num_vehs
                        n_veh += link_vehs.count
                        dist += v * link_vehs.count  # total distance traveled by all vehicles in one link
                        denominator += link_length[linkID]
                    density[iSection] = 1000 * n_veh / denominator  # number of vehicles per km
                    if 0 == n_veh:
                        speed[iSection] = 0
                    else:
                        speed[iSection] = dist / n_veh  # km/h

                ''' vehicle travel time '''
                vtt = vtt_container.ItemByKey(1)
                trav_tm = vtt.AttValue('TravTm(Current,Last,All)')

                ''' write log files : flow_section, speed, density '''

                den_file = open(den_file_name, "a")
                flow_file = open(flow_file_name, "a")
                vtt_file = open(vtt_file_name, "a")

                for i in range(n_sec):
                    den_file.write(str(density[i]) + '\t')
                    flow_file.write(str(density[i] * speed[i]) + '\t')
                vtt_file.write(str(trav_tm) + '\t')

                den_file.write('\n')
                den_file.close()
                flow_file.write('\n')
                flow_file.close()
                vtt_file.write('\n')
                vtt_file.close()

            if current_time == start_time_sec:
                # set incident scenario
                for i in range(bus_no):  # xrange
                    bus_array[i] = vehs.AddVehicleAtLinkPosition(300, scenarios[idx_scenario]['link'],
                                                                 scenarios[idx_scenario]['lane'],
                                                                 scenarios[idx_scenario]['coordinate'] + 20 * i, 0, 0)
                # Apply Lane change control
                if idx_controller == 2 or idx_controller == 3:
                    lc_control(scenarios[idx_scenario], links, 600 * idx_lane_closure)
                # Set cMode

            if current_time == end_time_sec:
                # Remove the incident
                for veh in bus_array:
                    vehs.RemoveVehicle(veh.AttValue('No'))

                # open all closed lanes
                for link in links:
                    for lane in link.Lanes:
                        lane.SetAttValue('BlockedVehClasses', '')

            # Compute the vsl command
            if 0 == current_time % t_ctrl_sec:
                if (idx_controller == 3 or idx_controller == 1) and (start_time_sec <= current_time < end_time_sec):
                    vsl, err_sum = vsl_feedback_linearization(density, speed, rho_star, vsl, section_length,
                                                              link_groups,
                                                              err_sum, perturbation)
                else:
                    for iVSL in range(n_sec):  # xrange
                        vsl[iVSL] = vf

                # Update vsl command in VISSIM
                for iSec in range(n_sec):  # xrange
                    for vslID in link_groups[iSec]["VSL"]:
                        VSL = VSLs.ItemByKey(vslID)
                        VSL.SetAttValue("DesSpeedDistr(10)", vsl[iSec])  # car
                        VSL.SetAttValue("DesSpeedDistr(20)", vsl[iSec])  # HGV

                vsl_file = open(vsl_file_name, "a")
                for i in range(n_sec):
                    vsl_file.write(str(vsl[i]) + '\t')
                vsl_file.write('\n')
                vsl_file.close()


    except pywintypes.com_error as err:
        vs = None
        print("err=", err)

        '''
        Error
        The specified configuration is not defined within VISSIM.

        Description
        Some methods for evaluations results need a previously configuration for data collection. 
        The error occurs when requesting results that have not been previously configured.
        For example, using the GetSegmentResult() method of the ILink interface to request
        density results can end up with this error if the density has not been requested within the configuration
        '''


def run_scenario():
    idx_scenario = 2  # 0: NoBlock 1: AlltimeBlock
    # idxController and kLaneClosure
    ctrl = [(3, 1)]  # [(1, 0), (1, 2)]
    demands = [5500, ]
    perturbations = [0, ]
    sec1_lens = [1.0, ]  # [1.0, 1.1, 1.2, 1.3, 1.4]  # Distance btw first two signs, must be in [1 , 3] (mi)

    simulation_time_sec = 100  # 5400  # 4000
    # nMonteCarlo = 3
    # randomSeeds = calcRandomSeeds(nMonteCarlo)

    # Vehicle Composition ID
    # 1: 10% Trucks

    # demandComposition = 2

    network_dir = 'C:\\Users\\fvall\\Documents\\Research\\TrafficSimulation\\VISSIM_networks'
    # os.path.abspath(os.curdir)
    for demand in demands:
        for jController, kLaneClosure in ctrl:
            folder_dir = os.path.join(network_dir, 'MicroResults')
            mkdir(folder_dir)
            result_file = open(os.path.join(folder_dir, "result.txt"), 'w')
            for perturbation in perturbations:
                for sec1_len in sec1_lens:
                    l_mc = round(random.uniform(1, 100))
                    run_simulation(simulation_time_sec, idx_scenario, jController, kLaneClosure, l_mc, folder_dir,
                                   network_dir, demand, perturbation, sec1_len)
