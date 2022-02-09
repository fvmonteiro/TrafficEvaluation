# attempt to use VISSIM COM interface through Pytyhon win32com ref : Mark
# Hammond, Andy Robinson (2000, Jan). Python Programming on Win32, [Online].
# Available: http://www.icodeguru.com/WebServer/Python-Programming-on-Win32/
# ref : VISSIM 5.30-04 - COM Interface Manual, PTV group, Karlsrueh, Germany,
# 2011.

import ctypes.wintypes
#import email_interface
import inspect
import os
import pywintypes
import win32api
import win32com.client as com
import shutil
import numpy as np
import random
#import win_interface

def getProgramName():
    # ref : In Python, how do I get the path and name of the file that is
    # currently executing?, 2008, [Online] Available at :
    # http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the
    # -path-and-name-of-the-file-that-is-currently-executin
    PROGRAM_FULL_NAME = inspect.getfile(inspect.currentframe())
    PROGRAM_NAME = os.path.split(PROGRAM_FULL_NAME)[-1]
    return PROGRAM_NAME

def mkdir(path):
    """ recursively create given path """
    upper, name = os.path.split(path)
    if not os.path.exists(upper):
        mkdir(upper)
    # if path already exists, do not create. Otherwise, error will occur
    if not os.path.exists(path):
        os.mkdir(path)


def calcRandomSeeds(nMonteCarlo):
    # list comprehension
    # same as {3i + 1 | 0 <= i < nMonteCarlo}
    return [3*i + 1 for i in range(nMonteCarlo)]


def writeHeader(fileName, title, Nsec, firstColumnHeader = "t/s", columnHeader = "Sec"):
    """write header line of a data file
    Parameters
    ----------
    fileName : string
        file name
    title : string
        first line to be written
    Nsec : int
        number of columns
    firstColumnHeader : string
        header of the first column
    columnHeader : string
        header of each column. Repeated Nsec times with headers
        example : Nsec = 2, columnHeader = "Sec"
        -> "Sec1\tSec2\t"
    """
    LogFile = open(fileName, 'w')
    # first line
    LogFile.write(title)
    LogFile.write('\n')
    # column headers
    #   first column header
    LogFile.write(firstColumnHeader)
    LogFile.write('\t')

    columnHeaderTemplate = columnHeader+"%d\t"
    # each column
    for i in range(Nsec): #xrange
        LogFile.write(columnHeaderTemplate % (i + 1))
    LogFile.write('\n')
    LogFile.close()

def LC_Control(scenario, links_obj, LC_distance):
    """Apply Lane Change control for specific scenario
    Parameters
    -------------
        scenario: dict
            the scenario dictionary
        links_obj: links object of VISSIM
            the links object
        link_groups: list
            the list of link groups
        ramp_links: list
            the list of links with off ramp
        LC_distance: int
        the distance of LC controlled section (m)

    """

    link_ID = scenario['link']
    link_ID = 10010
    lane = scenario['lane']
    link_obj = links_obj.ItemByKey(link_ID)
    for Lane in link_obj.Lanes:
        if Lane.AttValue('Index') == lane:
            Lane.SetAttValue('BlockedVehClasses', '10,20')


def vsl_FeedbackLinearization(density, speed, rho_star, vsl, section_length, link_groups, err_sum, perturbation):
    """
    density: (list[Nsec]) list of densities in all sections
    vsl: (list[Nsec]) current VSL command in each sections
    section_length: (list[Nsec]) length of each section

    """
    startSection = 0    #startSection: (int) the first section controlled with VSL
    endSection = 5
    #endSection: (int) the last section upstream the discharging section.
    #First downstream section of endSection is the discharging section.
    Nsec_controlled = endSection - startSection + 1
    #Nsec_controlled is the number of sections under control,
    #including the discharging section! The discharging section is considered as one single section

    wbar = 15*(1+perturbation)
    w = 30*(1+perturbation)
    C = 7200
    C_d = 4800
    vf = 100
    epsilon = 0.15
    rho_jbar = C/vf + C/wbar
    rho_j = C/vf + C/w
    vslMIN = 20
    vslMAX = 100
    rho_e = [rho_star]*(Nsec_controlled+1)

    #incorporate uncertainties
    density1 = [(1+0)*x for x in density]

    rho = density1[startSection:(endSection+2)]
    flow = np.multiply(density1,speed)
    q = flow[startSection:(endSection+2)]
    x = np.subtract(rho,rho_e)
    for i in range(len(x)):
        err_sum[i] = err_sum[i] + x[i]

    # we need N+1 measurements to produce N VSL commands
    Lambda1 = [200.0]*(Nsec_controlled)
    Lambda2 = [2.0]*(Nsec_controlled)
    c = [-4000.0]*(Nsec_controlled)
    v = [0.0]*(Nsec_controlled)
    qv = [0.0]*(Nsec_controlled)
    for i in range(Nsec_controlled):
        qv[i] = Lambda1[i]*x[i+1] + Lambda2[i]*(err_sum[i+1] + c[i])

    for i in range(Nsec_controlled):
        if i == 0:
            v[i] = (q[i+1] - qv[i]) * w / (w*rho_j - q[i+1] + qv[i])
        elif rho[i] == 0:
            v[i] == vf
        else:
            v[i] = (q[i+1] - qv[i]) / rho[i]


    for i in range(Nsec_controlled):
        v[i] = round(v[i] * 0.1) * 10
        if v[i] <= vsl[i + startSection]:
            v[i] = max(v[i], vsl[i + startSection] - 10, vslMIN)
        else:
            v[i] = min(v[i], vsl[i + startSection] + 10, vslMAX)
        if v[i] == 10:
            v[i] = 12
        if v[i] == 22:
            v[i] = 20
        vsl[i + startSection] = v[i]

    return vsl, err_sum






def runSimulation(simulationTime_sec, idxScenario, idxController, idxLaneClosure, randomSeed, folderDir, networkDir, demand, perturbation, sec1len):
    """run PTV VISSIM simulation using given arguments
    Parameters
    ----------
        simulationTime_sec : int
            Simulation time (s)
        startTime_sec : int
            Time accident starts (s)
        endTime_sec : int
            Time accident ends (s)
        vehicleInput_veh_hr : int
            vehicle input (veh/hr)
        idxScenario : int
            Index of simulation scenarios
        idxController : int
            Controller Index
        idxLaneClosure : int
            Whether Lane Change control is added
        randomSeed : int
            seed for random numbers
        folderDir : string, path
            location of data files
        networkDir : string, path
            location of network file
    """

    # Link Groups
    link_groups = [
                #{'MAINLINE': (1,2,3,4,5,6),   'ONRAMP': (),      'OFFRAMP': (),     'DC': 1, 'VSL': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)},
                {'MAINLINE': (1,5,6),         'ONRAMP': (),      'OFFRAMP': (),     'DC': 1, 'VSL': (4,5,6,13,14,15,16,17,18)},
                {'MAINLINE': (7,8),           'ONRAMP': (),      'OFFRAMP': (),     'DC': 4, 'VSL': (19,20,21,22,23,24)},
                {'MAINLINE': (10,11),         'ONRAMP': (),      'OFFRAMP': (),     'DC': 5, 'VSL': (25,26,27,28,29,30)},
                {'MAINLINE': (12,34),         'ONRAMP': (),      'OFFRAMP': (),     'DC': 6, 'VSL': (31,32,33,34,35,36)},
                {'MAINLINE': (17,30),         'ONRAMP': (),      'OFFRAMP': (),     'DC': 7, 'VSL': (37,38,39,40,41,42)},
                {'MAINLINE': (16,24),         'ONRAMP': (),      'OFFRAMP': (),     'DC': 8, 'VSL': (43,44,45,46,47,48)},
                {'MAINLINE': (13,10010),      'ONRAMP': (),      'OFFRAMP': (),     'DC': 9, 'VSL': (49,50,51)},
                {'MAINLINE': (9,),            'ONRAMP': (),      'OFFRAMP': (),     'DC': 10, 'VSL': (52,53,54)},
    ]


    '''setting of scenarios'''
    scenarios = [{'group': 8, 'link': 9, 'lane': 2, 'coordinate': 10, 'startTime_sec': 30, 'endTime_sec': 30},
                 {'group': 8, 'link': 9, 'lane': 2, 'coordinate': 10, 'startTime_sec': 30, 'endTime_sec': 3570},
                 {'group': 8, 'link': 9, 'lane': 2, 'coordinate': 10, 'startTime_sec': 600, 'endTime_sec': 4800}]


    Nsec = len(link_groups)   # number of sections
    simResolution = 5   # No. of simulation steps per second
    Tdata_sec = 30.0     # Data sampling period
    Tctrl_sec = 30.0 # Control time interval


    #   Incident time period
    startTime_sec = scenarios[idxScenario]['startTime_sec']
    endTime_sec = scenarios[idxScenario]['endTime_sec']


    vf = 100 # The free flow speed of highway in km/h
    #err_sum = 0
    rho_star = min(demand/vf,45)

    # speed[]: average speed of each section
    # density[]: vehicle density of each section
    # flowSection[]: flow rate of each section
    # flowLane[]: flow rate of each lane at bottleneck section
    # vsl[]: vsl command of each section at current control period, a list of dicts
    # vslOld[]: vsl command of each section at previous control period
    speed = [0.0] * Nsec
    density = [0.0] * Nsec
    flowSection = [0.0] * Nsec
    err_sum = [0.0] * Nsec
    vsl = [vf] * Nsec



    '''Define log file names'''
    FlowFileName = os.path.join(folderDir, "flow_Log.txt")
    DenFileName = os.path.join(folderDir, "den_Log.txt")
    VSLFileName = os.path.join(folderDir, "vsl_Log.txt")
    VTTFileName = os.path.join(folderDir, "vtt_Log.txt")


    ProgID = "VISSIM.Vissim.1000"

    '''file paths'''
    networkFileName = "I710 - MultiSec - 3mi.inpx"
    layoutFileName = "I710 - MultiSec - 3mi.layx"

    ''' Start VISSIM simulation '''
    ''' COM lines'''

    try:
        print("Client: Creating a Vissim instance")
        vs = com.Dispatch(ProgID)

        print("Client: read network and layout")
        vs.LoadNet(os.path.join(networkDir, networkFileName), 0)
        vs.LoadLayout(os.path.join(networkDir, layoutFileName))

        ''' initialize simulations '''
        ''' setting random seed, simulation time, simulation resolution, simulation speed '''
        print("Client: Setting simulations")
        vs.Simulation.SetAttValue('SimRes', simResolution)
        vs.Simulation.SetAttValue('UseMaxSimSpeed', True)
        vs.Simulation.SetAttValue('RandSeed', randomSeed)

        # Reference to road network
        net = vs.Net

        ''' Set vehicle input '''
        vehInputs = net.VehicleInputs
        vehInput = vehInputs.ItemByKey(1)
        vehInput.SetAttValue('Volume(1)', demand)

        ''' Set default speed limit and 1st sign location'''
        VSLs = net.DesSpeedDecisions

        for VSL in VSLs :
            VSL.SetAttValue("DesSpeedDistr(10)", vf)   # car (km/h)
            VSL.SetAttValue("DesSpeedDistr(20)", vf)   # HGV (km/h)

        pos = 3203 - (sec1len-1)*1600
        VSLs.ItemByKey(4).SetAttValue("Pos", pos)
        VSLs.ItemByKey(5).SetAttValue("Pos", pos)
        VSLs.ItemByKey(6).SetAttValue("Pos", pos)

        ''' Get Data collection and link objects '''
        dataCollections = net.DataCollectionMeasurements
        links = net.Links
        vTT = net.VehicleTravelTimeMeasurements

        # Make sure that all lanes are open      
        link_Nlane = {}
        link_length = {}

        for link in links:
            ID = link.AttValue('No')
            link_Nlane[ID] = link.AttValue('NumLanes')
            link_length[ID] = link.AttValue('Length2D') #meter
            # Open all lanes
            for lane in link.Lanes:
                lane.SetAttValue('BlockedVehClasses', '')


        section_length = [0.0] * Nsec # length of each section
        for iSec in range(len(link_groups)): #xrange
            for linkID in link_groups[iSec]['MAINLINE']:
                link = links.ItemByKey(linkID)
                section_length[iSec] += link.AttValue('Length2D') #meter



        busNo = 2 # No. of buses used to block the lane
        busArray = [0] * busNo

        vehs = net.Vehicles

        # start simulation
        print("Client: Starting simulation")
        print("Scenario:", idxScenario)
        print("Controller:", idxController)
        print("Lane Closure:", idxLaneClosure, "section")
        print("Random Seed:", randomSeed)

        currentTime = 0

        #vs.Simulation.SetAttValue("SimBreakAt", 200)
        #vs.Simulation.RunContinuous()

        '''Link Segment Results'''

        while currentTime <= simulationTime_sec:
            # run sigle step of simulation
            vs.Simulation.RunSingleStep()
            currentTime = vs.Simulation.AttValue('SimSec')

            if 0 == currentTime % Tdata_sec:
                # Get density, flow, speed of each section
                for iSection in range(Nsec): #xrange
                    # Flow volume in one sampling period
                    dataCollection = dataCollections.ItemByKey(link_groups[iSection]['DC'])
                    flowSection[iSection] = dataCollection.AttValue('Vehs(Current,Last,All)')

                    # Density and Speed in each section
                    Nveh = 0
                    dist = 0
                    denomenator = 0
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
                        Nveh += link_vehs.count
                        dist += v * link_vehs.count #total distance traveled by all vehicles in one link
                        denomenator += link_length[linkID]
                    density[iSection] = 1000 * Nveh / denomenator #number of vehicles per km
                    if 0 == Nveh:
                        speed[iSection] = 0
                    else:
                        speed[iSection] = dist / Nveh #km/h

                ''' vehicle travel time '''
                vtt = vTT.ItemByKey(1)
                TravTm = vtt.AttValue('TravTm(Current,Last,All)')


                ''' write log files : flowSection, speed, density '''

                denFile = open(DenFileName, "a")
                flowFile = open(FlowFileName, "a")
                vttFile = open(VTTFileName, "a")

                for i in range(Nsec):
                    denFile.write(str(density[i]) + '\t')
                    flowFile.write(str(density[i]*speed[i]) + '\t')
                vttFile.write(str(TravTm) + '\t')

                denFile.write('\n')
                denFile.close()
                flowFile.write('\n')
                flowFile.close()
                vttFile.write('\n')
                vttFile.close()



            if currentTime == startTime_sec:
                # set incident scenario
                for i in range(busNo): #xrange
                    busArray[i] = vehs.AddVehicleAtLinkPosition(300, scenarios[idxScenario]['link'], scenarios[idxScenario]['lane'], scenarios[idxScenario]['coordinate'] + 20 * i, 0, 0)
                # Apply Lane change control
                if idxController == 2 or idxController == 3:
                    LC_Control(scenarios[idxScenario], links, 600 * idxLaneClosure)
                # Set cMode

            if currentTime == endTime_sec:
                # Remove the incident
                for veh in busArray:
                    vehs.RemoveVehicle(veh.AttValue('No'))

                # open all closed lanes
                for link in links:
                    for lane in link.Lanes:
                        lane.SetAttValue('BlockedVehClasses', '')

            #Compute the VSL command
            if 0 == currentTime % Tctrl_sec:
                if (idxController == 3 or idxController == 1) and (startTime_sec <= currentTime < endTime_sec):
                    vsl, err_sum = vsl_FeedbackLinearization(density, speed, rho_star, vsl, section_length, link_groups, err_sum, perturbation)
                else:
                    for iVSL in range(Nsec): #xrange
                        vsl[iVSL] = vf

                #Update VSL command in VISSIM
                for iSec in range(Nsec): #xrange
                    for vslID in link_groups[iSec]["VSL"]:
                        VSL = VSLs.ItemByKey(vslID)
                        VSL.SetAttValue("DesSpeedDistr(10)", vsl[iSec])   # car
                        VSL.SetAttValue("DesSpeedDistr(20)", vsl[iSec])   # HGV

                VSLFile = open(VSLFileName, "a")
                for i in range(Nsec):
                    VSLFile.write(str(vsl[i]) + '\t')
                VSLFile.write('\n')
                VSLFile.close()


    except pywintypes.com_error as err:
        print("err=", err)

        '''
        Error
        The specified configuration is not defined within VISSIM.
        
        Description
        Some methods for evaulations results need a previously configuration for data collection. 
        The error occurs when requesting results that have not been previously configured.
        For example, using the GetSegmentResult() method of the ILink interface to request
        density results can end up with this error if the density has not been requested within the configuration
        '''


def main():
    idxScenario = 2 # 0: NoBlock 1: AlltimeBlock
    # idxController and kLaneClosure
    ctrl = [(3,1)] #[(1, 0), (1, 2)]
    demands = [5500,]
    perturbations = [0,]
    sec1lens = [1.0,1.1,1.2,1.3,1.4] # Distance btw first two signs, must be in [1 , 3] (mi)

    simulationTime_sec = 5400 #4000
    #nMonteCarlo = 3
    #randomSeeds = calcRandomSeeds(nMonteCarlo)

    # Vehicle Composition ID
    # 1: 10% Trucks

    #demandComposition = 2

    networkDir = os.path.abspath(os.curdir)
    for demand in demands:
        for jController, kLaneClosure in ctrl:
            folderDir = os.path.join(networkDir, 'MicroResults')
            mkdir(folderDir)
            resultFile = open(os.path.join(folderDir, "result.txt"), 'w')
            for perturbation in perturbations:
                for sec1len in sec1lens:
                    lMC = round(random.uniform(1,100))
                    runSimulation(simulationTime_sec, idxScenario, jController, kLaneClosure, lMC, folderDir, networkDir, demand, perturbation, sec1len)

if __name__ == '__main__':
    main()








