from enum import Enum
from typing import Union, Tuple

from math import sin, cos
import numpy as np


def find_collision_time_and_severity(initial_gaps,
                                     follower_v0, leader_v0,
                                     is_lane_changing,
                                     follower_type=100, leader_type=100,
                                     follower_a0=None):
    follower = Vehicle(follower_type)
    leader = Vehicle(leader_type)
    tf, delta_xf, vf, af = follower.simulate_emergency_braking_as_follower(
        0, follower_v0, is_lane_changing, follower_a0)
    tl, delta_xl, vl, al = leader.simulate_emergency_braking_as_leader(
        0, leader_v0)
    time_steps_diff = len(tf) - len(tl)
    if time_steps_diff > 0:
        time = tf
        delta_xl = np.append(delta_xl, np.ones(time_steps_diff) * delta_xl[-1])
        vl = np.append(vl, np.zeros(time_steps_diff))
        # al = np.append(al, np.zeros(time_steps_diff))
    else:
        time_steps_diff = -time_steps_diff
        time = tl
        delta_xf = np.append(delta_xf, np.ones(time_steps_diff) * delta_xf[-1])
        vf = np.append(vf, np.zeros(time_steps_diff))
        # af = np.append(af, np.zeros(time_steps_diff))

    # _, axes = plt.subplots(2, 1)
    # axes[0].plot(time, delta_xf)
    # axes[0].plot(time, delta_xl)
    # axes[0].set_title('x')
    #
    # axes[1].plot(time, vf)
    # axes[1].plot(time, vl)
    # axes[1].set_title('v')
    # plt.show()
    tc = - np.ones(len(initial_gaps))
    severity = np.zeros(len(initial_gaps))

    jerk_i = []
    jerk_time_idx = []
    for i in range(len(initial_gaps)):
        gap = initial_gaps[i] + delta_xl - delta_xf
        temp = np.where(gap <= 0)
        if temp[0].size > 0:
            time_index = temp[0][0]
            tc[i] = time[time_index]
            severity[i] = (vf - vl)[time_index]
        # if (follower.brake_delay < tc[i]
        #         < follower.brake_delay + follower.tau_j):
        #     jerk_i.append(i)
        #     jerk_time_idx.append(time_index)
    return tc, severity


# TODO: these VehicleType and Vehicle classes are messy. Consider better
#  organizing the data. Perhaps make VehicleType a regular class.
class VehicleType(Enum):
    HUMAN_DRIVEN = 0
    ACC = 1
    AUTONOMOUS = 2
    CONNECTED = 3
    PLATOON = 4
    TRAFFIC_LIGHT_ACC = 5
    TRAFFIC_LIGHT_CACC = 6
    TRUCK = 7
    BUS = 8
    MOTORCYCLE = 9


vehicle_type_to_str_map = {
    VehicleType.HUMAN_DRIVEN: 'no control',
    VehicleType.ACC: 'ACC',
    VehicleType.AUTONOMOUS: 'AV',
    VehicleType.CONNECTED: 'CAV',
    VehicleType.PLATOON: 'Platoon',
    VehicleType.TRAFFIC_LIGHT_ACC: 'TL-ACC',
    VehicleType.TRAFFIC_LIGHT_CACC: 'TL-CACC',
    VehicleType.TRUCK: 'truck',
    VehicleType.BUS: 'bus',
    VehicleType.MOTORCYCLE: 'motorcycle'
}


class Vehicle:
    """Class containing vehicle parameters"""

    NGSIM_MOTORCYCLE_ID = 1
    NGSIM_CAR_ID = 2
    NGSIM_TRUCK_ID = 3
    VISSIM_CAR_ID = 100
    VISSIM_ACC_CAR_ID = 105
    VISSIM_AUTONOMOUS_CAR_ID = 110
    VISSIM_CONNECTED_CAR_ID = 120
    VISSIM_TRAFFIC_LIGHT_ACC_ID = 130
    VISSIM_TRAFFIC_LIGHT_CACC_ID = 135
    VISSIM_PLATOON_CAR_ID = 140
    VISSIM_TRUCK_ID = 200
    VISSIM_BUS_ID = 300
    # TYPE_CAR = 'car'
    # TYPE_AV = 'autonomous vehicle'
    # TYPE_CAV = 'connected vehicle'
    # TYPE_TRUCK = 'truck'
    # TYPE_BUS = 'bus'
    # TYPE_MOTORCYCLE = 'motorcycle'

    RELEVANT_TYPES = {VehicleType.HUMAN_DRIVEN, VehicleType.TRUCK,
                      VehicleType.ACC, VehicleType.PLATOON,
                      VehicleType.AUTONOMOUS, VehicleType.CONNECTED,
                      VehicleType.TRAFFIC_LIGHT_ACC,
                      VehicleType.TRAFFIC_LIGHT_CACC}

    # VISSIM and NGSIM codes for different vehicle types. This dictionary is
    # necessary when reading results, i.e., from data source to python.
    _INT_TO_ENUM = {NGSIM_MOTORCYCLE_ID: VehicleType.MOTORCYCLE,
                    NGSIM_CAR_ID: VehicleType.HUMAN_DRIVEN,
                    NGSIM_TRUCK_ID: VehicleType.TRUCK,
                    VISSIM_CAR_ID: VehicleType.HUMAN_DRIVEN,
                    VISSIM_ACC_CAR_ID: VehicleType.ACC,
                    VISSIM_AUTONOMOUS_CAR_ID: VehicleType.AUTONOMOUS,
                    VISSIM_CONNECTED_CAR_ID: VehicleType.CONNECTED,
                    VISSIM_TRAFFIC_LIGHT_ACC_ID: VehicleType.TRAFFIC_LIGHT_ACC,
                    VISSIM_TRAFFIC_LIGHT_CACC_ID:
                        VehicleType.TRAFFIC_LIGHT_CACC,
                    VISSIM_PLATOON_CAR_ID: VehicleType.PLATOON,
                    VISSIM_TRUCK_ID: VehicleType.TRUCK,
                    VISSIM_BUS_ID: VehicleType.CONNECTED}

    # Useful when editing vissim simulation parameters
    ENUM_TO_VISSIM_ID = {
        VehicleType.HUMAN_DRIVEN: VISSIM_CAR_ID,
        VehicleType.ACC: VISSIM_ACC_CAR_ID,
        VehicleType.AUTONOMOUS: VISSIM_AUTONOMOUS_CAR_ID,
        VehicleType.CONNECTED: VISSIM_CONNECTED_CAR_ID,
        VehicleType.TRAFFIC_LIGHT_ACC: VISSIM_TRAFFIC_LIGHT_ACC_ID,
        VehicleType.TRAFFIC_LIGHT_CACC: VISSIM_TRAFFIC_LIGHT_CACC_ID,
        VehicleType.PLATOON: VISSIM_PLATOON_CAR_ID
    }

    # Typical parameters values
    _CAR_MAX_BRAKE = 6  # [m/s2]
    _CAR_MAX_JERK = 50  # [m/s3]
    _CAR_FREE_FLOW_VELOCITY = 33  # 33 m/s ~= 120km/h ~= 75 mph
    _TRUCK_MAX_BRAKE = 3  # From VISSIM: 5.5
    _TRUCK_MAX_JERK = 30
    _MAX_BRAKE_PER_TYPE = {VehicleType.HUMAN_DRIVEN: _CAR_MAX_BRAKE,
                           VehicleType.ACC: _CAR_MAX_BRAKE,
                           VehicleType.AUTONOMOUS: _CAR_MAX_BRAKE,
                           VehicleType.CONNECTED: _CAR_MAX_BRAKE,
                           VehicleType.TRAFFIC_LIGHT_ACC: _CAR_MAX_BRAKE,
                           VehicleType.TRAFFIC_LIGHT_CACC: _CAR_MAX_BRAKE,
                           VehicleType.PLATOON: _CAR_MAX_BRAKE,
                           VehicleType.TRUCK: _TRUCK_MAX_BRAKE}
    _MAX_JERK_PER_TYPE = {VehicleType.HUMAN_DRIVEN: _CAR_MAX_JERK,
                          VehicleType.ACC: _CAR_MAX_JERK,
                          VehicleType.AUTONOMOUS: _CAR_MAX_JERK,
                          VehicleType.CONNECTED: _CAR_MAX_JERK,
                          VehicleType.TRAFFIC_LIGHT_ACC: _CAR_MAX_JERK,
                          VehicleType.TRAFFIC_LIGHT_CACC: _CAR_MAX_JERK,
                          VehicleType.PLATOON: _CAR_MAX_JERK,
                          VehicleType.TRUCK: _TRUCK_MAX_JERK}
    _BRAKE_DELAY_PER_TYPE = {VehicleType.HUMAN_DRIVEN: 0.75,
                             VehicleType.ACC: 0.2,
                             VehicleType.AUTONOMOUS: 0.2,
                             VehicleType.CONNECTED: 0.1,
                             VehicleType.TRAFFIC_LIGHT_ACC: 0.2,
                             VehicleType.TRAFFIC_LIGHT_CACC: 0.1,
                             VehicleType.PLATOON: 0.1,
                             VehicleType.TRUCK: 0.5}
    _FREE_FLOW_VELOCITY_PER_TYPE = {
        VehicleType.HUMAN_DRIVEN: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.ACC: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.AUTONOMOUS: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.CONNECTED: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.TRAFFIC_LIGHT_ACC: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.TRAFFIC_LIGHT_CACC: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.PLATOON: _CAR_FREE_FLOW_VELOCITY,
        VehicleType.TRUCK: 25}

    _sampling_time = 0.01

    def __init__(self, i_type: Union[int, VehicleType], gamma: float = 1):
        """
        Assigns typical vehicle values based on the vehicle type
        :param i_type: integer describing the vehicle type
        :param gamma: factor multiplying standard value maximum braking
        """

        if isinstance(i_type, VehicleType):
            self.type = i_type
        else:
            try:
                self.type = self._INT_TO_ENUM[i_type]
            except KeyError:
                print('{}: KeyError: vehicle type {} not defined'.
                      format(self.__class__.__name__, i_type))
                raise
        if self.type not in self.RELEVANT_TYPES:
            self.is_relevant = False
            return
        self.is_relevant = True

        # Parameters independent of vehicle type
        self.accel_t0 = 0.5
        # Parameters dependent of vehicle type
        self.max_brake = self._MAX_BRAKE_PER_TYPE[self.type] * gamma
        self.max_brake_lane_change = self.max_brake / 2
        self.max_jerk = self._MAX_JERK_PER_TYPE[self.type]
        self.brake_delay = self._BRAKE_DELAY_PER_TYPE[self.type]
        self.free_flow_velocity = self._FREE_FLOW_VELOCITY_PER_TYPE[self.type]
        # Emergency braking parameters
        self.tau_j, self.lambda0, self.lambda1 = (
            self._compute_emergency_braking_parameters(self.max_brake))
        (self.tau_j_lane_change, self.lambda0_lane_change,
         self.lambda1_lane_change) = (
            self._compute_emergency_braking_parameters(
                self.max_brake_lane_change))

    def compute_vehicle_following_parameters(self, leader_max_brake: float,
                                             rho: float) -> (float, float):
        """
        Computes time headway (h) and constant term (d) of the time headway
        policy: g = h.v + d.

        Values of h and d are computed such that g is an
        overestimation of the collision free gap

        :param leader_max_brake:
        :param rho: defines the lower bound on the leader velocity following
         (1-rho)vE(t) <= vL(t). Must be in the interval [0, 1]
        :return: time headway (h) and constant term (d)
        """

        gamma = leader_max_brake / self.max_brake
        gamma_threshold = ((1 - rho) * self.free_flow_velocity
                           / (self.free_flow_velocity + self.lambda1))

        if gamma < gamma_threshold:
            h = ((rho ** 2 * self.free_flow_velocity / 2 + rho * self.lambda1)
                 / ((1 - gamma) * self.max_brake))
            d = self.lambda1 ** 2 / (
                    2 * (1 - gamma) * self.max_brake) + self.lambda0
        elif gamma > (1 - rho) ** 2:
            h = ((gamma - (1 - rho) ** 2) * self.free_flow_velocity / 2 / gamma
                 + self.lambda1) / self.max_brake
            d = self.lambda1 ** 2 / 2 / self.max_brake + self.lambda0
        else:
            h = self.lambda1 / self.max_brake
            d = self.lambda1 ** 2 / 2 / self.max_brake + self.lambda0

        return h, d

    def _compute_emergency_braking_parameters(self, max_brake):
        tau_j = (self.accel_t0 + max_brake) / self.max_jerk
        lambda1 = ((self.accel_t0 + max_brake)
                   * (self.brake_delay + tau_j / 2))
        lambda0 = -(self.accel_t0 + max_brake) / 2 * (
                self.brake_delay ** 2 + self.brake_delay * tau_j
                + tau_j ** 2 / 3)

        return tau_j, lambda0, lambda1

    def simulate_emergency_braking_as_leader(self, x0, v0):
        return self.simulate_emergency_braking(x0, v0, False, False,
                                               -self.max_brake)

    def simulate_emergency_braking_as_follower(self, x0, v0,
                                               is_lane_changing, a0=None):
        if a0 is None:
            a0 = self.accel_t0
        return self.simulate_emergency_braking(x0, v0, True,
                                               is_lane_changing, a0)

    def simulate_emergency_braking(self, x0, v0, is_follower,
                                   is_lane_changing, a0):
        sampling_time = self._sampling_time  # [s]
        t = [0]
        x = [x0]
        v = [v0]
        a = [a0]
        j = [0]
        while v[-1] > 0:
            x.append(x[-1]
                     + v[-1] * sampling_time
                     + a[-1] / 2 * sampling_time ** 2
                     + j[-1] / 6 * sampling_time ** 3)
            v.append(v[-1]
                     + a[-1] * sampling_time
                     + j[-1] / 2 * sampling_time ** 2)
            t.append(t[-1] + sampling_time)
            new_a, new_j = self.get_jerk_and_accel(t[-1], a[-1], is_follower,
                                                   is_lane_changing)
            a.append(new_a)
            j.append(new_j)

        v[-1] = 0
        return np.array(t), np.array(x), np.array(v), np.array(a)

    def get_jerk_and_accel(self, time, current_accel, is_follower,
                           is_lane_changing):
        if is_follower:
            tau_j = self.tau_j_lane_change if is_lane_changing else self.tau_j
            if time < self.brake_delay:
                new_accel = current_accel
                new_jerk = 0
            elif time < (self.brake_delay + tau_j):
                new_accel = current_accel - self.max_jerk * self._sampling_time
                new_jerk = -self.max_jerk
            else:
                # We want to ensure the accel equals max_brake after tau_j
                # in spite of possible rounding errors
                new_accel = -(self.max_brake_lane_change if is_lane_changing
                              else self.max_brake)
                new_jerk = 0
        else:
            new_accel = -self.max_brake
            new_jerk = 0
        return new_accel, new_jerk

    def simulate_emergency_braking_during_lane_change(self):
        pass


def severity_in_2d_collision(vx0: Tuple[float, float],
                             vy0: Tuple[float, float],
                             omega0: Tuple[float, float],
                             restitution_coefficient: float,
                             friction_coefficient: Union[float, str],
                             moment_coefficient: float,
                             mass: Tuple[float, float],
                             moment_of_inertia: Tuple[float, float],
                             heading_angle: Tuple[float, float],
                             angle_to_impact_line: Tuple[float, float],
                             impact_angle: float,
                             distance_to_impact_center: Tuple[float, float]
                             ):
    """
    Computes the Delta-V of a collision treating vehicles as rigid bodies.
    :param vx0:
    :param vy0:
    :param omega0:
    :param restitution_coefficient: defined as the ratio between final and
     initial relative speeds. It measures the loss of kinetic energy in the
     collision. Range: [0, 1]
    :param friction_coefficient: defined as the ratio of tangent impulse and
     normal impulse relative to the impact line. It measures the change in
     tangential velocities during the collision. Range: [0, mu_max],
     where mu_max depends on the restitution coefficient and initial relative
     velocities
    :param moment_coefficient: models how the collision might generate
     angular momentum on the vehicles. Range: {1} and [-1, 0]. The value 1
     means no moment is generated, which is reasonable for close-to-1D
     collisions. Value 0 leads to equal final angular velocities, which is a
     perfectly inelastic angular collision.
    :param mass:
    :param moment_of_inertia:
    :param heading_angle: heading angle relative to the road's x axis
    :param angle_to_impact_line: angle between the length axis of a vehicle
     and the impact line
    :param impact_angle: relative to the road's y axis
    :param distance_to_impact_center: distance from center of mass to the
     impact center
    :return:
    """
    v1x, v2x = vx0
    v1x, v2y = vy0
    omega1, omega2 = omega0
    e = restitution_coefficient
    em = moment_coefficient
    m1, m2 = mass
    I1, I2 = moment_of_inertia
    theta1, theta2 = heading_angle
    phi1, phi2 = angle_to_impact_line
    cos_beta = cos(impact_angle)
    sin_beta = sin(impact_angle)
    d1, d2 = distance_to_impact_center

    # Actually, we don't know the right value of mu_max for rigid-body
    # collisions
    mu_max = abs((v1x - v2y) / (v1x - v2x)) / (1 + e)
    mu = mu_max if friction_coefficient == 'max' else friction_coefficient
    # if friction_coefficient > mu_max:
    #     print('Selected friction coefficient is too high.\n'
    #           'Setting it to the max value:', mu_max)
    #     mu = mu_max

    da = d2 * sin(theta2 + phi2)
    db = d2 * cos(theta2 + phi2)
    dc = d1 * sin(theta1 + phi1)
    dd = d1 * cos(theta1 + phi1)

    A = np.array([[m1, 0, 0, m2, 0, 0],
                  [0, m1, 0, 0, m2, 0],
                  [0, m1 * (db + dd), I1, m2 * (da + dc), 0, I2],
                  [cos_beta, sin_beta, dc * cos_beta - dd * sin_beta,
                   -cos_beta, -sin_beta, da * cos_beta - db * sin_beta],
                  [0, m1 * (cos_beta - mu * sin_beta), 0,
                   m2 * (sin_beta + mu * cos_beta), 0, 0],
                  [-em * m1 * dc / I1, em * m1 * dd / I1, -1 + 2 * em,
                   -em * m2 * da / I2, em * m2 * db / I2, 1 - 2 * em]])
    A_b = np.copy(A)
    A_b[3, :] *= -e
    A_b[5, 2] = em
    A_b[5, 5] = -em
    b = np.dot(A_b, np.array([v1x, v1x, omega1, v2x, v2y, omega2]))
    x = np.linalg.solve(A, b)
    return x
