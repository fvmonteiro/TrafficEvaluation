from enum import Enum


class VehicleType (Enum):
    HUMAN = 0
    ACC = 1
    AUTONOMOUS = 2
    CONNECTED = 3
    TRUCK = 4
    BUS = 5
    MOTORCYCLE = 6


class Vehicle:
    """Class containing vehicle parameters"""

    NGSIM_MOTORCYCLE_ID = 1
    NGSIM_CAR_ID = 2
    NGSIM_TRUCK_ID = 3
    VISSIM_CAR_ID = 100
    # VISSIM_LONGITUDINAL_CONTROLLED_CAR_ID = 105
    VISSIM_AUTONOMOUS_CAR_ID = 110
    VISSIM_CONNECTED_CAR_ID = 120
    VISSIM_TRUCK_ID = 200
    VISSIM_BUS_ID = 300
    # TYPE_CAR = 'car'
    # TYPE_AV = 'autonomous vehicle'
    # TYPE_CAV = 'connected vehicle'
    # TYPE_TRUCK = 'truck'
    # TYPE_BUS = 'bus'
    # TYPE_MOTORCYCLE = 'motorcycle'

    RELEVANT_TYPES = {VehicleType.HUMAN, VehicleType.TRUCK,
                      VehicleType.ACC,
                      VehicleType.AUTONOMOUS, VehicleType.CONNECTED}

    # VISSIM and NGSIM codes for different vehicle types
    INT_TO_ENUM = {NGSIM_MOTORCYCLE_ID: VehicleType.MOTORCYCLE,
                   NGSIM_CAR_ID: VehicleType.HUMAN,
                   NGSIM_TRUCK_ID: VehicleType.TRUCK,
                   VISSIM_CAR_ID: VehicleType.HUMAN,
                   VISSIM_AUTONOMOUS_CAR_ID: VehicleType.AUTONOMOUS,
                   VISSIM_CONNECTED_CAR_ID: VehicleType.CONNECTED,
                   VISSIM_TRUCK_ID: VehicleType.TRUCK,
                   VISSIM_BUS_ID: VehicleType.CONNECTED}

    # Typical parameters values
    _MAX_BRAKE_PER_TYPE = {VehicleType.HUMAN: 7.5,
                           VehicleType.AUTONOMOUS: 7.5,
                           VehicleType.CONNECTED: 7.5,
                           VehicleType.TRUCK: 5.5}
    _MAX_JERK_PER_TYPE = {VehicleType.HUMAN: 50,
                          VehicleType.AUTONOMOUS: 50,
                          VehicleType.CONNECTED: 50,
                          VehicleType.TRUCK: 30}
    _BRAKE_DELAY_PER_TYPE = {VehicleType.HUMAN: 0.75,
                             VehicleType.AUTONOMOUS: 0.2,
                             VehicleType.CONNECTED: 0.1,
                             VehicleType.TRUCK: 0.5}
    # 33 m/s ~= 120km/h ~= 75 mph
    _FREE_FLOW_VELOCITY_PER_TYPE = {VehicleType.HUMAN: 33,
                                    VehicleType.AUTONOMOUS: 33,
                                    VehicleType.CONNECTED: 33,
                                    VehicleType.TRUCK: 25}

    def __init__(self, i_type: int, gamma: float = 1):
        """Assigns typical vehicle values based on the vehicle type
        :param i_type: integer describing the vehicle type
        :param gamma: factor multiplying standard value maximum braking
        """

        try:
            self.type = self.INT_TO_ENUM[i_type]
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
            h = ((rho**2 * self.free_flow_velocity / 2 + rho * self.lambda1)
                 / ((1-gamma)*self.max_brake))
            d = self.lambda1**2 / (2*(1-gamma)*self.max_brake) + self.lambda0
        elif gamma > (1 - rho)**2:
            h = ((gamma - (1-rho)**2)*self.free_flow_velocity/2/gamma
                 + self.lambda1) / self.max_brake
            d = self.lambda1**2 / 2 / self.max_brake + self.lambda0
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

    # def compute_gap_thresholds(self, ego_velocity, leader_velocity):
    #
    #     delta_v = leader_velocity - ego_velocity
    #     gap_thresholds = []
    #     gap_thresholds.append(delta_v**2 + 2*gap*(self.accel_t0 +
    #     leader.max_brake))
