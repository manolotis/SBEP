import matplotlib.pyplot as plt
import numpy as np
import pickle
from enum import Enum
from shapely.geometry import LineString


class WaymoTag(Enum):
    # Road user type
    RoadUserType_Ego = 1
    RoadUserType_Vehicle = 2
    RoadUserType_VRU = 3
    RoadUserType_VRU_Pedestrian = 4
    RoadUserType_VRU_Cyclist = 5

    ##### Trajectory properties
    # Speed & Velocity profile
    TrajectoryProperty_Velocity_Constant = 6
    TrajectoryProperty_Velocity_Variable = 7
    TrajectoryProperty_Speed_Constant = 8
    TrajectoryProperty_Speed_Variable = 9

    # Curvature
    TrajectoryProperty_Curvature_Straight = 10
    TrajectoryProperty_Curvature_NonStraight = 11

    # Behavior
    TrajectoryProperty_Behavior_Starting = 12
    TrajectoryProperty_Behavior_Stopping = 13
    TrajectoryProperty_Behavior_Still = 14

    # Relevance to ego vehicle
    TrajectoryProperty_Relevance_Relevant = 15
    TrajectoryProperty_Relevance_Irrelevant = 16

    # How much of the trajectory was observed
    TrajectoryProperty_Observation_Late = 17
    TrajectoryProperty_Observation_VeryLate = 18
    TrajectoryProperty_Observation_Full = 19
    TrajectoryProperty_Observation_None = 20
    TrajectoryProperty_Observation_Reappear = 21

    TrajectoryProperty_TrackToPredict_Yes = 30
    TrajectoryProperty_TrackToPredict_No = 31

    # Scenario properties
    ScenarioProperty_HasVehicles = 100
    ScenarioProperty_HasPedestrians = 101
    ScenarioProperty_HasCyclists = 102


class WaymoTrajectory:
    def __init__(self, RU_data):
        # states in the data:
        # ["x", "y", "z", "bbox_yaw", "length", "width", "height", "speed",
        # "timestamp_micros", "vel_yaw", "velocity_x", "velocity_y"]

        self.OBSERVATION_STEPS = 11  # 10 past + 1 current
        self.LATE_DETECTION_STEPS = 3
        self.VERY_LATE_DETECTION_STEPS = 1
        self.STILL_THRESHOLD = 0.01
        self.SINUOSITY_THRESHOLD = 1.1
        self.MIN_MOVEMENT = 0.5  # when calculating curvature, to prevent division by 0 and not compute curvature for very short trajectories
        # self.CV_CHANGE_THRESHOLD = 0.1  # Threshold to determine velocity profile

        self.tags = []
        self.states = RU_data['states']
        self.states_is_valid = RU_data['states_is_valid']
        self.is_ttp = RU_data['is_ttp']
        self.RU_ID = RU_data['id']
        self.RU_type = RU_data['type']

        self._determine_curvature()
        self._determine_behavior()
        self._determine_observation_length()
        self._determine_ttp()

    def _determine_curvature(self):
        xy_future = self.states[self.OBSERVATION_STEPS:, :2]
        is_valid_future = self.states_is_valid[self.OBSERVATION_STEPS:]

        xy_future_valid = xy_future[is_valid_future]

        if len(xy_future_valid) < 2:
            return

        ls_straight = LineString([xy_future_valid[0], xy_future_valid[-1]])
        ls = LineString(xy_future_valid)

        if ls_straight.length <= self.MIN_MOVEMENT:
            return

        sinuosity_index = ls.length / ls_straight.length

        if sinuosity_index <= self.SINUOSITY_THRESHOLD:
            self._add_tag(WaymoTag.TrajectoryProperty_Curvature_Straight)
        else:
            self._add_tag(WaymoTag.TrajectoryProperty_Curvature_NonStraight)

    def _determine_behavior(self):
        speeds_past = self.states[:self.OBSERVATION_STEPS, 7]
        speeds_future = self.states[self.OBSERVATION_STEPS:, 7]
        is_valid_past = self.states_is_valid[:self.OBSERVATION_STEPS]
        is_valid_future = self.states_is_valid[self.OBSERVATION_STEPS:]

        speeds_past_valid = speeds_past[is_valid_past]
        speeds_future_valid = speeds_future[is_valid_future]

        always_still_during_past = np.all(speeds_past_valid <= self.STILL_THRESHOLD)
        always_still_during_future = np.all(speeds_future_valid <= self.STILL_THRESHOLD)
        moving_during_past = np.any(speeds_past_valid > self.STILL_THRESHOLD)
        moving_during_future = np.any(speeds_future_valid > self.STILL_THRESHOLD)
        still_during_future = np.any(speeds_future_valid <= self.STILL_THRESHOLD)

        if always_still_during_past and always_still_during_future:
            self._add_tag(WaymoTag.TrajectoryProperty_Behavior_Still)

        if always_still_during_past and moving_during_future:
            self._add_tag(WaymoTag.TrajectoryProperty_Behavior_Starting)

        if moving_during_past and still_during_future:
            self._add_tag(WaymoTag.TrajectoryProperty_Behavior_Stopping)

    def _determine_reappearance(self):
        is_valid_past = self.states_is_valid[:self.OBSERVATION_STEPS]
        is_valid_future = self.states_is_valid[self.OBSERVATION_STEPS:]

        # if not available in the future at all, then it doesn't reappear
        if not np.any(is_valid_future):
            return

        # if fully available in observations, then it doesn't disappear
        if np.all(is_valid_past):
            return

        # if never available in observations, then it doesn't disappear
        if not np.any(is_valid_past):
            return

        # if not observed in the "current" step, then tag as reappearance
        if is_valid_past[-1] == False:
            self._add_tag(WaymoTag.TrajectoryProperty_Observation_Reappear)

    def _determine_observation_length(self):
        states_is_valid = self.states_is_valid[..., :self.OBSERVATION_STEPS]

        # if all observation steps available
        all_states_available = np.all(states_is_valid)
        if all_states_available:
            self._add_tag(WaymoTag.TrajectoryProperty_Observation_Full)
            return

        # otherwise, check if it's a late or and/or very late detection
        late_cutoff = self.OBSERVATION_STEPS - self.LATE_DETECTION_STEPS
        states_is_valid_beginning = states_is_valid[..., :late_cutoff]
        states_is_valid_end = states_is_valid[..., late_cutoff:]

        if np.any(states_is_valid_beginning):
            return  # then not late nor very late detection

        num_valid_steps_end = states_is_valid_end.sum()
        if num_valid_steps_end == 0:
            self._add_tag(WaymoTag.TrajectoryProperty_Observation_None)
            return

        if 0 < num_valid_steps_end <= self.LATE_DETECTION_STEPS:
            self._add_tag(WaymoTag.TrajectoryProperty_Observation_Late)

        if num_valid_steps_end <= self.VERY_LATE_DETECTION_STEPS:
            self._add_tag(WaymoTag.TrajectoryProperty_Observation_VeryLate)

    def _determine_ttp(self):
        if self.is_ttp:
            self._add_tag(WaymoTag.TrajectoryProperty_TrackToPredict_Yes)
        else:
            self._add_tag(WaymoTag.TrajectoryProperty_TrackToPredict_No)

    def _add_tag(self, tag):
        self.tags.append(tag)


class WaymoRoadUser:
    # toDo different thresholds for different types of road users?
    def __init__(self, RU_data):
        self.RU_data = RU_data
        self.id = int(RU_data['id'])
        self.trajectory = WaymoTrajectory(RU_data)
        self.tags = []
        self._add_RU_specific_tags()

    def _add_RU_specific_tags(self):
        # Should be implemented in specific RU class
        raise NotImplementedError

    @property
    def derived_tags(self):
        return self.derive_tags()

    def derive_tags(self):
        all_tags = []
        all_tags.extend(self.tags)
        all_tags.extend(self.trajectory.tags)
        return all_tags

    def is_ego(self):
        return self.RU_data['is_ego']

    def save_tags(self, scenario_id, config):
        all_tags = self.derived_tags

        savepath = f"{config.TAGS_FOLDER}{scenario_id}_{self.id}.pkl"

        with open(savepath, 'wb') as f:
            pickle.dump(all_tags, f)


class WaymoEgo(WaymoRoadUser):
    def __init__(self, RU_data):
        super().__init__(RU_data)

    def _add_RU_specific_tags(self):
        self.tags.append(WaymoTag.RoadUserType_Vehicle)


class WaymoVehicle(WaymoRoadUser):
    def __init__(self, RU_data):
        super().__init__(RU_data)

    def _add_RU_specific_tags(self):
        if self.is_ego():
            self.tags.append(WaymoTag.RoadUserType_Ego)
        self.tags.append(WaymoTag.RoadUserType_Vehicle)


class WaymoPedestrian(WaymoRoadUser):
    def __init__(self, RU_data):
        super().__init__(RU_data)

    def _add_RU_specific_tags(self):
        self.tags.append(WaymoTag.RoadUserType_VRU)
        self.tags.append(WaymoTag.RoadUserType_VRU_Pedestrian)


class WaymoCyclist(WaymoRoadUser):
    def __init__(self, RU_data):
        super().__init__(RU_data)

    def _add_RU_specific_tags(self):
        self.tags.append(WaymoTag.RoadUserType_VRU)
        self.tags.append(WaymoTag.RoadUserType_VRU_Cyclist)


class WaymoScenario:

    def __init__(self, data_scenario, config, load_preprocessed_rg=True):
        self.SCENARIO_ID = data_scenario['scenario_id'][0].decode()
        self.config = config
        self.states = data_scenario['states']
        self.states_is_valid = data_scenario['states_is_valid']
        self.tracks_to_predict = data_scenario['tracks_to_predict']
        self.object_ids = data_scenario['object_ids']
        self.object_types = data_scenario['object_types']
        self.is_ego = data_scenario['is_ego']
        self.RUs = []

        self._load_RUs()

        # ToDo: verify tags (e.g. there should be at least an ego vehicle)

    def save_tags(self):
        """
        For now just save the tags of the individual road agents. In the future there will be extra tags for the scenarios
        (e.g. depending on road layout, interaction of agents...etc)
        """
        scenario_tags = self._get_tags()

        savepath = f"{self.config.TAGS_FOLDER}{self.SCENARIO_ID}.pkl"
        with open(savepath, "wb") as f:
            pickle.dump(scenario_tags, f)

    def get_tags(self):
        return self._get_tags()

    def _get_tags(self):
        scenario_tags = {}

        for RU in self.RUs:
            scenario_tags[RU.id] = RU.derived_tags

        return scenario_tags

    def _load_RUs(self):
        valid_RU_indexes = np.argwhere(self.object_types > -1).flatten()
        for RU_index in valid_RU_indexes:
            RU_data = {
                'states': self.states[RU_index],
                'states_is_valid': self.states_is_valid[RU_index],
                'is_ttp': self.tracks_to_predict[RU_index],
                'id': self.object_ids[RU_index],
                'type': self.object_types[RU_index],
                'is_ego': self.is_ego[RU_index],
            }

            int2ru = {
                1: WaymoVehicle,
                2: WaymoPedestrian,
                3: WaymoCyclist
            }

            if RU_data['is_ego']:
                self.RUs.append(WaymoEgo(RU_data))
                continue

            self.RUs.append(int2ru[RU_data['type']](RU_data))

    def _plot_RUs(self):
        states = self.states
        states_is_valid = self.states_is_valid
        is_track_to_predict = self.tracks_to_predict

        # Vehicle=1, Pedestrian=2, Cyclist=3
        object_types = self.object_types
        is_vehicle = object_types == 1
        is_pedestrian = object_types == 2
        is_cyclist = object_types == 3

        masks = [is_vehicle, is_pedestrian, is_cyclist]
        masks_ttp = [is_vehicle & is_track_to_predict, is_pedestrian & is_track_to_predict,
                     is_cyclist & is_track_to_predict]

        labels = ["Vehicle", "Pedestrian", "Cyclist"]
        labels_ttp = ["Vehicle TTP", "Pedestrian TTP", "Cyclist TTP"]

        for i, type_mask in enumerate(masks):
            states_type = states[type_mask]
            states_type_is_valid = states_is_valid[type_mask]

            states_type_ttp = states[masks_ttp[i]]
            states_type_ttp_is_valid = states_is_valid[masks_ttp[i]]

            x, y = states_type[states_type_is_valid][:, 0], states_type[states_type_is_valid][:, 1]
            plt.scatter(x, y, s=1, marker='x', alpha=0.5, label=labels[i])

            x, y = states_type_ttp[states_type_ttp_is_valid][:, 0], states_type_ttp[states_type_ttp_is_valid][:, 1]
            plt.scatter(x, y, s=3, marker='o', label=labels_ttp[i])

    def plot(self):
        # ToDo: load and plot roadgraph
        self._plot_RUs()

        plt.tight_layout()
        plt.legend()
        plt.show()
