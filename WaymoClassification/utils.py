import tensorflow as tf
import numpy as np


def get_state_features_description():
    state_features = {
        "state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        "state/tracks_to_predict": tf.io.FixedLenFeature(
            [128], tf.int64, default_value=None
        ),
        "state/current/bbox_yaw": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/height": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/length": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/timestamp_micros": tf.io.FixedLenFeature(
            [128, 1], tf.int64, default_value=None
        ),
        "state/current/valid": tf.io.FixedLenFeature(
            [128, 1], tf.int64, default_value=None
        ),
        "state/current/vel_yaw": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/velocity_x": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/velocity_y": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/speed": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/width": tf.io.FixedLenFeature(
            [128, 1], tf.float32, default_value=None
        ),
        "state/current/x": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/y": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/z": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/future/bbox_yaw": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/height": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/length": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/timestamp_micros": tf.io.FixedLenFeature(
            [128, 80], tf.int64, default_value=None
        ),
        "state/future/valid": tf.io.FixedLenFeature(
            [128, 80], tf.int64, default_value=None
        ),
        "state/future/vel_yaw": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/velocity_x": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/velocity_y": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/speed": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/width": tf.io.FixedLenFeature(
            [128, 80], tf.float32, default_value=None
        ),
        "state/future/x": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/y": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/future/z": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        "state/past/bbox_yaw": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/height": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/length": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/timestamp_micros": tf.io.FixedLenFeature(
            [128, 10], tf.int64, default_value=None
        ),
        "state/past/valid": tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        "state/past/vel_yaw": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/velocity_x": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/velocity_y": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/speed": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/width": tf.io.FixedLenFeature(
            [128, 10], tf.float32, default_value=None
        ),
        "state/past/x": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/y": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "state/past/z": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        "scenario/id": tf.io.FixedLenFeature([1], tf.string, default_value=None),
    }
    return state_features


def get_roadgraph_features_description():
    roadgraph_features = {
        "roadgraph_samples/dir": tf.io.FixedLenFeature(
            [20000, 3], tf.float32, default_value=None
        ),
        "roadgraph_samples/id": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/type": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/valid": tf.io.FixedLenFeature(
            [20000, 1], tf.int64, default_value=None
        ),
        "roadgraph_samples/xyz": tf.io.FixedLenFeature(
            [20000, 3], tf.float32, default_value=None
        ),
    }
    return roadgraph_features


def get_traffic_lights_features_description():
    traffic_light_features = {
        "traffic_light_state/current/state": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/valid": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/id": tf.io.FixedLenFeature(
            [1, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/current/x": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/current/y": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/current/z": tf.io.FixedLenFeature(
            [1, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/state": tf.io.FixedLenFeature(
            [10, 16], tf.int64, default_value=None
        ),
        "traffic_light_state/past/valid": tf.io.FixedLenFeature(
            [10, 16], tf.int64, default_value=None
        ),
        # "traffic_light_state/past/id":
        # tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        "traffic_light_state/past/x": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/y": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
        "traffic_light_state/past/z": tf.io.FixedLenFeature(
            [10, 16], tf.float32, default_value=None
        ),
    }
    return traffic_light_features


def get_features_description():
    roadgraph_features = get_roadgraph_features_description()
    state_features = get_state_features_description()
    traffic_light_features = get_traffic_lights_features_description()

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)

    return features_description


def parse_states(decoded_example, state_features):
    past_states = [tf.cast(decoded_example['state/past/{}'.format(feature)], dtype=tf.float32) for feature in
                   state_features if feature != "valid"]
    past_states = tf.stack(past_states, -1)

    current_states = [tf.cast(decoded_example['state/current/{}'.format(feature)], dtype=tf.float32) for feature in
                      state_features if feature != "valid"]
    current_states = tf.stack(current_states, -1)

    future_states = [tf.cast(decoded_example['state/future/{}'.format(feature)], dtype=tf.float32) for feature in
                     state_features if feature != "valid"]
    future_states = tf.stack(future_states, -1)

    # entire trajectory ground truth
    states = tf.concat([past_states, current_states, future_states], 2)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0
    states_is_valid = tf.concat([past_is_valid, current_is_valid, future_is_valid], 2)

    # If a sample was not seen at all in the past, we declare the sample as invalid.
    sample_is_valid = tf.reduce_any(tf.concat([past_is_valid, current_is_valid], 2), 2)

    return states, states_is_valid, sample_is_valid


def rotate_points(points, angle, origin=(0, 0)):  # assumed to be 2D. Angle assumed to be in radians
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    origin = np.atleast_2d(origin)
    return (R @ (points.T - origin.T) + origin.T).T
