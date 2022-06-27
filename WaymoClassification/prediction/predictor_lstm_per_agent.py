import multiprocessing
import pickle
import tensorflow as tf
from tqdm import tqdm
from WaymoClassification import config, utils
import numpy as np
import os

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

features_description = utils.get_features_description()

state_features = ["x", "y", "bbox_yaw", "length", "width", "velocity_x", "velocity_y", "valid"]
MULTIPROCESSING = False

SCALER_EXCEPTIONS = ["valid"]

pedestrian_path = "../trained_models/lstmV1_pedestrians/"
cyclist_path = "../trained_models/lstmV1_cyclists/"
vehicle_path = "../trained_models/lstmV1_vehicles/"

model_pedestrians = tf.keras.models.load_model(pedestrian_path)
model_cyclists = tf.keras.models.load_model(cyclist_path)
model_vehicles = tf.keras.models.load_model(vehicle_path)


def load_scalers():
    scalers = []
    for feature in state_features:
        path = config.SCALERS_PATH + "{}.pkl".format(feature)
        if not os.path.exists(path):
            if feature in SCALER_EXCEPTIONS:
                continue
            print("No scaler found for feature {}".format(feature))
            print("Path: {}".format(path))
            raise ValueError

        with open(path, 'rb') as f:
            scaler = pickle.load(f)
            scalers.append(scaler)
            print("loaded {} scaler ".format(feature))
    return scalers


scalers = load_scalers()


def add_valid(decoded_example, states):  # adds whether or not a sample timestep is valid
    past_valid = tf.cast(decoded_example['state/past/valid'], dtype=tf.float32)
    past_valid = tf.expand_dims(past_valid, axis=-1)
    current_valid = tf.cast(decoded_example['state/current/valid'], dtype=tf.float32)
    current_valid = tf.expand_dims(current_valid, axis=-1)
    future_valid = tf.cast(decoded_example['state/future/valid'], dtype=tf.float32)
    future_valid = tf.expand_dims(future_valid, axis=-1)

    valid = tf.concat([past_valid, current_valid, future_valid], 2)
    states = tf.concat([states, valid], 3)

    return states


def parse_examples(examples):
    decoded_examples = tf.io.parse_example(examples, features_description)
    states, states_is_valid, sample_is_valid = utils.parse_states(decoded_examples, state_features)

    states = add_valid(decoded_examples, states)

    is_ego = decoded_examples['state/is_sdc'] > 0
    result = {
        'states': states,
        'states_is_valid': states_is_valid,
        'scenario_id': decoded_examples['scenario/id'],
        'sample_is_valid': sample_is_valid,
        'object_types': decoded_examples['state/type'],
        'object_ids': decoded_examples['state/id'],
        'is_ego': is_ego,
        'ego_states': states[is_ego],
        'tracks_to_predict': decoded_examples['state/tracks_to_predict'] > 0,
    }

    return result


def process_scenario(data_scenario):
    scenario_id = data_scenario['scenario_id'][0].decode()
    scenario_predictions = {}

    states = data_scenario['states']
    states_normalized = data_scenario['normalized_states']

    states_is_valid = data_scenario['states_is_valid']
    object_ids = data_scenario['object_ids']
    object_types = data_scenario['object_types']
    ego_states = data_scenario['ego_states']

    states_observed_normalized = states_normalized[:, :11, :]

    valid_RU_indexes = np.argwhere(object_types > -1).flatten()

    num_agents, num_steps, num_features = len(valid_RU_indexes), 16, 2
    shape = (num_agents, num_steps, num_features)
    predictions = np.zeros(shape)

    for RU_index in valid_RU_indexes:
        # ToDo: this can use some serious optimizations (e.g. calling the models in
        # batches instead of individually would be the first step)
        RU_type = object_types[RU_index]

        # states_RU = states[RU_index]
        states_is_valid_RU = states_is_valid[RU_index]

        if not np.any(states_is_valid_RU[:11]):
            continue  # no valid observed index, skip

        if not np.any(states_is_valid_RU[11:]):
            continue  # no valid future index, skip

        model_to_use = None
        if RU_type == 1:
            model_to_use = model_vehicles
        elif RU_type == 2:
            model_to_use = model_pedestrians
        elif RU_type == 3:
            model_to_use = model_cyclists

        RU_predictions = model_to_use.predict(states_observed_normalized[RU_index:RU_index + 1, :])
        predictions[RU_index] = RU_predictions

    predictions = tf.convert_to_tensor(predictions)
    predictions_batch = tf.expand_dims(predictions, axis=0)
    predictions_unnormalized = unnormalize(predictions_batch)
    predictions_unshifted = unshift(predictions_unnormalized, tf.expand_dims(ego_states, axis=0))[0]

    for RU_index in valid_RU_indexes:
        RU_ID = int(object_ids[RU_index])
        scenario_predictions[RU_ID] = predictions_unshifted[RU_index]

    savefolder = f"{config.PREDICTIONS_FOLDER}lstm_per_agent/"

    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savepath = savefolder + f"{scenario_id}.pkl"

    with open(savepath, 'wb') as f:
        pickle.dump(scenario_predictions, f)


def process_batch(data_batch):
    """
    Data batch has the following keys and shapes
    states (batch_size, num_agents, num_timesteps, num_features)
    states_is_valid (batch_size, num_agents, num_timesteps)
    scenario_id (batch_size, 1)
    sample_is_valid (batch_size, num_agents)
    """

    batch_size, num_agents, num_timesteps, num_features = data_batch['states'].shape

    p = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
    res = []

    for scenario_number in range(batch_size):
        data_scenario = {}
        for key in data_batch.keys():
            data_scenario[key] = data_batch[key][scenario_number]

        if MULTIPROCESSING:
            res.append(
                p.apply_async(
                    process_scenario,
                    kwds=dict(
                        data_scenario=data_scenario
                    )
                )
            )
        else:
            process_scenario(data_scenario)

        # break

    for r in res:
        r.get()

    p.close()
    p.join()


def _shift_positions(input_tensor, shifters, n_dims):
    states_np = input_tensor.numpy()
    states_np_pos = states_np[:, :, :, :n_dims]
    num_batches, num_samples, num_timesteps, num_features = states_np_pos.shape
    shifted_data = np.copy(states_np_pos)

    if shifters is None:
        feat_min = np.min(shifted_data, axis=2)
        feat_max = np.max(shifted_data, axis=2)
        shifters = (feat_min + ((feat_max - feat_min) / 2.0)).reshape(num_batches, num_samples, 1, num_features)

    shifted_data_pos = shifted_data - shifters
    states_np[:, :, :, :n_dims] = shifted_data_pos

    shifted_states = tf.convert_to_tensor(states_np)
    return shifted_states


def _shift_xyz(input_tensor, shifters=None):
    return _shift_positions(input_tensor, shifters, n_dims=3)


def _shift_xy(input_tensor, shifters=None):
    return _shift_positions(input_tensor, shifters, n_dims=2)


# centers all trajectories around (0,0,0)
def shift_around_own(decoded_example):
    states = decoded_example["states"]  # 4D - batch, sample, time, feature
    shifted_states = tf.py_function(_shift_xyz, inp=[states], Tout=states.dtype)
    shifted_states.set_shape(states.shape)  # !important
    decoded_example["shifted_states"] = shifted_states

    return decoded_example


# shifts trajectories (x,y,z) such that they are relative to the  current ego's position.
def shift_relative_to_host(decoded_example):
    states = decoded_example["states"]  # 4D - batch, sample, time, feature
    # num_batches, num_samples, num_timesteps, num_features = states.shape

    ego_states = decoded_example["ego_states"]
    host_xyz_current = ego_states[:, 10:11, :3]
    host_xyz_current = tf.expand_dims(host_xyz_current, axis=1)

    shifted_states = tf.py_function(_shift_xyz, inp=[states, host_xyz_current], Tout=states.dtype)
    shifted_states.set_shape(states.shape)  # !important
    decoded_example["shifted_states"] = shifted_states

    return decoded_example


def _normalize_tensor(input_tensor, unnormalize=False):
    states_np = input_tensor.numpy()
    num_batches, num_samples, num_timesteps, num_features = states_np.shape
    result = np.copy(states_np)

    dim1 = num_batches * num_samples * num_timesteps
    result = result.reshape((dim1, num_features))

    # use all scalers unless there are fewer features (e.g. predictions are only x,y)
    max_scalers = num_features if num_features < len(scalers) else len(scalers)

    if unnormalize:
        for i, scaler in enumerate(scalers[:max_scalers]):
            result[:, i] = scaler.inverse_transform(result[:, i].reshape(-1, 1)).reshape(dim1, )
        result = result.reshape((num_batches, num_samples, num_timesteps, num_features))
        return result

    for i, scaler in enumerate(scalers[:max_scalers]):
        result[:, i] = scaler.transform(result[:, i].reshape(-1, 1)).reshape(dim1, )
    result = result.reshape((num_batches, num_samples, num_timesteps, num_features))

    return result


def normalize(decoded_example):
    states = decoded_example["shifted_states"]  # 4D - batch, sample, time, feature
    shifted_normalized_states = tf.py_function(_normalize_tensor, inp=[states, False], Tout=states.dtype)
    shifted_normalized_states.set_shape(states.shape)  # !important

    decoded_example["normalized_states"] = shifted_normalized_states

    return decoded_example


def unnormalize(states):
    shifted_unnormalized_states = tf.py_function(_normalize_tensor, inp=[states, True], Tout=states.dtype)
    shifted_unnormalized_states.set_shape(states.shape)  # !important
    return shifted_unnormalized_states


def unshift(states, ego_states):
    host_xy_current = ego_states[:, 10:11, :2]
    host_xy_current = tf.expand_dims(host_xy_current, axis=1)
    unshifted_states = tf.py_function(_shift_xy, inp=[states, -host_xy_current], Tout=states.dtype)
    unshifted_states.set_shape(states.shape)  # !important
    return unshifted_states


def main():
    files = [config.TEST_FOLDER + file for file in sorted(os.listdir(config.TEST_FOLDER))]
    num_parallel_calls, prefetch_size = tf.data.AUTOTUNE, tf.data.AUTOTUNE
    batch_size = config.BATCH_SIZE

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_examples, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(shift_relative_to_host, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(normalize, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(prefetch_size)

    for data_batch in tqdm(dataset.as_numpy_iterator()):
        process_batch(data_batch)


if __name__ == "__main__":
    main()
