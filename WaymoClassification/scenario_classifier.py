import config
import os
import tensorflow as tf
from tqdm import tqdm
import utils
import multiprocessing
from WaymoClassification.models.waymo import WaymoScenario

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

features_description = utils.get_features_description()
state_features = ["x", "y", "z", "bbox_yaw", "length", "width", "height", "speed", "timestamp_micros", "vel_yaw",
                  "velocity_x", "velocity_y"]  # important: don't change order


def _parse_states(decoded_example):
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


def parse_examples(examples):
    decoded_examples = tf.io.parse_example(examples, features_description)
    states, states_is_valid, sample_is_valid = _parse_states(decoded_examples)

    result = {
        'states': states,
        'states_is_valid': states_is_valid,
        'scenario_id': decoded_examples['scenario/id'],
        'sample_is_valid': sample_is_valid,
        'object_types': decoded_examples['state/type'],
        'object_ids': decoded_examples['state/id'],
        'is_ego': decoded_examples['state/is_sdc'] > 0,
        'tracks_to_predict': decoded_examples['state/tracks_to_predict'] > 0,
    }

    return result


def process_scenario(data_scenario):
    scenario = WaymoScenario(data_scenario, config)
    scenario.save_tags()


def process_batch(data_batch):
    """
    Data batch has the following keys and shapes
    states (batch_size, num_agents, num_timesteps, num_features)
    states_is_valid (batch_size, num_agents, num_timesteps)
    scenario_id (batch_size, 1)
    sample_is_valid (batch_size, num_agents)
    """

    batch_size, num_agents, num_timesteps, num_features = data_batch['states'].shape

    p = multiprocessing.Pool(config.N_PROCESSES)
    res = []

    for scenario_number in range(batch_size):
        data_scenario = {}
        for key in data_batch.keys():
            data_scenario[key] = data_batch[key][scenario_number]

        if config.MULTIPROCESSING:
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

    for r in res:
        r.get()

    p.close()
    p.join()


def classify_split(split_path):
    print("Processing split: ", split_path)
    files = [split_path + file for file in sorted(os.listdir(split_path))]

    num_parallel_calls, prefetch_size = tf.data.AUTOTUNE, tf.data.AUTOTUNE
    batch_size = config.BATCH_SIZE

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_examples, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(prefetch_size)

    for data_batch in tqdm(dataset.as_numpy_iterator()):
        process_batch(data_batch)

def main():
    classify_split(config.TRAIN_FOLDER)
    classify_split(config.VALIDATION_FOLDER)
    classify_split(config.TEST_FOLDER)


if __name__ == "__main__":
    main()
