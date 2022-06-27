import multiprocessing
import os.path
import pickle
import time
import tensorflow as tf
from tqdm import tqdm
import config
import utils
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

features_description = utils.get_features_description()
state_features = ["x", "y"]
MULTIPROCESSING = True


def parse_examples(examples):
    decoded_examples = tf.io.parse_example(examples, features_description)
    states, states_is_valid, sample_is_valid = utils.parse_states(decoded_examples, state_features)

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
    scenario_id = data_scenario['scenario_id'][0].decode()
    scenario_gt = {}

    states = data_scenario['states']
    states_is_valid = data_scenario['states_is_valid']
    object_ids = data_scenario['object_ids']
    object_types = data_scenario['object_types']

    valid_RU_indexes = np.argwhere(object_ids > -1).flatten()
    for RU_index in valid_RU_indexes:
        RU_ID = int(object_ids[RU_index])
        states_RU = states[RU_index]
        states_is_valid_RU = states_is_valid[RU_index]

        scenario_gt[RU_ID] = {
            'states': states_RU,
            'states_is_valid': states_is_valid_RU,
            'type': object_types[RU_index]
        }

    savefolder = f"{config.PREDICTIONS_FOLDER}gt/"

    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savepath = f"{savefolder}{scenario_id}.pkl"
    with open(savepath, 'wb') as f:
        pickle.dump(scenario_gt, f)


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


def main():
    FILES = [config.TEST_FOLDER + file for file in sorted(os.listdir(config.TEST_FOLDER))]
    num_parallel_calls, prefetch_size = tf.data.AUTOTUNE, tf.data.AUTOTUNE
    batch_size = config.BATCH_SIZE

    dataset = tf.data.TFRecordDataset(FILES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_examples, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(prefetch_size)

    for data_batch in tqdm(dataset.as_numpy_iterator()):
        process_batch(data_batch)


if __name__ == "__main__":
    main()
