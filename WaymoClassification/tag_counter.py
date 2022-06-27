import multiprocessing
import os.path
import pickle
import time
import tensorflow as tf
from tqdm import tqdm
import config
import utils
from models.waymo import WaymoScenario, WaymoTag

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


def get_scenario_tags(path, scenario_id=None):
    scenario_ttp_tags = {}
    scenario_all_tags = {}

    with open(path, 'rb') as f:
        scenario_tags = pickle.load(f)

    for RU_ID, tags in scenario_tags.items():
        for tag in tags:
            if tag not in scenario_all_tags:
                scenario_all_tags[tag] = 0

            scenario_all_tags[tag] += 1

            if WaymoTag.TrajectoryProperty_TrackToPredict_Yes in tags:
                if tag not in scenario_ttp_tags:
                    scenario_ttp_tags[tag] = 0
                scenario_ttp_tags[tag] += 1

    return scenario_all_tags, scenario_ttp_tags


def count_scenario_tags(data_scenario):
    scenario_id = data_scenario['scenario_id'][0].decode()
    tags_path = config.TAGS_FOLDER + scenario_id + ".pkl"
    scenario_all_tags, scenario_ttp_tags = get_scenario_tags(tags_path, scenario_id)

    return scenario_all_tags, scenario_ttp_tags


def process_batch(data_batch):
    """
    Data batch has the following keys and shapes
    states (batch_size, num_agents, num_timesteps, num_features)
    states_is_valid (batch_size, num_agents, num_timesteps)
    scenario_id (batch_size, 1)
    sample_is_valid (batch_size, num_agents)
    """

    batch_size, num_agents, num_timesteps, num_features = data_batch['states'].shape
    batch_all_tags, batch_ttp_tags = {}, {}

    p = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
    res = []

    for scenario_number in range(batch_size):
        data_scenario = {}
        for key in data_batch.keys():
            data_scenario[key] = data_batch[key][scenario_number]

        res.append(
            p.apply_async(
                count_scenario_tags,
                kwds=dict(
                    data_scenario=data_scenario
                )
            )
        )

        # break

    for r in res:
        scenario_all_tags, scenario_ttp_tags = r.get()

        for tag, count in scenario_all_tags.items():
            if tag not in batch_all_tags:
                batch_all_tags[tag] = 0

            batch_all_tags[tag] += count

        for tag, count in scenario_ttp_tags.items():
            if tag not in batch_ttp_tags:
                batch_ttp_tags[tag] = 0

            batch_ttp_tags[tag] += count

    p.close()
    p.join()

    return batch_all_tags, batch_ttp_tags


def count_split(split):
    split2files = {
        "training": [config.TRAIN_FOLDER + file for file in sorted(os.listdir(config.TRAIN_FOLDER))],
        "validation": [config.VALIDATION_FOLDER + file for file in sorted(os.listdir(config.VALIDATION_FOLDER))],
        "testing": [config.TEST_FOLDER + file for file in sorted(os.listdir(config.TEST_FOLDER))],
    }

    FILES = split2files[split]

    num_parallel_calls, prefetch_size = tf.data.AUTOTUNE, tf.data.AUTOTUNE
    batch_size = config.BATCH_SIZE

    dataset = tf.data.TFRecordDataset(FILES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_examples, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(prefetch_size)

    all_tags, ttp_tags = {}, {}

    for data_batch in tqdm(dataset.as_numpy_iterator()):
        batch_all_tags, batch_ttp_tags = process_batch(data_batch)

        for tag, count in batch_all_tags.items():
            if tag not in all_tags:
                all_tags[tag] = 0

            all_tags[tag] += count

        for tag, count in batch_ttp_tags.items():
            if tag not in ttp_tags:
                ttp_tags[tag] = 0

            ttp_tags[tag] += count

    savefolder = config.TAG_COUNTS_FOLDER

    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savepath = f"{savefolder}tag_counts_{split}.pkl"

    with open(savepath, 'wb') as f:
        pickle.dump(all_tags, f)
    savepath = f"{savefolder}tag_ttp_counts_{split}.pkl"
    with open(savepath, 'wb') as f:
        pickle.dump(ttp_tags, f)

    print("Tags", all_tags)
    print("TTP tags", ttp_tags)


def main():
    count_split("training")
    count_split("validation")
    count_split("testing")


if __name__ == "__main__":
    main()
