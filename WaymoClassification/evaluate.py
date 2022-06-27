import pickle
import tensorflow as tf
from tqdm import tqdm
import config
import utils
import numpy as np
import os
from models.waymo import WaymoTag

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

features_description = utils.get_features_description()
state_features = ["x", "y"]
MULTIPROCESSING = False


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


def load_ground_truth_future(scenario_id, downsampler=5):
    path = f"{config.PREDICTIONS_FOLDER}gt/{scenario_id}.pkl"
    with open(path, 'rb') as f:
        gt = pickle.load(f)

    if downsampler > 1:
        gt_downsampled = {}
        for ru_id, gt_dict in gt.items():
            states = gt_dict["states"]
            states_is_valid = gt_dict["states_is_valid"]
            gt_downsampled[ru_id] = {
                "states": states[11::downsampler],
                "states_is_valid": states_is_valid[11::downsampler]
            }
        return gt_downsampled
    return gt


def load_predictions(scenario_id, model, downsampler=1):
    path = f"{config.PREDICTIONS_FOLDER}{model}/{scenario_id}.pkl"
    with open(path, 'rb') as f:
        predictions = pickle.load(f)

    predictions_downsampled = {}
    if downsampler > 1:
        if model == "motioncnn":
            for ru_id, predictions in predictions.items():
                ru_predictions_downsampled = []
                for prediction in predictions:
                    conf = prediction["confidence"]
                    pred = prediction["prediction"][::downsampler]
                    ru_predictions_downsampled.append({
                        "confidence": conf,
                        "prediction": pred
                    })

                predictions_downsampled[ru_id] = ru_predictions_downsampled
        else:
            for ru_id, prediction in predictions.items():
                predictions_downsampled[ru_id] = prediction[::downsampler]

        return predictions_downsampled

    return predictions


def get_scenario_tags(scenario_id):
    tags_path = f"{config.TAGS_FOLDER}{scenario_id}.pkl"
    with open(tags_path, 'rb') as f:
        tags = pickle.load(f)
    return tags


def process_scenario(data_scenario, tag=None):
    scenario_id = data_scenario['scenario_id'][0].decode()

    object_ids = data_scenario['object_ids']
    object_types = data_scenario['object_types']

    scenario_tags = get_scenario_tags(scenario_id)
    gt = load_ground_truth_future(scenario_id, downsampler=5)
    cv_predictions = load_predictions(scenario_id, "cv", downsampler=5)
    motioncnn_predictions = load_predictions(scenario_id, "motioncnn", downsampler=5)
    lstmV1_per_agent_predictions = load_predictions(scenario_id, "lstm_per_agent")

    scenario_errors_per_model_per_ru = {
        "cv": {},
        "lstmV1_per_agent": {},
        "motionCNN_top1": {},
        "motionCNN_top6": {},
    }

    # rus = []
    int2ru = {
        1: "vehicles",
        2: "pedestrians",
        3: "cyclists",
    }

    first_ru = list(cv_predictions.keys())[0]
    timesteps = cv_predictions[first_ru].shape[0]

    for ru_id, _ in cv_predictions.items():
        ru_type = int2ru[object_types[object_ids == ru_id][0]]

        if tag is not None and tag not in scenario_tags[ru_id]:
            continue

        cv_errors = np.abs(cv_predictions[ru_id] - gt[ru_id]["states"])
        lstmV1_per_agent_errors = np.abs(lstmV1_per_agent_predictions[ru_id] - gt[ru_id]["states"])
        motioncnn_errors = np.abs(motioncnn_predictions[ru_id][0]["prediction"] - gt[ru_id]["states"])
        motioncnn_top6_errors = np.array(
            [np.abs(motioncnn_predictions[ru_id][k]["prediction"] - gt[ru_id]["states"]) for k in range(6)])

        # print("confidences: ", [np.abs(motioncnn_predictions[ru_id][k]["confidence"]) for k in range(6)])

        # print("!!shape!", motioncnn_errors.shape)
        # print("!!shape!", motioncnn_top6_errors.shape)
        # exit()

        # motioncnn_top6_errors

        # models = ["cv", "lstmV1_per_agent", "motionCNN_top1"]
        models = ["cv", "lstmV1_per_agent", "motionCNN_top1", "motionCNN_top6"]
        # models = list(scenario_errors_per_model_per_ru.keys())
        errors = [cv_errors, lstmV1_per_agent_errors, motioncnn_errors, motioncnn_top6_errors]

        for i, model in enumerate(models):
            if ru_type not in scenario_errors_per_model_per_ru[model]:
                scenario_errors_per_model_per_ru[model][ru_type] = [[] for _ in range(timesteps)]

            for timestep in range(timesteps):
                if gt[ru_id]['states_is_valid'][timestep]:
                    if model == "motionCNN_top6":
                        errorsK = [errors[i][k][timestep].tolist() for k in range(6)]
                        # print("Timestep: ", timestep)
                        # print(len(errorsK[0]))
                        # print(errorsK[0])
                        # print(model, timestep, np.array(errorsK).shape)
                        if len(errorsK) < 6:
                            print("!!!!!!!!!!!!")
                            raise ValueError
                            exit()

                        scenario_errors_per_model_per_ru[model][ru_type][timestep].append(errorsK)
                    else:
                        # print(model, timestep, np.array(errors[i][timestep].tolist()).shape)
                        scenario_errors_per_model_per_ru[model][ru_type][timestep].append(errors[i][timestep].tolist())

    # exit()
    return scenario_errors_per_model_per_ru


def process_batch(data_batch, tag=None):
    """
    Data batch has the following keys and shapes
    states (batch_size, num_agents, num_timesteps, num_features)
    states_is_valid (batch_size, num_agents, num_timesteps)
    scenario_id (batch_size, 1)
    sample_is_valid (batch_size, num_agents)
    """

    batch_size, num_agents, num_timesteps, num_features = data_batch['states'].shape

    batch_errors_per_model_per_ru = {}

    for scenario_number in range(batch_size):
        data_scenario = {}
        for key in data_batch.keys():
            data_scenario[key] = data_batch[key][scenario_number]

        scenario_errors_per_model_per_ru = process_scenario(data_scenario, tag=tag)
        extend_errors(batch_errors_per_model_per_ru, scenario_errors_per_model_per_ru)

    return batch_errors_per_model_per_ru


def extend_errors(errors_to_extend, errors):
    timesteps = 16
    for model, batch_model_errors in errors.items():
        if model not in errors_to_extend:
            errors_to_extend[model] = {}

        for ru_type, ru_type_errors in batch_model_errors.items():
            if ru_type not in errors_to_extend[model]:
                errors_to_extend[model][ru_type] = [[] for _ in range(timesteps)]

            for t in range(timesteps):
                errors_to_extend[model][ru_type][t].extend(ru_type_errors[t])


def compute_and_save_errors(tags=None):
    if tags is not None:  # compute for all
        compute_and_save_errors(tags=None)

    FILES = [config.TEST_FOLDER + file for file in sorted(os.listdir(config.TEST_FOLDER))]
    num_parallel_calls, prefetch_size = tf.data.AUTOTUNE, tf.data.AUTOTUNE
    batch_size = config.BATCH_SIZE

    dataset = tf.data.TFRecordDataset(FILES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_examples, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(prefetch_size)

    if tags is None:
        print("Compute overall errors")
        all_errors = {}

        for data_batch in tqdm(dataset.as_numpy_iterator()):
            batch_errors = process_batch(data_batch)
            extend_errors(all_errors, batch_errors)

        if not os.path.exists(config.EVALUATIONS_FOLDER):
            os.mkdir(config.EVALUATIONS_FOLDER)

        with open(f"{config.EVALUATIONS_FOLDER}evaluation_errors.pkl", "wb") as f:
            pickle.dump(all_errors, f)

    else:
        print("Compute errors per tag")
        for tag in tqdm(tags):
            all_errors = {}

            for data_batch in dataset.as_numpy_iterator():
                batch_errors = process_batch(data_batch, tag=tag)
                extend_errors(all_errors, batch_errors)

            with open(f"{config.EVALUATIONS_FOLDER}evaluation_errors_{tag.name}.pkl", "wb") as f:
                pickle.dump(all_errors, f)


def compute_metrics_from_errors(errors_path):
    with open(errors_path, "rb") as f:
        all_errors = pickle.load(f)

    metrics = {}

    for model, model_errors in all_errors.items():
        if model not in metrics:
            metrics[model] = {}

        for ru_type, ru_type_errors in model_errors.items():
            if ru_type not in metrics[model]:
                metrics[model][ru_type] = {}
            num_timesteps = len(ru_type_errors)

            metrics[model][ru_type]["FDE"] = []
            metrics[model][ru_type]["ADE"] = []

            timesteps_euclidian_distances = []

            for timestep in range(num_timesteps):
                if model == "motionCNN_top6":
                    # print("!! ru_type_errors", ru_type_errors)
                    # print("!! len ru_type_errors", len(ru_type_errors))
                    # print("!! len ru_type_errors[t=0]", len(ru_type_errors[0]))
                    # print("!! len ru_type_errors[t=0][k=0]", len(ru_type_errors[0][0]))
                    # print("!! ru_type_errors", ru_type_errors.shape)
                    timestep_ru_type_errors_np = np.array(ru_type_errors[timestep])
                    # print(timestep_ru_type_errors_np.shape)

                    timestep_xy_errors = [timestep_ru_type_errors_np[:, k] for k in range(6)]
                    if len(timestep_xy_errors[0]) == 0:  # no data points to compute errors
                        metrics[model][ru_type]["FDE"].append([np.NaN, np.NaN, np.NaN, np.NaN])
                        metrics[model][ru_type]["ADE"].append([np.NaN, np.NaN, np.NaN, np.NaN])
                        continue

                    timestep_euclidian_distances = []
                    timestep_FDEs = []
                    timestep_FDEs_std = []
                    timestep_FDEs_min = []
                    timestep_FDEs_max = []

                    for k in range(6):
                        timestep_euclidian_distances.append(np.linalg.norm(timestep_xy_errors[k], axis=1))
                        timestep_FDEs.append(timestep_euclidian_distances[k].mean())
                        timestep_FDEs_std.append(timestep_euclidian_distances[k].std())
                        timestep_FDEs_min.append(timestep_euclidian_distances[k].min())
                        timestep_FDEs_max.append(timestep_euclidian_distances[k].max())

                    timesteps_euclidian_distances.append(timestep_euclidian_distances)
                    min_index = np.argmin(timestep_FDEs).flatten()[0]
                    # print("!! min index",min_index)
                    # timesteps_euclidian_distances.append(timestep_euclidian_distances[min_index])

                    errs = [timestep_FDEs[min_index], timestep_FDEs_std[min_index], timestep_FDEs_min[min_index],
                            timestep_FDEs_max[min_index]]
                    metrics[model][ru_type]["FDE"].append(errs)

                    # ADE
                    num_steps = len(timesteps_euclidian_distances)
                    timestep_ADEs = []
                    timestep_ADEs_std = []
                    timestep_ADEs_min = []
                    timestep_ADEs_max = []

                    for k in range(6):
                        all_previous_distances = []
                        for t in range(num_steps):
                            all_previous_distances.extend(timesteps_euclidian_distances[t][k])
                        all_previous_distances = np.array(all_previous_distances)

                        timestep_ADEs.append(all_previous_distances.mean())
                        timestep_ADEs_std.append(all_previous_distances.std())
                        timestep_ADEs_min.append(all_previous_distances.min())
                        timestep_ADEs_max.append(all_previous_distances.max())

                    min_index = np.argmin(timestep_ADEs).flatten()[0]
                    errs = [timestep_ADEs[min_index], timestep_ADEs_std[min_index], timestep_ADEs_min[min_index],
                            timestep_ADEs_max[min_index]]
                    metrics[model][ru_type]["ADE"].append(errs)


                else:
                    timestep_xy_errors = np.array(ru_type_errors[timestep])

                    if len(timestep_xy_errors) == 0:  # no data points to compute errors
                        metrics[model][ru_type]["FDE"].append([np.NaN, np.NaN, np.NaN, np.NaN])
                        metrics[model][ru_type]["ADE"].append([np.NaN, np.NaN, np.NaN, np.NaN])
                        continue

                    timestep_euclidian_distances = np.linalg.norm(timestep_xy_errors, axis=1)
                    timesteps_euclidian_distances.append(timestep_euclidian_distances)

                    # FDE
                    timestep_FDE = timestep_euclidian_distances.mean()
                    timestep_FDE_std = timestep_euclidian_distances.std()
                    timestep_FDE_min = timestep_euclidian_distances.min()
                    timestep_FDE_max = timestep_euclidian_distances.max()
                    metrics[model][ru_type]["FDE"].append(
                        [timestep_FDE, timestep_FDE_std, timestep_FDE_min, timestep_FDE_max])

                    # ADE
                    num_steps = len(timesteps_euclidian_distances)
                    all_previous_distances = []
                    for t in range(num_steps):
                        all_previous_distances.extend(timesteps_euclidian_distances[t])
                    all_previous_distances = np.array(all_previous_distances)

                    timestep_ADE = all_previous_distances.mean()
                    timestep_ADE_std = all_previous_distances.std()
                    timestep_ADE_min = all_previous_distances.min()
                    timestep_ADE_max = all_previous_distances.max()
                    metrics[model][ru_type]["ADE"].append(
                        [timestep_ADE, timestep_ADE_std, timestep_ADE_min, timestep_ADE_max])

    return metrics


def save_obj(obj, savepath):
    with open(savepath, "wb") as f:
        pickle.dump(obj, f)


def compute_and_save_metrics(tags=None):
    if tags is not None:  # compute for all
        compute_and_save_metrics(tags=None)

    if tags is None:
        print("Compute overall metrics")
        errors_path = f"{config.EVALUATIONS_FOLDER}evaluation_errors.pkl"
        savepath = f"{config.EVALUATIONS_FOLDER}evaluation_metrics.pkl"
        metrics = compute_metrics_from_errors(errors_path)
        save_obj(metrics, savepath)
        # pprint(metrics)
    else:
        print("Compute metrics per tag")
        for tag in tqdm(tags):
            print(tag.name)
            errors_path = f"{config.EVALUATIONS_FOLDER}evaluation_errors_{tag.name}.pkl"
            savepath = f"{config.EVALUATIONS_FOLDER}evaluation_metrics_{tag.name}.pkl"
            metrics = compute_metrics_from_errors(errors_path)
            save_obj(metrics, savepath)


def get_tags_to_evaluate():
    tags = [
        ##### Trajectory properties
        # Speed & Velocity profile
        # WaymoTag.TrajectoryProperty_Velocity_Constant,
        # WaymoTag.TrajectoryProperty_Velocity_Variable,
        # WaymoTag.TrajectoryProperty_Speed_Constant,
        # WaymoTag.TrajectoryProperty_Speed_Variable,

        # Curvature
        WaymoTag.TrajectoryProperty_Curvature_Straight,
        WaymoTag.TrajectoryProperty_Curvature_NonStraight,

        # Behavior
        WaymoTag.TrajectoryProperty_Behavior_Starting,
        WaymoTag.TrajectoryProperty_Behavior_Stopping,
        WaymoTag.TrajectoryProperty_Behavior_Still,

        # Relevance to ego vehicle
        # WaymoTag.TrajectoryProperty_Relevance_Relevant,
        # WaymoTag.TrajectoryProperty_Relevance_Irrelevant,

        # How much of the trajectory was observed
        WaymoTag.TrajectoryProperty_Observation_Late,
        WaymoTag.TrajectoryProperty_Observation_VeryLate,
        WaymoTag.TrajectoryProperty_Observation_Full,
        WaymoTag.TrajectoryProperty_Observation_Reappear,

        # WaymoTag.TrajectoryProperty_ObservationLength_None,

        WaymoTag.TrajectoryProperty_TrackToPredict_Yes,
        WaymoTag.TrajectoryProperty_TrackToPredict_No,
    ]

    return tags


if __name__ == "__main__":
    tags = get_tags_to_evaluate()
    compute_and_save_errors(tags=tags)
    compute_and_save_metrics(tags=tags)
