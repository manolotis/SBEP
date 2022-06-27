import numpy as np
import torch
from tqdm import tqdm
import pickle
import os
from torch.utils.data import DataLoader, Dataset
import timm
from WaymoClassification import config

IN_CHANNELS = 47
TL = 80
N_TRAJS = 8


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        if self.is_test:
            center = data["shift"]
            yaw = data["yaw"]
            agent_id = data["object_id"]
            scenario_id = data["scenario_id"]

            return (
                raster,
                center,
                yaw,
                agent_id,
                str(scenario_id),
                data["_gt_marginal"],
                data["gt_marginal"],
            )

        trajectory = data["gt_marginal"]

        is_available = data["future_val_marginal"]

        if self.return_vector:
            return raster, trajectory, is_available, data["vector_data"]

        return raster, trajectory, is_available


def main():
    model_path = config.MOTION_CNN_PATH
    batch_size = 8

    test_data = config.MOTION_CNN_TEST_DATA
    save_folder = config.PREDICTIONS_FOLDER + "motioncnn/"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    model = timm.create_model(
        "xception71",
        pretrained=True,
        in_chans=IN_CHANNELS,
        num_classes=N_TRAJS * 2 * TL + N_TRAJS,
    )
    model.load_state_dict(torch.load(model_path)["model_state_dict"])

    model.cuda()
    model.eval()

    dataset = WaymoLoader(test_data, is_test=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=min(batch_size, 16))

    RES = {}

    with torch.no_grad():
        for x, center, yaw, agent_id, scenario_id, _, _ in tqdm(loader):
            # print("x shape", x.shape)
            x = x.cuda()

            # confidences_logits, logits = model(x)
            outputs = model(x)

            confidences_logits, logits = (
                outputs[:, : N_TRAJS],
                outputs[:, N_TRAJS:],
            )
            logits = logits.view(-1, N_TRAJS, TL, 2)

            confidences = torch.softmax(confidences_logits, dim=1)

            logits = logits.cpu().numpy()
            confidences = confidences.cpu().numpy()
            agent_id = agent_id.cpu().numpy()
            center = center.cpu().numpy()
            yaw = yaw.cpu().numpy()
            for p, conf, aid, sid, c, y in zip(
                    logits, confidences, agent_id, scenario_id, center, yaw
            ):
                if sid not in RES:
                    RES[sid] = []

                RES[sid].append(
                    {"aid": aid, "conf": conf, "pred": p, "yaw": -y, "center": c}
                )

    # selector = np.arange(4, args.time_limit + 1, 5) # downsample timesteps
    selector = np.arange(0, 80, 1)  # downsample timesteps or select all

    for scenario_id, data in tqdm(RES.items()):

        scenario_predictions = {}

        for d in data:
            object_id = int(d["aid"])

            y = d["yaw"]
            rot_matrix = np.array([
                [np.cos(y), -np.sin(y)],
                [np.sin(y), np.cos(y)],
            ])

            object_predictions = []

            for i in np.argsort(-d["conf"]):  # sorted by condfidence
                p = d["pred"][i][selector] @ rot_matrix + d["center"]

                object_predictions.append({
                    "confidence": d["conf"][i],
                    "prediction": p[:, :2]
                })

                # object_predictions[] = prediction

            scenario_predictions[object_id] = object_predictions

        savepath = save_folder + scenario_id + ".pkl"
        # print(scenario_predictions)
        with open(savepath, "wb") as f:
            pickle.dump(scenario_predictions, f)
            # print("saved ", savepath)
            # exit()


if __name__ == "__main__":
    main()
