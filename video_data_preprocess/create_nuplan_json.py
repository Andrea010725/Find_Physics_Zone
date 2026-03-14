import argparse
import json
import os

import tqdm
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--split_id", default=0, type=int)
    parser.add_argument("--split_num", default=1, type=int)
    parser.add_argument("--split_name", default="mini", choices=["mini", "trainval", "test"])
    parser.add_argument("--nuplan_data_root", type=str, required=True)
    parser.add_argument("--maps_root", type=str, required=True)
    parser.add_argument("--map_version", type=str, default="nuplan-maps-v1.0")
    parser.add_argument("--sensor_blobs_root", type=str, default=None)
    parser.add_argument("--db_root", type=str, default=None)
    parser.add_argument("--ego_save_dir", type=str, required=True)
    parser.add_argument("--seq_save_dir", type=str, required=True)
    parser.add_argument("--db_names", nargs="*", default=None)
    return parser.parse_args()


def resolve_sensor_blobs_root(args):
    if args.sensor_blobs_root is not None:
        return args.sensor_blobs_root

    candidates = [
        os.path.join(args.nuplan_data_root, "sensor_blobs"),
        os.path.join(args.nuplan_data_root, "data", "sensor_blobs"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("Cannot resolve sensor_blobs_root from nuplan_data_root.")


def resolve_db_root(args):
    if args.db_root is not None:
        return args.db_root

    candidates = [
        os.path.join(args.nuplan_data_root, "splits", args.split_name),
        os.path.join(args.nuplan_data_root, "data", "splits", args.split_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("Cannot resolve db_root from nuplan_data_root.")


def load_single_log_db_data(log_db_name, log_db, sensor_blobs_root):
    print("In load_single_log_db")
    images_data = log_db.image
    img_root_path = os.path.join(sensor_blobs_root, log_db_name)

    cameras = {
        "CAM_L2": [],
        "CAM_F0": [],
        "CAM_R2": [],
        "CAM_L0": [],
        "CAM_L1": [],
        "CAM_R0": [],
        "CAM_R1": [],
        "CAM_B0": [],
        "data_root": img_root_path,
    }
    seq_tmp = []

    ego_pose = {
        "CAM_L2": {},
        "CAM_F0": {},
        "CAM_R2": {},
        "CAM_L0": {},
        "CAM_L1": {},
        "CAM_R0": {},
        "CAM_R1": {},
        "CAM_B0": {},
    }

    pre_scene_token = None
    for idx, img_item in enumerate(images_data):
        camera_channel = img_item.camera.channel
        img_name = os.path.basename(img_item.filename_jpg)
        loaded_db_name = img_item.filename_jpg.split("/")[0]
        assert loaded_db_name == log_db_name

        scene_token = img_item.lidar_pc.scene_token
        img_path = os.path.join(sensor_blobs_root, img_item.filename_jpg)
        if not os.path.exists(img_path):
            print(f"!!!!WARNING: {img_path} does not exist.")
            continue

        curr_ego_pose = img_item.lidar_pc.ego_pose
        ego_pose[camera_channel][f"{camera_channel}/{img_name}"] = {
            "x": curr_ego_pose.x,
            "y": curr_ego_pose.y,
            "z": curr_ego_pose.z,
            "qw": curr_ego_pose.qw,
            "qx": curr_ego_pose.qx,
            "qy": curr_ego_pose.qy,
            "qz": curr_ego_pose.qz,
            "vx": curr_ego_pose.vx,
            "vy": curr_ego_pose.vy,
            "ax": curr_ego_pose.acceleration_x,
            "ay": curr_ego_pose.acceleration_y,
            "timestamp": curr_ego_pose.timestamp,
        }

        if camera_channel != "CAM_F0":
            continue

        if (scene_token != pre_scene_token) and (pre_scene_token is not None):
            cameras["CAM_F0"].append({"seq": seq_tmp, "scene": pre_scene_token})
            seq_tmp = [img_name]
        else:
            seq_tmp.append(img_name)
        pre_scene_token = scene_token

    if seq_tmp:
        cameras["CAM_F0"].append({"seq": seq_tmp, "scene": pre_scene_token})

    return cameras, ego_pose


def loop_over_db_files(split_db_list, sensor_blobs_root, ego_save_dir, seq_save_dir, nuplandb_wrapper):
    total_sequence = []
    sensor_data_db_list = sorted(os.listdir(sensor_blobs_root))
    print("In Loop over db.")

    for idx, db_name in tqdm.tqdm(enumerate(split_db_list), total=len(split_db_list)):
        print("id", idx)
        if db_name not in sensor_data_db_list:
            print(f"!!!! {db_name} does not exist in sensor blobs.")
            continue

        print(f">>>>Start {db_name}.")
        log_db = nuplandb_wrapper.get_log_db(db_name)
        cameras, ego_pose = load_single_log_db_data(db_name, log_db, sensor_blobs_root)

        scene_num = len(cameras["CAM_F0"])
        db_sequences = []
        for i in range(scene_num):
            new_seq_meta = {
                "CAM_F0": cameras["CAM_F0"][i]["seq"],
                "scene": cameras["CAM_F0"][i]["scene"],
                "data_root": cameras["data_root"],
                "pose": f"ego_meta/{db_name}.json",
            }
            db_sequences.append(new_seq_meta)
            total_sequence.append(new_seq_meta)

        ego_meta_path = os.path.join(ego_save_dir, f"{db_name}.json")
        os.makedirs(os.path.dirname(ego_meta_path), exist_ok=True)
        with open(ego_meta_path, "w") as f:
            json.dump(ego_pose, f)

        seq_meta_path = os.path.join(seq_save_dir, f"{db_name}.json")
        os.makedirs(os.path.dirname(seq_meta_path), exist_ok=True)
        with open(seq_meta_path, "w") as f:
            json.dump(db_sequences, f)

    return total_sequence


def main():
    args = add_arguments()
    sensor_blobs_root = resolve_sensor_blobs_root(args)
    db_root = resolve_db_root(args)

    if args.db_names:
        db_list = sorted(args.db_names)
    else:
        db_list = sorted(
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(db_root)
            if file_name.endswith(".db")
        )

    split_db_list = db_list[args.split_id::args.split_num]
    db_path_lists = [
        os.path.join(db_root, f"{db_name}.db")
        for db_name in split_db_list
        if os.path.exists(os.path.join(db_root, f"{db_name}.db"))
    ]
    split_db_list_update = [
        os.path.splitext(os.path.basename(db_path))[0]
        for db_path in db_path_lists
    ]

    print("sensor_blobs_root =", sensor_blobs_root)
    print("db_root =", db_root)
    print("num dbs =", len(db_list))
    print("num selected dbs =", len(split_db_list_update))

    nuplandb_wrapper = NuPlanDBWrapper(
        data_root=args.nuplan_data_root,
        map_root=args.maps_root,
        db_files=db_path_lists,
        map_version=args.map_version,
    )

    print("Start loop.")
    loop_over_db_files(
        split_db_list=split_db_list_update,
        sensor_blobs_root=sensor_blobs_root,
        ego_save_dir=args.ego_save_dir,
        seq_save_dir=args.seq_save_dir,
        nuplandb_wrapper=nuplandb_wrapper,
    )


if __name__ == "__main__":
    main()
