import os
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from utils.transformation import *
from functools import partial
from multiprocessing import Pool
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from tqdm import tqdm

def prepare_training_data_for_scene(level5data, first_sample_token, output_folder, bev_shape, voxel_size, z_offset, box_scale, classes):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    sample_token = first_sample_token

    while sample_token:

        sample = level5data.get("sample", sample_token)

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                           inverse=False)

        try:
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue

        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        boxes = level5data.get_boxes(sample_lidar_token)

        target = np.zeros_like(bev)

        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        bev_im = np.round(bev * 255).astype(np.uint8)
        target_im = target[:, :, 0]  # take one channel only

        cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)

        sample_token = sample["next"]


def generate_training_data(level5data, classes, train_df, train_data_folder, validation_df, validation_data_folder, NUM_WORKERS, box_scale, voxel_size, z_offset, bev_shape):

    for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
        print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
        first_samples = df.first_sample_token.values

        os.makedirs(data_folder, exist_ok=True)

        process_func = partial(prepare_training_data_for_scene, level5data=level5data,
                               output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset,
                               box_scale=box_scale, classes=classes)

        pool = Pool(NUM_WORKERS)
        for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
            pass
        pool.close()
        del pool



