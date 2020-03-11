# some methods for loading data were taken from project https://github.com/apratimbhattacharyya18/onboard_long_term_prediction
import cv2
import h5py
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_sequence_image_path(first_path, last_path, sequence_length, base_path):
    _, ff = os.path.split(first_path)
    _, lf = os.path.split(last_path)

    params = ff.split('_')

    sequence_name = params[0]
    sequence_num = params[1]
    start_frame_num = int(params[2])

    # params = lf.split('_')
    # end_frame_num = int(params[2])

    image_path_list = []
    for frame_num in range(start_frame_num, start_frame_num + sequence_length):
        res = str('_').join([sequence_name, sequence_num, format(frame_num, '06d'), 'leftImg8bit.png'])
        res = os.path.join(base_path, sequence_name, res)
        image_path_list.append(res)

    return image_path_list


def slice_tracks(bboxes, length):
    return [bboxes[i:i + length] for i in range(len(bboxes) - length + 1)]


def get_diff_array(arr):
    arr = arr[:, 0:4]
    arr1 = arr[1:]
    arr2 = arr[0]
    return np.subtract(arr1, arr2)


def get_diff_array_v2(arr):
    arr = arr[:, 0:4]
    return np.diff(arr, axis=0)


def get_features_and_labels(arr, in_frames):
    x = arr[0:in_frames - 1]
    y = arr[in_frames - 1:]
    x = np.reshape(x, (in_frames - 1, 4))
    return x, y


def get_features_and_labels_odometry(arr, in_frames):
    x = arr[0:in_frames - 1]
    y = arr[in_frames - 1:]
    return x, y


#accumulate move from t+1 to tn this is used when v2 diff array is used
def accumulate_move(data_x, data_y):
    accumulated_array = []
    for i in range(data_y.shape[0]):
        x_value = np.sum(data_x[i], axis=0)

        sample = []
        for tn in range(1, data_y.shape[1] + 1):
            sample.append(np.add(x_value, np.sum(data_y[i, :tn], axis=0)))

        accumulated_array.append(sample)

    return np.asarray(accumulated_array, dtype=np.float32)


def bbox_inverse_transform(bboxes):
    inverse_bbs = []
    for bbox in bboxes:
        inverse_bb = [
            2048 - bbox[0],
            bbox[1],
            2048 - bbox[2],
            bbox[3],
            bbox[4]
        ]
        inverse_bbs.append(inverse_bb)

    return inverse_bbs


def get_odometry_json(first_path, last_path, sequence_length, base_path):
    _, ff = os.path.split(first_path)
    _, lf = os.path.split(last_path)

    params = ff.split('_')

    sequence_name = params[0]
    sequence_num = params[1]
    start_frame_num = int(params[2])

    params = lf.split('_')
    # end_frame_num = int(params[2])

    json_list = []
    for frame_num in range(start_frame_num, start_frame_num + sequence_length):
        res = str('_').join([sequence_name, sequence_num, format(frame_num, '06d'), 'vehicle.json'])
        res = os.path.join(base_path, sequence_name, res)
        json_list.append(res)

    return json_list


def get_odometry(file_path, base_path, in_frames, out_frames, sequence_name=None, scale_coef_steering=10):
    source_f = h5py.File(file_path, 'r')

    seq_len = in_frames + out_frames
    odometry_data_x, odometry_data_y = [], []

    for track_key in source_f:

        curr_track = json.loads(source_f[track_key].value.decode())
        first_frame = curr_track['firstFrame']
        last_frame = curr_track['lastFrame']

        bboxes = curr_track['bboxes']

        _, ff = os.path.split(first_frame)
        params = ff.split('_')
        current_sequence_name = params[0]

        if len(bboxes) >= in_frames + out_frames and (sequence_name == current_sequence_name or sequence_name is None):
            odometry = []
            odometry_json_list = get_odometry_json(first_frame, last_frame, len(bboxes), base_path)
            for json_path in odometry_json_list:
                json_file = open(json_path, mode='r')
                data = json.load(json_file)
                yawRate = data['yawRate']
                speed = data['speed']
                odometry.append([speed, yawRate])
                json_file.close()

            odometry_slices = slice_tracks(odometry, seq_len)
            for odometry_slice in odometry_slices:
                odometry_slice = odometry_slice[1:]
                x, y = get_features_and_labels_odometry(odometry_slice, in_frames)
                odometry_data_x.append(x)
                odometry_data_y.append(y)

    odometry_data_x = np.array(odometry_data_x, dtype=np.float32)
    odometry_data_y = np.array(odometry_data_y, dtype=np.float32)

    odometry_data_x[:, :, 1] = odometry_data_x[:, :, 1] * scale_coef_steering
    odometry_data_y[:, :, 1] = odometry_data_y[:, :, 1] * scale_coef_steering

    return odometry_data_x, odometry_data_y


def get_data_set(in_frames, out_frames, file_path, diff_fn=get_diff_array_v2, use_inverse_bbs=False):
    source_f = h5py.File(file_path, 'r')

    seq_len = in_frames + out_frames
    data_x, data_y = [], []

    for track_key in source_f:

        curr_track = json.loads(source_f[track_key].value.decode())
        bboxes = curr_track['bboxes']

        if use_inverse_bbs:
            bboxes = bbox_inverse_transform(bboxes)

        if len(bboxes) >= in_frames + out_frames:
            bbox_slices = slice_tracks(bboxes, seq_len)
            for bbox_slice in bbox_slices:
                diff_array = diff_fn(np.asarray(bbox_slice))
                x, y = get_features_and_labels(diff_array, in_frames)
                data_x.append(x)
                data_y.append(y)

    return np.asarray(data_x, dtype=np.float32), np.asarray(data_y, dtype=np.float32)


def get_whole_data_set(in_frames, out_frames, tracks_base_path, odometry_base_path, img_sequences_base_path,
                       sequence_names=None, diff_fn=get_diff_array, load_img_paths=True,
                       max_num_of_sequences=np.iinfo(np.int).max):
    source_f = h5py.File(tracks_base_path, 'r')
    seq_len = in_frames + out_frames

    data_x, data_y = [], []
    initial_bboxes = []
    image_sequence_paths = []
    odometry_data_x, odometry_data_y = [], []
    initial_odometry = []

    seq_counter = 0
    for track_key in source_f:
        if seq_counter >= max_num_of_sequences:
            break
        curr_track = json.loads(source_f[track_key].value.decode())
        bboxes = curr_track['bboxes']
        first_frame = curr_track['firstFrame']
        last_frame = curr_track['lastFrame']

        _, ff = os.path.split(first_frame)
        params = ff.split('_')
        current_sequence_name = params[0]

        # if only certain sequences are needed
        if sequence_names is not None:
            if current_sequence_name not in sequence_names:
                continue

        if len(bboxes) >= in_frames + out_frames:

            #img paths
            image_paths = get_sequence_image_path(first_frame, last_frame, len(bboxes), img_sequences_base_path)
            image_slices = slice_tracks(image_paths, seq_len)

            for image_slice in image_slices:
                image_sequence_paths.append(image_slice)

            #bounding boxes
            bbox_slices = slice_tracks(bboxes, seq_len)
            for bbox_slice in bbox_slices:
                initial_bb = bbox_slice[0]
                diff_array = diff_fn(np.array(bbox_slice))
                (x, y) = get_features_and_labels(diff_array, in_frames)
                data_x.append(x)
                data_y.append(y)
                initial_bboxes.append(initial_bb)

            #odometry
            odometry = []
            odometry_json_list = get_odometry_json(first_frame, last_frame, len(bboxes), odometry_base_path)
            for json_path in odometry_json_list:
                json_file = open(json_path, mode='r')
                data = json.load(json_file)
                yaw_rate = data['yawRate']
                speed = data['speed']
                odometry.append([speed, yaw_rate])
                json_file.close()

            odometry_slices = slice_tracks(odometry, seq_len)
            for odometry_slice in odometry_slices:
                initial_odo = odometry_slice[0]
                odometry_slice = odometry_slice[1:]
                x, y = get_features_and_labels_odometry(odometry_slice, in_frames)
                odometry_data_x.append(x)
                odometry_data_y.append(y)
                initial_odometry.append(initial_odo)

            seq_counter += 1


    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    initial_bboxes = np.asarray(initial_bboxes, dtype=np.int)
    initial_bboxes = initial_bboxes[:, :4]
    odometry_data_x = np.asarray(odometry_data_x)
    odometry_data_y = np.asarray(odometry_data_y)
    initial_odometry = np.asarray(initial_odometry)

    return data_x, data_y, initial_bboxes, image_sequence_paths, odometry_data_x, odometry_data_y, initial_odometry


def clean_data(data_x, data_y, min_x, max_x, min_y, max_y):
    cleaned_data_x = []
    cleaned_data_y = []
    assert data_x.shape[0] == data_y.shape[0], '[-] Input and output are not same size'
    for i in range(len(data_x)):
        fxx = data_x[i, :, [0, 2]].flatten()
        fxy = data_x[i, :, [1, 3]].flatten()
        fyx = data_y[i, :, [0, 2]].flatten()
        fyy = data_y[i, :, [1, 3]].flatten()

        fx = np.concatenate((fxx,fyx))
        fy = np.concatenate((fxy,fyy))

        if np.min(fx) > min_x and np.max(fx) < max_x and np.min(fy) > min_y and np.max(fy) < max_y:
            cleaned_data_x.append(data_x[i])
            cleaned_data_y.append(data_y[i])

    return np.asarray(cleaned_data_x), np.asarray(cleaned_data_y)


def permutation(data_x, data_y):
    assert data_y.shape[0] == data_x.shape[0], 'Different num of samples in data_x and data_y'
    perm = np.random.permutation(data_x.shape[0])
    return data_x[perm], data_y[perm]


def standardization(x, mean, std):
    return (x - mean) / std


def inverse_standardization(x, mean, std):
    return (x * std) + mean


def save_pickle(path, object):
    with open(path, 'wb') as handle:
        pickle.dump(object, handle)


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)