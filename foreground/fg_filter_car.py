import os
import cv2
import numpy as np
from utils.track_util import iou
from common import delete_static_pair_data, read_ann_file, get_tracks, save_track_to_csv, save_ann_to_csv

fg_filter_hyper_param_dict = {
    "iou_threshold": 0.93,
    "xor_threshold": 0.026,
    "xor_ultra_small_threshold": 0.015,
    "xor_growth_factor": np.exp(1),
    "check_interval": 8,
    "ultra_small_scale_targets_th": 0.0014,
    "video_area": 1920 * 1080
}
IOU_THRESHOLD = "iou_threshold"
XOR_THRESHOLD = "xor_threshold"
XOR_ULTRA_SMALL_THRESHOLD = "xor_ultra_small_threshold"
XOR_GROWTH_FACTOR = "xor_growth_factor"
ULTRA_SMALL_SCALE_TARGETS_TH = "ultra_small_scale_targets_th"
VIDEO_AREA = "video_area"
CHECK_INTERVAL = "check_interval"

TRACK_ID = "track_id"
BBOX = "bbox"
FG_INST = "fg_inst"
FG_STATE = "state"
FG_PATH = "fg_path"
LEAK_DETECTION_SUPPLEMENT_FLAG = "leak_detection_supplement_flag"
HOP_COUNT = "hop_count"
SAVE_ORIGIN = "save_origin"
BORDER_THRESHOLD = "border_threshold"
FRAME_H = "frame_h"
FRAME_W = "frame_w"
PAIR_STR = "pair_str"
KEYPOINTS_Y = "keypoints_y"
KEYPOINTS_X = "keypoints_x"
PAIR_NAME = "name"


PAIR_STR = "pair_str"
KEYPOINTS_Y = "keypoints_y"
KEYPOINTS_X = "keypoints_x"
PAIR_NAME = "name"


# 人体关键点检测
ANNOTATION_HUMAN_NAME = "annotation_human.csv"
# 车辆关键点检测
ANNOTATION_VEHICLE_NAME = "annotation_vehicle.csv"
# 人体bbox
PERSON_BOX_NAME = "person_box.txt"
# 人体目标快照，待重建快照对应
PERSON_PAIR_NAME = "person_pair.csv"
# 车辆bbox
VEHICLE_BOX_NAME = "vehicle_box.txt"
# 车辆目标快照，待重建快照对应
VEHICLE_PAIR_NAME = "vehicle_pair.csv"

def set_params(iou_threshold=None, xor_threshold=None, xor_ultra_small_threshold=None, xor_growth_factor=None, ultra_small_scale_targets_th=None):
    if not iou_threshold is None:
        fg_filter_hyper_param_dict[IOU_THRESHOLD] = iou_threshold
    if not xor_threshold is None:
        fg_filter_hyper_param_dict[XOR_THRESHOLD] = xor_threshold
    if not xor_ultra_small_threshold is None:
        fg_filter_hyper_param_dict[XOR_ULTRA_SMALL_THRESHOLD] = xor_ultra_small_threshold
    if not xor_growth_factor is None:
        fg_filter_hyper_param_dict[XOR_GROWTH_FACTOR] = xor_growth_factor
    if not ultra_small_scale_targets_th is None:
        fg_filter_hyper_param_dict[ULTRA_SMALL_SCALE_TARGETS_TH] = ultra_small_scale_targets_th
    
    print(fg_filter_hyper_param_dict)

def split_frame_num_and_bbox(frame_and_bbox_str: str):
    results = frame_and_bbox_str.split("/")
    bbox_str = results[1].split(".")[0]
    bbox_str_list = bbox_str.split("_")
    bbox_tuple = tuple(map(int, bbox_str_list))
    return results[0], bbox_tuple

def create_new_temp_dict(pair_from):
    temp_dict = {}
    temp_dict["from"] = pair_from
    temp_dict["to_list"] = []
    return temp_dict

def add_data_to_temp_dict(temp_dict, pair_from, to_str):
    to_list = temp_dict[pair_from]
    assert type(to_list) == list
    to_list.append(to_str)
    return temp_dict

def save_temp_data(pair_data_list: list, temp_dict: dict):
    pair_data_list.append(temp_dict)
    return pair_data_list

def set_video_area(area):
    fg_filter_hyper_param_dict[VIDEO_AREA] = area

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    x1 += 1
    y1 += 1
    x2 += 1
    y2 += 1
    return (y2 - y1) * (x2 - x1)

def set_bbox_state_and_more(main_object_list):
    for frame_bbox_list in main_object_list:
        for fg_bbox_dict in frame_bbox_list:
            fg_bbox_dict["state"] = 1
    return main_object_list

def get_file_path_by_filename(directory, filename):
    # 列出目录中的所有文件
    files_in_directory = os.listdir(directory)
    # 尝试找到匹配的文件名
    extension = None
    for file_in_directory in files_in_directory:
        if file_in_directory.startswith(filename):
            # 分割文件名和扩展名
            parts = file_in_directory.split('.')
            if len(parts) > 1:
                extension = parts[-1]
                break
            else:
                extension = None
    if extension is None:
        raise FileExistsError(os.path.join(directory, filename) + "not exists")
    return os.path.join(directory, filename + "." + extension), extension

def bbox_to_filename(bbox):
    return str(int(bbox[0]) + 1) + '_' + str(int(bbox[1]) + 1) + '_' + str(int(bbox[2]) + 1) + '_' + str(int(bbox[3]) + 1)

def get_bool_mask(img_nd):
    # 二值化
    img_nd = img_nd.astype(bool) 
    return img_nd

def calculate_amplified_threshold(bbox_iou):
    iou_th = fg_filter_hyper_param_dict[IOU_THRESHOLD]
    growth_factor = fg_filter_hyper_param_dict[XOR_GROWTH_FACTOR]

    bbox_iou = max(iou_th, min(1, bbox_iou))

    # 定义一个基础放大倍数，当bbox_iou = IOU_THRESHOLD时，放大倍数为1
    base_amplification_factor = 1

    normalized_value = (bbox_iou - iou_th) / (1 - iou_th)

    # 计算放大倍数，使用指数函数来增加t对放大倍数的影响
    amplification_factor = base_amplification_factor * (growth_factor ** (normalized_value))
    
    # 计算调整后的阈值
    amplified_threshold = fg_filter_hyper_param_dict[XOR_THRESHOLD] * amplification_factor
    return amplified_threshold

def get_xor_th(bbox):
    xor_th = fg_filter_hyper_param_dict[XOR_THRESHOLD]
    bbox_area = get_bbox_area(bbox)
    scale_coefficient = bbox_area / fg_filter_hyper_param_dict[VIDEO_AREA]
    if scale_coefficient < fg_filter_hyper_param_dict[ULTRA_SMALL_SCALE_TARGETS_TH]:
        xor_th = calculate_amplified_threshold(scale_coefficient)
    return xor_th

def get_xor_th_by_iou(c_n_iou):
    return calculate_amplified_threshold(c_n_iou)

def instance_is_not_change(c_bbox_file_path, n_bbox_file_path, c_n_iou):    
    c_bbox_nd = cv2.imread(c_bbox_file_path, cv2.IMREAD_GRAYSCALE)
    n_bbox_nd = cv2.imread(n_bbox_file_path, cv2.IMREAD_GRAYSCALE)

    size_tuple = (c_bbox_nd.shape[1], c_bbox_nd.shape[0])
    c_nd = c_bbox_nd
    if c_bbox_nd.shape == n_bbox_nd.shape:
        n_nd = n_bbox_nd
    else:
        n_nd = cv2.resize(n_bbox_nd, size_tuple)
    c_nd = get_bool_mask(c_nd)
    n_nd = get_bool_mask(n_nd)

    xor_nd = c_nd ^ n_nd
    xor_th = get_xor_th_by_iou(c_n_iou)
    # xor_th = fg_filter_hyper_param_dict[XOR_THRESHOLD]
    return (np.sum(xor_nd) / np.sum(c_nd)) < xor_th

def filter_cf(main_object_list):
    l = []
    for frame_bbox_list in main_object_list:
        for fg_bbox_dict in frame_bbox_list:
            if fg_bbox_dict["save_origin"] == 1 and fg_bbox_dict["state"] == 1:
                    l.append(fg_bbox_dict)


    for l_d_1_index in range(len(l)):
        l_d_1 = l[l_d_1_index]
        if l_d_1["save_origin"] == 0:
                continue
        for l_d_2_index in range(len(l)):
            if l_d_1_index == l_d_2_index:
                continue
            l_d_2 = l[l_d_2_index]
            l_d_1_bbox = l_d_1["bbox"]
            l_d_2_bbox = l_d_2["bbox"]
            if iou(l_d_1_bbox, l_d_2_bbox) > fg_filter_hyper_param_dict[IOU_THRESHOLD]:
                l_d_2["save_origin"] = 0
    return main_object_list

def check_not_fg_change(compress_path, start_data, next_data):
    _, start_bbox = split_frame_num_and_bbox(start_data[PAIR_STR])
    _, next_bbox = split_frame_num_and_bbox(next_data[PAIR_STR])
    bbox_iou = iou(bbox1=start_bbox, bbox2=next_bbox)
    if bbox_iou > fg_filter_hyper_param_dict[IOU_THRESHOLD]:
        return instance_is_not_change(os.path.join(compress_path, start_data[PAIR_STR]), 
                                      os.path.join(compress_path, next_data[PAIR_STR]), 
                                      bbox_iou)
    else:
        return False

def check_single_track(compress_path, from_str, data_list):
    start_index = 0
    n = len(data_list)
    k = fg_filter_hyper_param_dict[CHECK_INTERVAL]
    h = int(n / k) if n % k == 0 else int(n / k + 1)
    for i in range(1, h):
        g = i + 1
        next_index = int(g * k) if int(g * k) < n else (n - 1)
        start_data = data_list[start_index]
        next_data = data_list[next_index]
        if check_not_fg_change(compress_path, start_data, next_data):
            for j in range(start_index, next_index + 1):
                temp_data = data_list[j]
                if not from_str == temp_data[PAIR_STR]:
                    temp_data[FG_STATE] = 0
        start_index = next_index


def check_tracks(compress_path, track_dict):
    for key, value in track_dict.items():
        check_single_track(compress_path, key, value)
    return track_dict

def filter_key_vehicle(compress_path):
    # 读取关键点数据
    vehicle_annotation_pd = read_ann_file(os.path.join(compress_path, ANNOTATION_VEHICLE_NAME))
    
    vehicle_track_path = os.path.join(compress_path, VEHICLE_PAIR_NAME)
    # 读取追踪数据

    vehicle_track_dict = get_tracks(vehicle_track_path)

    if os.path.exists(vehicle_track_path):
        os.remove(vehicle_track_path)

    # print("-----------------------vehicle start--------------------------------------")
    vehicle_track_dict = check_tracks(compress_path, vehicle_track_dict)
    # print("-----------------------vehicle end--------------------------------------")


    # 遍历整个轨迹，如果就那一个state是1，就说明整个链都没变化，删掉from的快照
    vehicle_track_dict = delete_static_pair_data(compress_path, vehicle_track_dict)
    if len(vehicle_track_dict) > 0:
        save_track_to_csv(vehicle_track_path, vehicle_track_dict)

    # 遍历整个轨迹，拿出来state为1的那些关键点，给写进去，其余的就算是删了
    save_ann_to_csv(compress_path, ANNOTATION_VEHICLE_NAME, vehicle_annotation_pd, vehicle_track_dict)

# 用原始的图像做判断，但删除的是重建目录里的图像
def filter_key_vehicle_decomp(compress_path, decomp_compress_path):
    # 读取关键点数据
    vehicle_annotation_pd = read_ann_file(os.path.join(decomp_compress_path, ANNOTATION_VEHICLE_NAME))
    
    vehicle_track_path = os.path.join(decomp_compress_path, VEHICLE_PAIR_NAME)
    # 读取追踪数据

    vehicle_track_dict = get_tracks(vehicle_track_path)

    if os.path.exists(vehicle_track_path):
        os.remove(vehicle_track_path)

    # print("-----------------------vehicle start--------------------------------------")
    vehicle_track_dict = check_tracks(compress_path, vehicle_track_dict)
    # print("-----------------------vehicle end--------------------------------------")


    # 遍历整个轨迹，如果就那一个state是1，就说明整个链都没变化，删掉from的快照
    vehicle_track_dict = delete_static_pair_data(decomp_compress_path, vehicle_track_dict)
    if len(vehicle_track_dict) > 0:
        save_track_to_csv(vehicle_track_path, vehicle_track_dict)

    # 遍历整个轨迹，拿出来state为1的那些关键点，给写进去，其余的就算是删了
    save_ann_to_csv(compress_path, ANNOTATION_VEHICLE_NAME, vehicle_annotation_pd, vehicle_track_dict)




