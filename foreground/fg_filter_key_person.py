import os
import copy
import json
import numpy as np
from utils.track_util import iou
from common import delete_static_pair_data, read_ann_file, get_tracks, save_track_to_csv, save_ann_to_csv

fg_filter_hyper_param_dict = {
    "iou_threshold": 0.93,
    "xor_threshold": 0.03,
    "xor_ultra_small_threshold": 0.015,
    "xor_growth_factor": np.exp(1),
    "ultra_small_scale_targets_th": 0.0014,
    "border_threshold": 20,
    "video_area": 1920 * 1080,
    "check_interval": 8,
    "key_std_threshold": 0.5,
    "frame_h": 1080,
    "frame_w": 1920
}
IOU_THRESHOLD = "iou_threshold"
XOR_THRESHOLD = "xor_threshold"
XOR_ULTRA_SMALL_THRESHOLD = "xor_ultra_small_threshold"
XOR_GROWTH_FACTOR = "xor_growth_factor"
ULTRA_SMALL_SCALE_TARGETS_TH = "ultra_small_scale_targets_th"
VIDEO_AREA = "video_area"
CHECK_INTERVAL = "check_interval"
KEY_STD_THRESHOLD = "key_std_threshold"

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

test_big_count = [0]
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

def set_video_area(area):
    fg_filter_hyper_param_dict[VIDEO_AREA] = area

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    x1 += 1
    y1 += 1
    x2 += 1
    y2 += 1
    return (y2 - y1) * (x2 - x1)

def split_frame_num_and_bbox(frame_and_bbox_str: str):
    results = frame_and_bbox_str.split("/")
    bbox_str = results[1].split(".")[0]
    bbox_str_list = bbox_str.split("_")
    bbox_tuple = tuple(map(int, bbox_str_list))
    return results[0], bbox_tuple

def cal_key_std(key1, key2):
    if not type(key1) == np.ndarray:
        key1 = np.array(key1)
    if not type(key2) == np.ndarray:
        key2 = np.array(key2)
    # 计算两个数组对应位置数据的差的平方
    differences_squared = (key1 - key2) ** 2
    # 计算平均差的平方
    mean_difference_squared = np.mean(differences_squared)
    # 计算标准差
    std_dev = np.sqrt(mean_difference_squared)
    return std_dev

def key_point_aligning(keyp1, keyp2):
    k1 = copy.deepcopy(keyp1)
    k2 = copy.deepcopy(keyp2)
    for i in range(len(k1)):
        k_t1 = k1[i]
        k_t2 = k2[i]
        if k_t1 == -1 or k_t2 == -1:
            k1[i] = k2[i]
    return k1, k2

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
    amplified_threshold = fg_filter_hyper_param_dict[KEY_STD_THRESHOLD] * amplification_factor
    return amplified_threshold

def get_key_mean_th(fg_iou):
    return calculate_amplified_threshold(fg_iou)

def check_not_fg_change(start_data, next_data, annotation_pd):
    _, start_bbox = split_frame_num_and_bbox(start_data[PAIR_STR])
    _, next_bbox = split_frame_num_and_bbox(next_data[PAIR_STR])
    bbox_iou = iou(bbox1=start_bbox, bbox2=next_bbox)
    change_flag = False
    if bbox_iou > fg_filter_hyper_param_dict[IOU_THRESHOLD]:
        start_key_str = annotation_pd.loc[start_data[PAIR_STR]]
        next_key_str = annotation_pd.loc[next_data[PAIR_STR]]

        start_key_y = json.loads(start_key_str[KEYPOINTS_Y])
        start_key_x = json.loads(start_key_str[KEYPOINTS_X])
        next_key_y = json.loads(next_key_str[KEYPOINTS_Y])
        next_key_x = json.loads(next_key_str[KEYPOINTS_X])
        start_key_y, next_key_y = key_point_aligning(start_key_y, next_key_y)
        start_key_x, next_key_x = key_point_aligning(start_key_x, next_key_x)
        y_std = cal_key_std(start_key_y, next_key_y)
        x_std = cal_key_std(start_key_x, next_key_x)
        mean_std = np.mean([y_std, x_std])
        print(mean_std, bbox_iou)
        # if mean_std > 4.065:
        #     test_big_count[0] += 1
        key_mean_th = get_key_mean_th(bbox_iou)
        if mean_std < key_mean_th:
            change_flag = True
    
    return change_flag

def check_single_track(from_str, data_list, annotation_pd):
    start_index = 0
    n = len(data_list)
    k = fg_filter_hyper_param_dict[CHECK_INTERVAL]
    h = int(n / k) if n % k == 0 else int(n / k + 1)
    for i in range(1, h):
        next_index = int(i * k) if int(i * k) < n else (n - 1)
        start_data = data_list[start_index]
        next_data = data_list[next_index]
        if check_not_fg_change(start_data, next_data, annotation_pd):
            for j in range(start_index, next_index + 1):
                temp_data = data_list[j]
                if not from_str == temp_data[PAIR_STR]:
                    temp_data[FG_STATE] = 0
        start_index = next_index


def check_tracks(track_dict, annotation_pd):
    for key, value in track_dict.items():
        check_single_track(key, value, annotation_pd)
    return track_dict

def filter_key_person(compress_path):
    # 读取关键点数据
    person_annotation_pd = read_ann_file(os.path.join(compress_path, ANNOTATION_HUMAN_NAME))
    
    person_track_path = os.path.join(compress_path, PERSON_PAIR_NAME)
    # 读取追踪数据

    person_track_dict = get_tracks(person_track_path)

    if os.path.exists(person_track_path):
        os.remove(person_track_path)

    # print("-----------------------vehicle start--------------------------------------")
    person_track_dict = check_tracks(person_track_dict, person_annotation_pd)
    # print("-----------------------vehicle end--------------------------------------")


    # 遍历整个轨迹，如果就那一个state是1，就说明整个链都没变化，删掉from的快照
    person_track_dict = delete_static_pair_data(compress_path, person_track_dict)
    if len(person_track_dict) > 0:
        save_track_to_csv(person_track_path, person_track_dict)

    # 遍历整个轨迹，拿出来state为1的那些关键点，给写进去，其余的就算是删了
    save_ann_to_csv(compress_path, ANNOTATION_HUMAN_NAME, person_annotation_pd, person_track_dict)

# 用原始的图像做判断，但删除的是重建目录里的图像
def filter_key_person_decomp(compress_path, decomp_compress_path):
    # 读取关键点数据
    person_annotation_pd = read_ann_file(os.path.join(decomp_compress_path, ANNOTATION_HUMAN_NAME))
    
    person_track_path = os.path.join(decomp_compress_path, PERSON_PAIR_NAME)
    # 读取追踪数据

    person_track_dict = get_tracks(person_track_path)

    if os.path.exists(person_track_path):
        os.remove(person_track_path)

    # print("-----------------------vehicle start--------------------------------------")
    person_track_dict = check_tracks(person_track_dict, person_annotation_pd)
    # print("-----------------------vehicle end--------------------------------------")


    # 遍历整个轨迹，如果就那一个state是1，就说明整个链都没变化，删掉from的快照
    person_track_dict = delete_static_pair_data(decomp_compress_path, person_track_dict)
    if len(person_track_dict) > 0:
        save_track_to_csv(person_track_path, person_track_dict)

    # 遍历整个轨迹，拿出来state为1的那些关键点，给写进去，其余的就算是删了
    save_ann_to_csv(compress_path, ANNOTATION_HUMAN_NAME, person_annotation_pd, person_track_dict)




if __name__ == "__main__":
    compress_path = "video_session3_right_2k.mp4_comp"




