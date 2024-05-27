

'''
    对于文件的压缩, 主要目的是进一步提升压缩性能。目前有三类文件, 一种是box文件, 用于后续track实例;
    一种是ann文件, 用于记录实例的关键点; 最后一种是重建映射表, 用于记录目标快照与待重建目标之间的映射关系, 简称为映射关系文件。
    另外还要注意的是, 为了重建视频帧时方便对应实例的位置, 实例bbox为被设置为快照的文件名。

    压缩流程：
        首先, 完成除了此步骤以外, 所有压缩流程的步骤(生成所有文件)。首先读取映射关系文件(原始字段存储逻辑是 帧号/bbox.后缀),
        为每一个from制作一个新的字典映射序列, 并将内部的bbox和from列的bbox做差值存储(有一定概率会是0)为字典内的bboxs列表。
        将from列的图片存到名称为s的文件夹中, 并重命名为字典列表的序号。
        记录to列的帧号为to_rc2的列表
        修改ann内的图像路径为f_index 或 t_findex_index, 方便后续重建时的对应
        将上述字典列表保存为json文件

        大致格式如下,最外层index0是人体, 1是车辆
        [
        [
            {
                f_bbox:"x1_y1_x2_y2"
                f_i:帧号
                f_ann: "关键点"
                t_bboxs:[x1_y1_x2_y2, x1_y1_x2_y2, x1_y1_x2_y2](差值)
                TODO 看一下[帧号] 和 "0_1_2_3_4"哪种存储更低
                t_rc2:[帧号]
                t_anns:[关键点]
            },
            {
                f_bbox:"x1_y1_x2_y2"
                f_i:帧号
                f_ann: "关键点"
                t_bboxs:[x1_y1_x2_y2, x1_y1_x2_y2, x1_y1_x2_y2](差值)
                TODO 看一下[帧号] 和 "0_1_2_3_4"哪种存储更低
                t_rc2:[帧号]
                t_anns:[关键点]
            },
            ……
        ],
        [……]
        ]
        TODO看一下json能不能被进一步压缩

'''

import pandas as pd

import os
import copy
import json
from shutil import rmtree
from shutil import move
from tqdm import tqdm

# 超参数
# 人体关键点检测
annotation_human_name = "annotation_human.csv"
# 车辆关键点检测
annotation_vehicle_name = "annotation_vehicle.csv"
# 人体bbox
person_box_name = "person_box.txt"
# 人体目标快照，待重建快照对应
person_pair_name = "person_pair.csv"
# 车辆bbox
vehicle_box_name = "vehicle_box.txt"
# 车辆目标快照，待重建快照对应
vehicle_pair_name = "vehicle_pair.csv"

comp_json_name = "datas.json"

snapshoot_dir_name = "s"

human_ext = ".png"

vehicle_ext = ".jpg"

human_class_name = "h"
vehicle_class_name = "v"

def split_frame_num_and_bbox(frame_and_bbox_str: str):
    results = frame_and_bbox_str.split("/")
    bbox_str = results[1].split(".")[0]
    return results[0], bbox_str

def comp_num_collection(num_collection):
    result_str = ""
    for i in range(len(num_collection)):
        result_str += str(num_collection[i])
        if (i + 1) != len(num_collection):
            result_str += "_"
    return result_str

def decomp_num_list(data_str: str):
    data_list = []
    data_str_list = data_str.split("_")
    for i in range(len(data_str_list)):
        data_list.append(int(data_str_list[i]))
    return data_list

def create_new_dict(pair_from, f_ann):
    temp_dict = {}
    frame_num, bbox_str = split_frame_num_and_bbox(pair_from)
    temp_dict["f_bbox"] = bbox_str
    temp_dict["f_i"] = int(frame_num)
    temp_dict["f_ann"] = f_ann
    temp_dict["t_bboxs"] = []
    temp_dict["t_rc2"] = []
    temp_dict["t_anns"] = []
    return temp_dict

def get_to_difference_value(from_bbox_str: str, to_bbox_str: str):
    f_x1, f_y1, f_x2, f_y2 = from_bbox_str.split("_")
    t_x1, t_y1, t_x2, t_y2 = to_bbox_str.split("_")
    d_x1 = int(t_x1) - int(f_x1)
    d_y1 = int(t_y1) - int(f_y1)
    d_x2 = int(t_x2) - int(f_x2)
    d_y2 = int(t_y2) - int(f_y2)
    return comp_num_collection([d_x1, d_y1, d_x2, d_y2])

def get_origin_bbox_str(from_bbox_str: str, to_d_bbox_str: str):
    f_x1, f_y1, f_x2, f_y2 = from_bbox_str.split("_")
    t_d_x1, t_d_y1, t_d_x2, t_d_y2 = to_d_bbox_str.split("_")
    t_x1 = int(t_d_x1) + int(f_x1)
    t_y1 = int(t_d_y1) + int(f_y1)
    t_x2 = int(t_d_x2) + int(f_x2)
    t_y2 = int(t_d_y2) + int(f_y2)
    return comp_num_collection([t_x1, t_y1, t_x2, t_y2])

def temp_save_data(result_list, temp_dict, compress_path, pair_from, class_name, snapshoot_index):
    result_list.append(copy.deepcopy(temp_dict))
    snapshoot_path = os.path.join(compress_path, copy.deepcopy(pair_from))
    new_snapshoot_dir_path = os.path.join(compress_path, snapshoot_dir_name, class_name)
    if not os.path.exists(new_snapshoot_dir_path):
        os.makedirs(new_snapshoot_dir_path)
    ext = os.path.splitext(snapshoot_path)[1]
    new_snapshoot_path = os.path.join(new_snapshoot_dir_path, str(snapshoot_index) + ext)
    move(snapshoot_path, new_snapshoot_path)
    # os.rename(snapshoot_path, os.path.join(compress_path, str(snapshoot_index) + ext))
    # snapshoot_path = os.path.join(compress_path, str(snapshoot_index) + ext)
    snapshoot_index += 1
    return result_list, snapshoot_index

def get_data_list(compress_path, pair_file, annotation_file, class_name="v"):
    result_list = []
    pair_from = None
    # f_bbox:"x1_y1_x2_y2"
    # f_i:帧号
    # f_ann: "关键点"
    # t_bboxs:[x1_y1_x2_y2, x1_y1_x2_y2, x1_y1_x2_y2](差值)
    # TODO 看一下[帧号] 和 "0_1_2_3_4"哪种存储更低
    # t_rc2:[帧号]
    # t_anns:[关键点]
    snapshoot_index = 0
    count = 0
    for _, p in tqdm(pair_file.iterrows()):
        if pair_from is None or pair_from != p["from"]:
            if not pair_from is None:
                result_list, snapshoot_index = temp_save_data(result_list, temp_dict, compress_path, pair_from, class_name, snapshoot_index)

            pair_from = p["from"]
            f_ann_row = annotation_file.loc[p["from"]]
            f_keypoints_y_str = comp_num_collection(json.loads(f_ann_row["keypoints_y"]))
            f_keypoints_x_str = comp_num_collection(json.loads(f_ann_row["keypoints_x"]))
            temp_dict = create_new_dict(pair_from, f_keypoints_y_str + "_" + f_keypoints_x_str)  

        to_bboxs = temp_dict["t_bboxs"]
        to_rc2s = temp_dict["t_rc2"]
        t_anns = temp_dict["t_anns"]
        to_frame_num, to_bbox_str = split_frame_num_and_bbox(p["to"])
        to_d_bbox_str = get_to_difference_value(temp_dict["f_bbox"], to_bbox_str)
        to_bboxs.append(to_d_bbox_str)
        to_rc2s.append(int(to_frame_num))
        to_ann_row = annotation_file.loc[p["to"]]
        t_keypoints_y_str = comp_num_collection(json.loads(to_ann_row["keypoints_y"]))
        t_keypoints_x_str = comp_num_collection(json.loads(to_ann_row["keypoints_x"]))
        t_anns.append(t_keypoints_y_str + "_" + t_keypoints_x_str)

        count += 1
        if count == len(pair_file.index):
            result_list, snapshoot_index = temp_save_data(result_list, temp_dict, compress_path, pair_from, class_name, snapshoot_index)


    return result_list

def delete_folders(path, exclude=None):  
    for root, dirs, _ in os.walk(path):
        for folder in dirs:
            try:  
                _ = int(folder)  
                rmtree(os.path.join(root, folder))
            except ValueError:  
                pass

def remove_fun(path):
    try:
        os.remove(path)
    except FileNotFoundError as e:
        pass

def get_comp_data_list(compress_path, csv_path, annotation_file, class_name):
    try:
        pair_file = pd.read_csv(csv_path) 
    except FileNotFoundError as e:
        return []
    return get_data_list(compress_path, pair_file, annotation_file, class_name)

def read_ann_file(csv_path):
    annotation_file = pd.read_csv(csv_path, sep=':')
    annotation_file = annotation_file.set_index("name")
    return annotation_file 

def comp_files(compress_path):
    # 读取标注文件
    human_annotation_file = read_ann_file(os.path.join(compress_path, annotation_human_name))
    vehicle_annotation_file = read_ann_file(os.path.join(compress_path, annotation_vehicle_name))
    # 获取压缩好的数据
    print("压缩行人数据")
    human_data_list = get_comp_data_list(compress_path, os.path.join(compress_path, person_pair_name), human_annotation_file, human_class_name)
    print("压缩车辆数据")
    vehicle_data_list = get_comp_data_list(compress_path, os.path.join(compress_path, vehicle_pair_name), vehicle_annotation_file, vehicle_class_name)
    # TODO 存一个bginfo，放在第三个位置
    # TODO 读背景的json
    # TODO 直接放第三个位置
    # TODO 修改一下重建部分的逻辑，依据split_list重建
    output_list = [human_data_list, vehicle_data_list]
    print("存json以及删除文件夹")
    # 存json
    with open(os.path.join(compress_path, comp_json_name), "w") as f:
        json.dump(output_list, f)
    # 删除原来的文件以及帧号文件夹
    remove_fun(os.path.join(compress_path, annotation_human_name))
    remove_fun(os.path.join(compress_path, annotation_vehicle_name))
    remove_fun(os.path.join(compress_path, person_pair_name))
    remove_fun(os.path.join(compress_path, vehicle_pair_name))
    remove_fun(os.path.join(compress_path, person_box_name))
    remove_fun(os.path.join(compress_path, vehicle_box_name))
    delete_folders(compress_path, exclude=snapshoot_dir_name)
    print("压缩完成")

def restore_snapshoot_location(compress_path, data_list, ext, class_name):
    snapshoot_dir_path = os.path.join(compress_path, snapshoot_dir_name, class_name)

    for i in range(len(data_list)):
        data_dict = data_list[i]
        snapshoot_path = os.path.join(snapshoot_dir_path, str(i) + ext)
        origin_name = data_dict["f_bbox"] + ext
        origin_s_dir = str(data_dict["f_i"])
        os.rename(snapshoot_path, os.path.join(snapshoot_dir_path, origin_name))
        origin_snapshoot_path = os.path.join(compress_path, origin_s_dir, origin_name)
        move(os.path.join(snapshoot_dir_path, origin_name), origin_snapshoot_path)

def write_ann(annotation_file, frame_i, bbox_str, ann_str, split_num, ext):
    origin_save_path = str(frame_i) + "/" + bbox_str + ext
    f_ann = ann_str
    f_ann_num_list = decomp_num_list(f_ann)
    f_ann_y = f_ann_num_list[:split_num]
    f_ann_x = f_ann_num_list[split_num:]
    annotation_file.write("%s:%s:%s" % (origin_save_path, str(f_ann_y), str(f_ann_x))+'\n')


def restore_ann_file(compress_path, data_list, ext, class_name):
    file_name = annotation_human_name if class_name == human_class_name else annotation_vehicle_name
    split_num = 18 if class_name == human_class_name else 20

    annotation_file = open(os.path.join(compress_path, file_name + ".txt"), 'w')
    annotation_file.write('name:keypoints_y:keypoints_x'+'\n')
    annotation_file.flush()
    # f_bbox:"x1_y1_x2_y2"
    # f_i:帧号
    # f_ann: "关键点"
    # t_bboxs:[x1_y1_x2_y2, x1_y1_x2_y2, x1_y1_x2_y2](差值)
    # TODO 看一下[帧号] 和 "0_1_2_3_4"哪种存储更低
    # t_rc2:[帧号]
    # t_anns:[关键点]
    for data_dict in data_list:
        # 先写from的关键点
        write_ann(annotation_file, data_dict["f_i"], data_dict["f_bbox"], data_dict["f_ann"], split_num, ext)
        f_bbox = data_dict["f_bbox"]
        f_save_path = str(data_dict["f_i"]) + "/" + f_bbox + ext
        
        # 再写入to的关键点
        t_bboxs = data_dict["t_bboxs"]
        t_rc2 = data_dict["t_rc2"]
        t_anns = data_dict["t_anns"]

        for j in range(len(t_anns)):
            frame_i = t_rc2[j]
            bbox_d_str = t_bboxs[j]
            bbox_str = get_origin_bbox_str(f_bbox, bbox_d_str)
            t_save_path = str(frame_i) + "/" + bbox_str + ext
            if f_save_path == t_save_path:
                continue        
            ann_str = t_anns[j]
            write_ann(annotation_file, frame_i, bbox_str, ann_str, split_num, ext)
    annotation_file.flush()
    annotation_file.close()
    os.rename(os.path.join(compress_path, file_name + ".txt"), os.path.join(compress_path, file_name))

def restore_pair_file(compress_path, data_list, ext, class_name):
    file_name = person_pair_name if class_name == human_class_name else vehicle_pair_name
    
    # f_bbox:"x1_y1_x2_y2"
    # f_i:帧号
    # f_ann: "关键点"
    # t_bboxs:[x1_y1_x2_y2, x1_y1_x2_y2, x1_y1_x2_y2](差值)
    # TODO 看一下[帧号] 和 "0_1_2_3_4"哪种存储更低
    # t_rc2:[帧号]
    # t_anns:[关键点]
    csv_list = []
    for data_dict in data_list:
        f_i = data_dict["f_i"]
        f_bbox = data_dict["f_bbox"]
        f_path = str(f_i) + "/" + f_bbox + ext

        t_bboxs = data_dict["t_bboxs"]
        t_rc2 = data_dict["t_rc2"]
        for j in range(len(t_rc2)):
            frame_i = t_rc2[j]
            bbox_d_str = t_bboxs[j]
            bbox_str = get_origin_bbox_str(f_bbox, bbox_d_str)
            t_path = str(frame_i) + "/" + bbox_str + ext
            csv_list.append([f_path, t_path])
    
    
    df =pd.DataFrame(csv_list, columns = ['from','to'])
    df.to_csv(os.path.join(compress_path, file_name),index=False)

def restore_datas(compress_path, data_list, ext, class_name):
    # 恢复快照位置
    restore_snapshoot_location(compress_path, data_list, ext, class_name)
    # 恢复ann文件
    restore_ann_file(compress_path, data_list, ext, class_name)
    # 恢复pair文件
    restore_pair_file(compress_path, data_list, ext, class_name)


def decomp_files(compress_path, num_frames):
    # 读json
    json_path = os.path.join(compress_path, comp_json_name)
    comp_data_list = None
    with open(json_path, "r") as f:
        comp_data_list = json.load(f)

    assert not comp_data_list is None

    # 创建帧号文件夹
    for i in range(num_frames):
        frame_dir_path = os.path.join(compress_path, str(i + 1))
        if not os.path.exists(frame_dir_path):
            os.mkdir(frame_dir_path)

    human_data_list = comp_data_list[0]
    vehicle_data_list = comp_data_list[1]
    # 恢复数据
    if len(human_data_list) > 0:
        restore_datas(compress_path, human_data_list, human_ext, human_class_name)
    if len(vehicle_data_list) > 0:
        restore_datas(compress_path, vehicle_data_list, vehicle_ext, vehicle_class_name)    

if __name__ == "__main__":
    # compress_path = "videos/bg_test/long/video_crop57.rmp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/new_bpp/video_session3_right_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/video_comp/test_psnr/bpp/VID_20231030_165519_one_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/bistu/recon/test/VID_20231031_082631_one_2k.mp4_comp"
    # num_frames = 2000
    # # TODO 找一下json的压缩方法
    # comp_files(compress_path)
    # decomp_files(compress_path, num_frames)

    # compress_path = "D:/PythonProjects/datas/video_datas/bistu/recon/bistu_yes/VID_20231031_070225_one_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/bistu/recon/bistu_yes/VID_20231031_070225_three_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/bistu/recon/bistu_yes/VID_20231031_165213_one_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/bistu/recon/bistu_yes/VID_20231031_165213_two_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/brno2016/recon/video_session3_right_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/yousun/file_comp/VID_20231031_165213_one_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/yousun/file_comp/VID_20231031_165213_two_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/yousun/file_comp/VID_20231031_165523_one_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/yousun/file_comp/video_session3_right_2k.mp4_comp"
    # compress_path = "D:/PythonProjects/datas/video_datas/yousun/file_comp/video_session4_right_2k.mp4_comp"
    compress_path = r"D:\PythonProjects\datas\video_session3_right_2k.mp4_comp"

    comp_files(compress_path)
    # comp_files(bistu_v_2)
    # comp_files(bistu_v_3)
    # comp_files(bistu_v_4)
    # comp_files(bistu_v_5)
    










