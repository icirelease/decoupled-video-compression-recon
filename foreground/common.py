import os
import warnings
import numpy as np
from tqdm import tqdm
import mmcv
from mmcv.runner import load_checkpoint
from manager.HRNet_inference.point_detect import cordinates_from_image_batch as compute_cord
from manager.pct.models import build_posenet
from mmpose.datasets import DatasetInfo
from mmpose.apis import inference_top_down_pose_model

import pandas as pd


PAIR_STR = "pair_str"
KEYPOINTS_Y = "keypoints_y"
KEYPOINTS_X = "keypoints_x"
PAIR_NAME = "name"

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

def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def compute_cord_human(image_batch, human_pose_model):
    dataset = human_pose_model.cfg.data['test']['type']
    dataset_info = human_pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    output = []
    for i in range(len(image_batch)):
        # image_ndarry, transform_param = resize_img_keep_ratio(image_batch[i],(288,384))
        image_ndarry = image_batch[i]
        image_ndarry = np.array(image_ndarry)
        h = image_ndarry.shape[0]
        w = image_ndarry.shape[1]
        person_results = None
        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
        # debug_condition(img_path, "./videos/video3.mp4_comp/3/230_658_311_820.png", "筛选完框", person_result)
        pose_results, returned_outputs = inference_top_down_pose_model(
            human_pose_model,
            image_ndarry,
            person_results,
            bbox_thr=None,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        pose_keypoints = pose_results[0]["keypoints"]

        # deprocess_point(pose_keypoints, transform_param)

        single_joint = []
        for j in range(len(pose_keypoints)):
            score = pose_keypoints[j][2]
            if score > 0.25 and (int(pose_keypoints[j][0] <= w)) and (int(pose_keypoints[j][1]) <= h):
                single_joint.append([int(pose_keypoints[j][1]),int(pose_keypoints[j][0])])
            else:
                single_joint.append([-1,-1])
        output.append(single_joint)

    return output
    

def compute_instance_cord_batch(image_dict,instance_cord_file,instance_pose_model, target="vehicle"):
    output_img_dir_list = []
    image_batch = []
    for k,v in image_dict.items():
        output_img_dir_list.append(k)
        image_batch.append(v)

    if target == "vehicle":
        pose_cords = compute_cord(image_batch,instance_pose_model)
    elif target == "human":
        pose_cords = compute_cord_human(image_batch,instance_pose_model)
    else:
        raise RuntimeError("目前只支持人和车")

    pose_cords = np.array(pose_cords)
    for i,output_img_dir in enumerate(output_img_dir_list):
        save_dir = ''
        for strs in output_img_dir.split('/')[1:]:
            save_dir = os.path.join(save_dir, strs)
        instance_cord_file.write("%s:%s:%s" % (save_dir, str(list(pose_cords[i][:, 0])), str(list(pose_cords[i][:, 1])))+'\n')
    instance_cord_file.flush()

def ignore_instance_with_condition(ignore, classes, scores, masks, boxes, instance_num):

    ignore.tolist()
    if len(ignore):
        classes = np.delete(classes, ignore, 0)
        scores = np.delete(scores, ignore, 0)
        masks = np.delete(masks, ignore, 0)
        boxes = np.delete(boxes, ignore, 0)
        instance_num = instance_num - len(ignore)
    return classes, scores, masks, boxes, instance_num

def delete_static_pair_data(compress_path, track_dict):
    wii_delete_key_list = []
    for key, value in track_dict.items():
        d_f = True
        state_num = 0
        for t_data in value:
            state = t_data[FG_STATE]
            if state == 1:
                state_num += 1
            else:
                to_img_path = os.path.join(compress_path, t_data[PAIR_STR])
                if os.path.exists(to_img_path):
                    os.remove(to_img_path)
            if state_num > 1:
               d_f = False
               break
        if d_f:
            wii_delete_key_list.append(key)

    for wii_delete_key in wii_delete_key_list:
        # 删除快照
        img_path = os.path.join(compress_path, wii_delete_key)
        print(img_path)
        if os.path.exists(img_path):
            os.remove(img_path)
        # 删除追踪序列
        del track_dict[wii_delete_key]
    
    return track_dict

def read_ann_file(csv_path):
    annotation_file = pd.read_csv(csv_path, sep=':')
    annotation_file = annotation_file.set_index("name")
    return annotation_file 

def get_tracks(pair_csv_path):
    pair_file = pd.read_csv(pair_csv_path)
    pair_from = None
    temp_to_list = None
    track_dict = {}
    count = 0
    for _, p in tqdm(pair_file.iterrows()):
        if pair_from is None or pair_from != p["from"]:
            if not pair_from is None:
                track_dict[pair_from] = temp_to_list
            pair_from = p["from"]
            temp_to_list = []

        assert temp_to_list is not None
        temp_dict = {}
        temp_dict[PAIR_STR] = p["to"]
        temp_dict[FG_STATE] = 1
        temp_to_list.append(temp_dict)

        count += 1
        if count == len(pair_file.index):
            track_dict[pair_from] = temp_to_list
    
    return track_dict

def save_track_to_csv(pair_csv_path, track_dict):
    csv_list = []
    for key, value in track_dict.items():
        for to_data_dict in value:
            if to_data_dict[FG_STATE] == 1:
                csv_list.append([key, to_data_dict[PAIR_STR]])
    
    df =pd.DataFrame(csv_list, columns = ['from','to'])
    df.to_csv(pair_csv_path, index=False)

def write_ann(annotation_file, name, keypoints_y, keypoints_x):
    annotation_file.write("%s:%s:%s" % (name, keypoints_y, keypoints_x)+'\n')
    annotation_file.flush()

def save_ann_to_csv(compress_path, file_name, annotation_pd, track_dict):
    csv_path = os.path.join(compress_path, file_name)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    annotation_file = open(os.path.join(compress_path, file_name + ".txt"), 'w')
    annotation_file.write('name:keypoints_y:keypoints_x'+'\n')
    annotation_file.flush()
    for key, value in track_dict.items():
        from_data_pd = annotation_pd.loc[key]
        from_name = key
        from_keypoints_y = from_data_pd[KEYPOINTS_Y]
        from_keypoints_x = from_data_pd[KEYPOINTS_X]
        write_ann(annotation_file, from_name, from_keypoints_y, from_keypoints_x)
        for to_data_dict in value:
            to_state = to_data_dict[FG_STATE]
            to_name_pair = to_data_dict[PAIR_STR]
            if key == to_name_pair:
                continue
            if to_state == 1:
                to_data_pd = annotation_pd.loc[to_name_pair]
                to_name = to_name_pair
                to_keypoints_y = to_data_pd[KEYPOINTS_Y]
                to_keypoints_x = to_data_pd[KEYPOINTS_X]
                write_ann(annotation_file, to_name, to_keypoints_y, to_keypoints_x)

    annotation_file.flush()
    annotation_file.close()
    os.rename(os.path.join(compress_path, file_name + ".txt"), os.path.join(compress_path, file_name))

def del_imgs(vid_path,pairs):
    pairs = pd.read_csv(pairs,sep=',',encoding='utf8')
    for _,row in pairs.iterrows():
        from_obj = row[0]
        to_obj = row[1]
        if not from_obj == to_obj:
            os.remove(os.path.join(vid_path,to_obj))