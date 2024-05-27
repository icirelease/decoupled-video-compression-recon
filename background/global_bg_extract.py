import os
import sys
import copy
import math
import json
from tqdm import tqdm
import cv2
import numpy as np

from utils.track_util import load_mot, iou

from res_bg_extract import set_hop_count_for_bbox
from res_bg_extract import check_leak_detection_and_supplement_bbox

# 超参数
FG_DETECTOR_ALL_NAME = "all"
FG_DETECTOR_BBOX_NAME = "bbox"
FG_DETECTOR_BGMM_NAME = "bgmm"

video_bg_extract_v3_hyper_param_dict = {
    "fg_detector_switch": "all",
    "fg_detector_switch_collection": [FG_DETECTOR_ALL_NAME, FG_DETECTOR_BBOX_NAME, FG_DETECTOR_BGMM_NAME]
}



FG_DETECTOR_SWITCH = "fg_detector_switch"
FG_DETECTOR_SWITCH_COLLECTION = "fg_detector_switch_collection"

# 高斯建模的间隔帧数
interval_frame = 4

# 背景提取区间最小帧数间隔
min_frame_num_for_split = 100

# 背景关键帧切分帧数（每frame_per_section 提取一个背景关键帧。若值为-1，则不按此规则划分区间）
frame_per_section = -1

# 背景突变的阈值
mutation_threshold = 0.95

# 最长暂存bbox区域帧数据
max_temp_locality_threshold = 530

# 区域内前景最大面积
max_area = 1500

# 切片数量(建议是可以开根号的数值，且是分辨率可以整除的值)
crop_num = 64

# 行人检测框的文件名
person_box_file_name = "person_box.txt"

# 车辆检测框的文件名
vehicle_box_file_name = "vehicle_box.txt"

def add_section(split_list, num_frames):
    section_index_list = []
    for index in range(frame_per_section, num_frames, frame_per_section):
        if num_frames - index >= frame_per_section: 
            section_index_list.append(index)
    
    result_list = None
    if len(split_list) > 0:
        # 合并两个列表  
        merged_list = section_index_list + split_list
        # 对合并后的列表进行排序  
        sorted_list = sorted(merged_list)
        sorted_list = check_section(sorted_list)
        result_list = sorted_list
    else:
        result_list = section_index_list
    
    return result_list

def check_section(split_list):
    start_i = 0
    result_split_list = []
    for split_index in split_list:
        if split_index - start_i >= min_frame_num_for_split:
            result_split_list.append(split_index)
        start_i = split_index
    return result_split_list


# 均值法提取背景输入需要保持(T, H, W, C)
def mean_extract(nd_array):
    return np.mean(nd_array, axis=0)


def str_to_raw(s):
    raw_map = {8:r'\b', 7:r'\a', 12:r'\f', 10:r'\n', 13:r'\r', 9:r'\t', 11:r'\v'}
    return r''.join(i if ord(i) > 32 else raw_map.get(ord(i), i) for i in s)

# 不够的帧数要补空列表
def supplement_zero_frame_data(object_detections_list:list, num_frames:int):
    object_detections_len = len(object_detections_list)
    if object_detections_len < num_frames:
        supplement_len = num_frames - object_detections_len
        for i in range(supplement_len):
            object_detections_list.append([])
    
    return object_detections_list

# 通过目标框的文件，获取每一帧的目标box数据，不够的帧要补空list（比如视频长300帧，只有100帧有目标，就补200个空list）
def get_object_detections_list(box_file_path, num_frames:int):
    object_detections_list = []
    if os.path.getsize(box_file_path)!= 0:        
        with open(box_file_path, 'r') as infile:
            data1 = infile.readlines()
        if len(data1) > 0:
            object_detections_list = load_mot(box_file_path, nms_overlap_thresh=None, with_classes=False)

    return supplement_zero_frame_data(object_detections_list, num_frames)

# 高斯全局建模获取split_list 输出一个帧号的列表
def get_split_list(video_path):
    # 帧号列表
    split_list = []

    load_video_path = str_to_raw(video_path)
    cap = cv2.VideoCapture(load_video_path) #参数为0是打开摄像头，文件名是打开视频
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgbg = cv2.createBackgroundSubtractorMOG2()#混合高斯背景建模算法
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 全局面积
    global_area = float(width * height)

    for i in tqdm(range(num_frames)):
        # 读取当前帧
        _, frame = cap.read()
        # 高斯滤波(7,7)为卷积核大小
        # gauss_frame=cv2.GaussianBlur(frame,(7,7),0)
        gauss_frame = frame
        #对图像帧进行高斯背景建模
        fgmask = fgbg.apply(gauss_frame)
        # 间隔多少帧取一个
        if i % interval_frame != 0:
           continue
        
        # 形态学去噪
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # 开运算去噪 
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  
        # 寻找前景
        contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        temp_area = 0
        for cont in contours:
            # 计算轮廓面积
            area = cv2.contourArea(cont) 
            # 过滤面积小于10的形状 
            if area < 300:  
                continue
            temp_area += area
            
        # 如果当前帧面积的比值大于规定的阈值，则判定为发生了突变，添加帧号到split_list
        area_ratio = temp_area / global_area
        if area_ratio > mutation_threshold and i != 0:
            # split_list.append((copy.deepcopy(i), area_ratio))
            split_list.append(copy.deepcopy(i))
            
    cap.release()

    return split_list, num_frames

# 背景建模方法对前景区域进行评估
def foreground_estimation(frame, fgbg):
    # 对图像帧进行高斯背景建模
    fgmask = fgbg.apply(frame)
    # 形态学去噪
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 开运算去噪 
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)

    return fgmask

def get_bgmm_exists_flag(fgmask):
    bgmm_exists_flag = False
    # 寻找前景
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    t_area = 0
    for cont in contours:
        # 计算轮廓面积
        Area = cv2.contourArea(cont)
        # 过滤面积小于10的形状  
        if Area < 300:  
            continue
        t_area += Area
        # 计数加一
        count += 1

    # 目前区域内前景面积是不是比较大
    if max_area < t_area:
        bgmm_exists_flag = True
    return bgmm_exists_flag

def get_bbox_exists_flag(object_ls, local_box):
    bbox_exists_flag = False
    # 检测框的是否和目前的区域有交集
    for object_dict in object_ls:
        bbox = object_dict["bbox"]
        if iou(bbox, local_box) > 0:
            bbox_exists_flag = True
            break
    return bbox_exists_flag

# 判别给定区域内是否有前景
def exist_fg(fgmask, object_ls, local_box):
    fg_detect_switch = video_bg_extract_v3_hyper_param_dict[FG_DETECTOR_SWITCH]
    assert fg_detect_switch in video_bg_extract_v3_hyper_param_dict[FG_DETECTOR_SWITCH_COLLECTION]
    bgmm_exists_flag = False
    if fg_detect_switch == FG_DETECTOR_ALL_NAME or fg_detect_switch == FG_DETECTOR_BGMM_NAME:
        bgmm_exists_flag = get_bgmm_exists_flag(fgmask)
    
    bbox_exists_flag = False
    if fg_detect_switch == FG_DETECTOR_ALL_NAME or fg_detect_switch == FG_DETECTOR_BBOX_NAME:
        bbox_exists_flag = get_bbox_exists_flag(object_ls, local_box)
          
    return bgmm_exists_flag or bbox_exists_flag

def set_hyper_param(fg_detect_switch):
    video_bg_extract_v3_hyper_param_dict[FG_DETECTOR_SWITCH] = fg_detect_switch

# 暂存局部数据，如果超过阈值，先取众数，然后再存最新的区域。这样可以防止爆内存
def save_temp_datas(save_data_list, data, handel_fun):
    if len(save_data_list) < max_temp_locality_threshold:
        save_data_list.append(data)
    else:
        save_data_list_nd = np.array(save_data_list, dtype=np.int32)
        temp_output = handel_fun(save_data_list_nd)
        save_data_list = []
        save_data_list.append(temp_output)
        save_data_list.append(data)
    return save_data_list

def get_crop_size(origin_height, origin_width):
    crop_cond = int(math.sqrt(crop_num))
    slice_size = (origin_height // crop_cond, origin_width // crop_cond, 3)
    return slice_size


def get_slice_size(img_nd):
    crop_cond = int(math.sqrt(crop_num))
    height = img_nd.shape[0]
    width = img_nd.shape[1]
    # 计算每个切片的大小
    slice_size = (height // crop_cond, width // crop_cond, 3)
    return slice_size

def crop_img(img_nd):
    height = img_nd.shape[0]
    width = img_nd.shape[1]
    # 计算每个切片的大小
    slice_size = get_slice_size(img_nd)
    # 切分数组
    slices = [img_nd[i:i + slice_size[0], j:j + slice_size[1], :] if len(img_nd.shape) == 3 else img_nd[i:i + slice_size[0], j:j + slice_size[1]]
            for i in range(0, height, slice_size[0]) for j in range(0, width, slice_size[1])]
    
    return slices

def get_background_slice(background_slices, fgmask_slices, frame_slices, object_ls, local_height, local_width):
    for i in range(len(fgmask_slices)):
        fgmask_slice = fgmask_slices[i]
        s_i, s_j = get_start_i_j_crop_index(i, local_height, local_width)
        temp_box = (s_j, s_i, s_j + local_width, s_i + local_height)
        if not exist_fg(fgmask_slice, object_ls, temp_box):
            background_slice_list = background_slices[i]
            frame_slice = frame_slices[i]
            background_slice_list = save_temp_datas(background_slice_list, frame_slice, mean_extract) 
            background_slices[i] = background_slice_list 
    return background_slices
        
def init_list_includ_list(need_num):
    list_includ_list = []
    for _ in range(need_num):
        list_includ_list.append([])
    return list_includ_list


def set_start_get_end_index(video, object_list, start_index=-1, end_index=-1):
    if object_list is not None:
        o_l = copy.deepcopy(object_list)
    else:
        o_l = None
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 下面两个if是在区间的情况下设置视频区间范围
    if start_index != -1:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        if object_list is not None:
            o_l = o_l[start_index:]

    end_f = num_frames
    if end_index != -1:
        end_f = end_index
        if object_list is not None:
            o_l = o_l[:end_index]
    start_f = start_index if start_index != -1 else 0
    return video, o_l, start_f, end_f

# 取众数函数，输入和输出都是numpy数组
def video_his(imgs_array):
    final_bg = np.zeros((imgs_array.shape[1],imgs_array.shape[2],3))
    for i in range(imgs_array.shape[1]):
        for j in range(imgs_array.shape[2]):
            b = np.bincount(imgs_array[:,i,j,:][:,0])
            g = np.bincount(imgs_array[:,i,j,:][:,1])
            r = np.bincount(imgs_array[:,i,j,:][:,2])
            b[0],g[0],r[0] = 0,0,0
            final_bg[i][j][0] = np.argmax(b)
            final_bg[i][j][1] = np.argmax(g)
            final_bg[i][j][2] = np.argmax(r)
    return final_bg

def frame_background_estimation(frame, fgbg):
    #对图像帧进行高斯背景建模
    fgmask = fgbg.apply(frame)
    # 形态学去噪
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 开运算去噪 
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)
    #寻找前景
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count=0
    for cont in contours:
        # 计算轮廓面积
        Area = cv2.contourArea(cont)
        # 过滤面积小于10的形状  
        if Area < 300:  
            continue
        # 计数加一
        count += 1  
        #提取矩形坐标
        rect = cv2.boundingRect(cont) 
        # 黑白前景上绘制矩形
        cv2.rectangle(fgmask,(rect[0]-5,rect[1]-5),(rect[0]+rect[2]+10,rect[1]+rect[3]+10),(0xff, 0xff, 0xff), -1)  

    # 小于64一律归为背景
    fgmask[fgmask < 64] = 0 
    # 二值化
    fgmask = fgmask.astype(bool) 
    # 反向前景mask，得到背景mask
    bgmask = ~fgmask 
    # 转换为uint8，为0和255的二值mask
    bgmask = bgmask.astype(np.uint8) 
    # 变为三通道
    bgmask = np.expand_dims(bgmask,2).repeat(3,axis=2) 
    # 背景mask作用到图像帧中
    current_bg = bgmask * frame

    return current_bg

def get_start_i_j_crop_index(his_index, local_height, local_width):
    crop_cond = int(math.sqrt(64))
    temp_a = his_index // crop_cond
    if temp_a > 0:
        temp_b = his_index - (temp_a * crop_cond)
        return int(temp_a * local_height), int(temp_b * local_width)
    else:
        return 0, int(his_index * local_width)

def get_local_bg(current_bg, his_index_list, bg_local_list, local_height, local_width):
    # 计算每个切片的大小
    slice_size = get_slice_size(current_bg)
    
    for t_j in range(len(his_index_list)):
        his_index = his_index_list[t_j]
        his_list_l = bg_local_list[t_j]
        i, j = get_start_i_j_crop_index(his_index, local_height, local_width)
        bg_crop = current_bg[i:i + slice_size[0], j:j + slice_size[1], :]
        his_list_l = save_temp_datas(his_list_l, bg_crop, video_his)
        bg_local_list[t_j] = his_list_l
    return bg_local_list


def get_his(video, his_index_list, local_height, local_width, start_index=-1, end_index=-1):
    video, _, start_f, end_f = set_start_get_end_index(video, object_list=None, start_index=start_index, end_index=end_index)
    # 初始化高斯建模
    fgbg = cv2.createBackgroundSubtractorMOG2()

    bg_local_list = init_list_includ_list(len(his_index_list))
    for _ in tqdm(range(end_f - start_f)):
        video, frame = get_frame(video)
        gauss_frame=cv2.GaussianBlur(frame,(7,7),0)
        current_bg = frame_background_estimation(gauss_frame, fgbg)
        bg_local_list = get_local_bg(current_bg, his_index_list, bg_local_list, local_height, local_width)

    his_list = []
    for b_l in bg_local_list:
        nd_array = np.array(b_l, dtype=np.int32)
        bg_his = video_his(nd_array)
        his_list.append(bg_his)
    
    return his_list

def get_frame(video):
    read_flag, frame = video.read()
    if not read_flag:
        raise RuntimeError("视频帧读取失败")
    return video, frame

def concat_bg(bg_final_slice_list, origin_height, origin_width):
    crop_size = get_crop_size(origin_height, origin_width)
    final_bg = np.zeros((origin_height, origin_width, 3))

    k = 0
    for i in range(0, origin_height, crop_size[0]):
        for j in range(0, origin_width, crop_size[1]):
            final_bg[i:i + crop_size[0], j:j + crop_size[1], :] = bg_final_slice_list[k]
            k += 1
    
    return final_bg


def extract_background(video_path, object_list, start_index=-1, end_index=-1):
    video = cv2.VideoCapture(video_path)
    origin_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    origin_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_cond = int(math.sqrt(64))
    local_height, local_width = (origin_height // crop_cond, origin_width // crop_cond)    
    
    # 初始化高斯建模
    fgbg = cv2.createBackgroundSubtractorMOG2()

    video, object_list, start_f, end_f = set_start_get_end_index(video, object_list, start_index, end_index)

    # 对每一个区间内的视频段进行高斯建模，分割背景并保存无背景区域，计算无背景区域的均值/滤波（TODO），返回背景关键帧
    background_slices = init_list_includ_list(crop_num)
    i = 0
    for _ in tqdm(range(start_f, end_f)):

        video, frame = get_frame(video)        
        # 获取全局的前景区域
        fgmask = foreground_estimation(frame, fgbg)
        # 获取本帧内所有检测框
        # [{'bbox': (986.0, 44.0, 1094.0, 137.0), 'score': 0.82701886, 'class': 'pedestrian'},……] 
        object_l = object_list[i]
        i += 1

        fgmask_slices = crop_img(fgmask)
        frame_slices = crop_img(frame)

        background_slices = get_background_slice(background_slices, fgmask_slices, frame_slices, object_l, local_height, local_width)
    video.release()

    # 如果存在一直有前景的帧，那就对这部分进行高斯建模取众数
    his_index_list = []
    bg_final_slice_list = []
    for j in range(len(background_slices)):
        background_slice_list = background_slices[j]
        if len(background_slice_list) == 0:
            his_index_list.append(j)
            bg_final_slice_list.append(None)
        else:
            bg_slice_nd = np.array(background_slice_list, dtype=np.int32)
            mean_bg = mean_extract(bg_slice_nd)
            bg_final_slice_list.append(mean_bg)
 
    if len(his_index_list) != 0:
        print("高斯建模取众数出背景")
        video = cv2.VideoCapture(video_path)
        his_list = get_his(video, his_index_list, local_height, local_width, start_index, end_index)
        for k in range(len(his_index_list)):
            his_index = his_index_list[k]
            bg_final_slice_list[his_index] = his_list[k]
        video.release()

    return concat_bg(bg_final_slice_list, origin_height, origin_width)       

def get_backgrounds(video_path, compress_path, split_list):
    load_video_path = str_to_raw(video_path)
    video = cv2.VideoCapture(load_video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 视频宽
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频高
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    person_box_file_path = os.path.join(compress_path, person_box_file_name)
    vehicle_box_file_path = os.path.join(compress_path, vehicle_box_file_name)

    # 读取行人bbox文件的数据 每帧一个列表，每个列表里都是每个bbox的dict，其中'bbox'里边就是一个实例坐标的元组xyxy
    person_detections = get_object_detections_list(person_box_file_path, num_frames)
    
    # 读取车辆bbox文件的数据 每帧一个列表，每个列表里都是每个bbox的dict，其中'bbox'里边就是一个实例坐标的元组xyxy
    vehicle_detections = get_object_detections_list(vehicle_box_file_path, num_frames)

    # 这里判定一下哪个是最大的列表
    main_object_list = vehicle_detections

    main_object_len = len(main_object_list)

    # 合并两个list
    for i in range(main_object_len):
        person_frame_list = person_detections[i]
        main_frame_objects = main_object_list[i]
        main_frame_objects.extend(person_frame_list)

    main_object_list = set_hop_count_for_bbox(main_object_list)
    main_object_list = check_leak_detection_and_supplement_bbox(main_object_list, video_height, video_width)
    
    result_list = []
    
    if len(split_list) == 0:
        background_key = extract_background(load_video_path, main_object_list)
        result_list.append(copy.deepcopy(background_key))
    else:
        split_list_len = len(split_list)
        start_index = 0
        for j in range(split_list_len):
            split_index = split_list[j]
            background_key = extract_background(load_video_path, main_object_list, start_index=start_index, end_index=split_index)
            result_list.append(copy.deepcopy(background_key))
            start_index = split_index
            if j == split_list_len - 1:
                background_key = extract_background(load_video_path, main_object_list, start_index=start_index, end_index=-1)
                result_list.append(copy.deepcopy(background_key))
                               
    return result_list


def video_bg_extract_v3(video_path, compress_path):
    # 高斯全局建模获取split_list，有几个区间就决定了最终生成多少个背景关键帧
    print("获取视频背景建模区间")
    split_list, num_frames = get_split_list(video_path)
    # 过滤掉不合适的区间
    if len(split_list) > 0:
        split_list = check_section(split_list)
    if not frame_per_section == -1:
        split_list = add_section(split_list, num_frames) 
    print("视频分割区间：", end=" ")
    print(split_list)
    print("生成背景关键帧")
    # 对每一个区间内的视频段进行高斯建模，分割背景并保存无背景区域，计算无背景区域的均值/滤波（TODO），返回背景关键帧
    bg_list = get_backgrounds(video_path, compress_path, split_list)

    json_dict = {"n_f": num_frames, "split_list": split_list}
    # 存json
    with open(os.path.join(compress_path, "split_list.json"), "w") as f:
        json.dump(json_dict, f)
    bg_index = 0
    for bg in bg_list:
        bg_path = os.path.join(compress_path, str(bg_index) + ".png")
        cv2.imwrite(bg_path, bg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        bg_index += 1

    

    
# TODO 把帧数也存到json里,  split_list也存进去  
if __name__ == "__main__":
    # video_path = "videos/bg_test/video_crop59.mp4"
    # out_path = "videos/bg_test/test_v3_59.png"
    video_path = "D:/PythonProjects/datas/video_datas/video_session3_right_2k.mp4"
    compress_path = "D:/PythonProjects/datas/video_datas/video_session3_right_2k.mp4_comp"
    # out_path = "videos/bg_test/video_crop5_test_2k5_bg.png"
    print("视频路径: " + str(video_path))
    # print("输出路径: " + str(out_path))
    video_bg_extract_v3(video_path, compress_path)
    # cv2.imwrite(out_path, bgs[0], [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print('-----Finished Extract ' + video_path + ' -----')
    # cv2.imwrite('./'+args.input.split('/')[-1]+'_bg.png', bg)
