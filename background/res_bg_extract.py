import numpy as np
import cv2
from tqdm import tqdm
import os
import copy
import json
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.track_util import load_mot, iou


"""
    这是第四版的背景提取逻辑
        将这部分的输出数据分为三个部分。1 一张背景图片.png 2 一个帧号和位置的索引表 3 存储背景图片变化的numpy矩阵
        算法思路:
            背景不是完全不变的, 整个视频用一张图片做背景的话, PSNR不会高。 如果存多张背景, 又会有不少的冗余数据。
        所以就是要以一个背景为基础, 再存一个变化矩阵在相应位置去更改背景。
        工程思路: 
            背景图片以第一帧为基准,最终这部分数据存一张无损的PNG图片。
            从第二帧开始维护一个背景变化矩阵, numpy格式。如果检测到存在背景变化, 且超过设置的阈值(pixel_change_ratio_threshold),
        则将图像块记录进变化矩阵。并将图像块的帧号以及位置编码记录给一个json格式的数据frame_num_and_pos_dict。下面的例子
        展示这种存储逻辑。
        
        图像块shape: [H_i, W_j, C]
        变化矩阵:[K, H_i, W_j, C] 这里设置整个视频存了K个图像块
        frame_num_and_pos_dict: {"帧号":[位置编码]}  比如{"1": [2, 15], "7": [8, 106, 108, ……], ……}

            解释一下为什么这样存数据。首先对于图像的压缩, PNG是一个比较成熟且好使的压缩算法。正好我们又要存一个基础的背景
        那么这部分就存了一个图像。变化矩阵的的长度K是无法预知的, 所以变化矩阵就存numpy数据, 并想办法进行压缩。
        frame_num_and_pos_dict中存储的就是变化矩阵中每一个图像块的帧号以及位置信息。假设这个dict的长度为n, 每个dict里面
        有一个list。那么所有的list的长度加起来就等于K。并且这种数据结构也可以支持随机访问。比如想重建第10帧的某一个图像块
        就把前面九帧的list长度获取到, 然后看一下需要访问的第10帧哪一个位置的图像块, 就可以定位到变化矩阵里相应图像块的索引。
        这里随机访问的时间复杂度为O(n), n为frame_num_and_pos_dict的长度。因为变化矩阵图像块的索引是和这个dict有强相关关系
        所以遍历的时候只需要把前面帧的list长度加起来再加上当前帧以及当前位置在list里的一个索引, 就能唯一确认变化矩阵的图像块。
        frame_num_and_pos_dict里面因为存的基本上是非结构化数据, 所以json格式比较适合。json格式也可以进行进一步压缩。
            最后再解释一下, 分图像明明会产生两个数据, 为什么位置编码只存一个数就行。假设要把一个图像分为四块, 位置索引简记
        为(0, 0), (0, 1), (1, 0), (1, 1)。但其实用0, 1, 2, 3就可以唯一的确认一个图像块的位置了。拿到一个编号, 先用最大的
        列数除一下,  能整除的那部分的结果是行标号, 余数就是列标号。
"""

# "max_row_num": 216,
# "max_col_num": 384,
# 超参数
hyper_param_dict = {
    "max_row_num": 108,
    "max_col_num": 192,
    "border_threshold": 20,
    "min_border_bbox_threshold": 200,
    "max_hop_count": 80,
    "iou_threshold": 0.7,
    "pixel_value_change_threshold": 45,
    "pixel_change_ratio_threshold": 0.5,
    "interval_frame": 4,
    "min_psnr_threshold": 28.5,
    "person_box_file_name": "person_box.txt",
    "vehicle_box_file_name": "vehicle_box.txt",
    "base_bg_name": "0.png",
    "ndarray_name": "a.npy",
    "change_dict_name": "c.json"
}

# 对于一个宽高为1980 * 1080的视频, 刚好能分成198 * 108个10 * 10的小图像块
MAX_ROW_NUM = "max_row_num"
MAX_COL_NUM = "max_col_num"

# 最低PSNR阈值
MIN_PSNR_THRESHOLD = "min_psnr_threshold"

# 图像边界阈值
BORDER_THRESHOLD = "border_threshold"

# 在边界上的bbox是否小于一定长度
MIN_BORDER_BBOX_THRESHOLD = "min_border_bbox_threshold"

# 检测有没有漏检的框的最大跳数
MAX_HOP_COUNT = "max_hop_count"

# 相邻两帧bbox的IOU阈值，当前帧的bbox，在下一帧的序列中找到IOU大于此阈值的，才算是没有发生漏检
IOU_THRESHOLD = "iou_threshold"

# 变化像素值的阈值
PIXEL_VALUE_CHANGE_THRESHOLD = "pixel_value_change_threshold"

# 所有通道中像素变化的比例
PIXEL_CHANGE_RATIO_THRESHOLD = "pixel_change_ratio_threshold"

# 高斯建模的间隔帧数
INTERVAL_FRAME = "interval_frame"

# 行人检测框的文件名
PERSON_BOX_FILE_NAME = "person_box_file_name"

# 车辆检测框的文件名
VEHICLE_BOX_FILE_NAME = "vehicle_box_file_name"

# 背景名称
BASE_BG_NAME = "base_bg_name"

# 变化矩阵文件名称
NDARRAY_NAME = "ndarray_name"

# frame_num_and_pos_dict文件名称
CHANGE_DICT_NAME = "change_dict_name"

def str_to_raw(s):
    raw_map = {8:r'\b', 7:r'\a', 12:r'\f', 10:r'\n', 13:r'\r', 9:r'\t', 11:r'\v'}
    return r''.join(i if ord(i) > 32 else raw_map.get(ord(i), i) for i in s)

# 检查bbox的边长是否小于某个值
def check_bbox_close_border(bbox):
    x1, y1, x2, y2 = bbox
    if abs(x2 - x1) < hyper_param_dict[MIN_BORDER_BBOX_THRESHOLD]:
        return True
    if abs(y2 - y1) < hyper_param_dict[MIN_BORDER_BBOX_THRESHOLD]:
        return True
    return False

# 检查bbox是否在图像边界
def bbox_close_to_the_border(bbox, frame_h, frame_w):
    x1, y1, x2, y2 = bbox    
    if (x1 - 0) < hyper_param_dict[BORDER_THRESHOLD]:
        return check_bbox_close_border(bbox)
    if (y1 - 0) < hyper_param_dict[BORDER_THRESHOLD]:
        return check_bbox_close_border(bbox)
    if (frame_w - x2) < hyper_param_dict[BORDER_THRESHOLD]:
        return check_bbox_close_border(bbox)
    if (frame_h - y2) < hyper_param_dict[BORDER_THRESHOLD]:
        return check_bbox_close_border(bbox)
    return False

# 对比当前帧和下一帧的所有bbox，如果当前帧的bbox不在画面边界，
# 且下一帧的所有bbox里都没有和当前bbox的iou大于某个阈值的情况，
# 则认为目标检测发生了漏检，补充当前帧的bbox到下一帧的序列。
def check_leak_detection_and_supplement_bbox(main_list, frame_h, frame_w):
    for i in range(len(main_list) - 1):
        this_frame_list = main_list[i]
        next_frame_list = main_list[i + 1]
        for bbox_dict in this_frame_list:
            bbox = bbox_dict["bbox"]
            # 看当前的bbox在不在边界
            if(bbox_close_to_the_border(bbox, frame_h, frame_w)):
                continue
            has_flag = False
            for next_bbox_dict in next_frame_list:
                next_bbox = next_bbox_dict["bbox"]
                if iou(bbox1=bbox, bbox2=next_bbox) > hyper_param_dict[IOU_THRESHOLD]:
                    has_flag = True
            if not has_flag:
                hop_count = bbox_dict["hop_count"]
                if not hop_count < hyper_param_dict[MAX_HOP_COUNT]:
                    continue
                c_bbox_dict = copy.deepcopy(bbox_dict)
                c_bbox_dict["hop_count"] = hop_count + 1
                next_frame_list.append(c_bbox_dict)

    return main_list

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

def set_hop_count_for_bbox(main_object_list):
    for frame_bbox_list in main_object_list:
        for fg_bbox_dict in frame_bbox_list:
            fg_bbox_dict["hop_count"] = 0
    return main_object_list
    

# 按目标框的中心点扩充目标框 避免目标框没有括住目标的情况，默认扩大1.2倍
def expand_bbox(bbox, frame_h, frame_w, scale=1.2):  
    x1, y1, x2, y2 = bbox
    center_x = (x2 + x1) / 2
    center_y = (y2 + y1) / 2  
    dx = (x2 - x1) * scale / 2
    dy = (y2 - y1) * scale / 2
    new_x1 = int(center_x - dx) if int(center_x - dx) > 0 else 0 
    new_y1 = int(center_y - dy) if int(center_y - dy) > 0 else 0
    new_x2 = int(center_x + dx) if int(center_x + dx) <= frame_w else frame_w
    new_y2 = int(center_y + dy) if int(center_y + dy) <= frame_h else frame_h
    return (new_x1, new_y1, new_x2, new_y2)

# 检测画面是否发生了变化
def picture_changes(bg_block, frame_block):
    # 使用PSNR判定图像是否变化
    return psnr(bg_block, frame_block) < hyper_param_dict[MIN_PSNR_THRESHOLD]

# 获取图像块的尺寸
def get_crop_size(origin_height, origin_width):
    max_row_num = hyper_param_dict[MAX_ROW_NUM]
    max_col_num = hyper_param_dict[MAX_COL_NUM]
    # TODO 不能整除的时候要处理
    slice_size = (origin_height // max_row_num, origin_width // max_col_num, 3)
    return slice_size

# 根据索引和图像块宽高获取坐标
def get_coordinates_by_pos_index(pos_index, local_height, local_width):
    s_i, s_j = get_start_i_j_crop_index(pos_index, local_height, local_width)
    x1 = s_j
    x2 = s_j + local_width
    y1 = s_i
    y2 = s_i + local_height
    return (x1, y1, x2, y2)


def get_coordinates(origin_height, origin_width):
    slice_size = get_crop_size(origin_height, origin_width)
    coordinate_list = [(j, i, j + slice_size[1], i + slice_size[0])
            for i in range(0, origin_height, slice_size[0]) for j in range(0, origin_width, slice_size[1])]
    return coordinate_list

# 切分图片
def crop_img(img_nd):
    height = img_nd.shape[0]
    width = img_nd.shape[1]
    # 计算每个切片的大小
    slice_size = get_crop_size(height, width)
    # 切分数组
    slices = [img_nd[i:i + slice_size[0], j:j + slice_size[1], :] if len(img_nd.shape) == 3 else img_nd[i:i + slice_size[0], j:j + slice_size[1]]
            for i in range(0, height, slice_size[0]) for j in range(0, width, slice_size[1])]    
    return slices


# 位置编码更换，获取的是图像块开始像素的i和j
def get_start_i_j_crop_index(pos_index, local_height, local_width):
    max_col_num = hyper_param_dict[MAX_COL_NUM]
    temp_a = pos_index // max_col_num
    if temp_a > 0:
        temp_b = pos_index - (temp_a * max_col_num)
        return int(temp_a * local_height), int(temp_b * local_width)
    else:
        return 0, int(pos_index * local_width)



def get_frame(video):
    # [1080, 1920, 3]
    read_flag, frame = video.read()
    if not read_flag:
        raise RuntimeError("视频帧读取失败")
    gauss_frame=cv2.GaussianBlur(frame, (5, 5), 0.4, 0.4)
    return video, gauss_frame

def get_main_object_list(person_box_file_path, vehicle_box_file_path, num_frames):
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

    return main_object_list

def get_int_coordinates(bbox_t):
    x1, y1, x2, y2 = bbox_t
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    return x1, y1, x2, y2


# 获取背景变化矩阵以及frame_num_and_pos_dict, 更新first_frame
def get_bg_update_array(video, main_object_list, first_bg):
    bg_update_list = []
    frame_num_and_pos_dict = {}
    # 初始化
    origin_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    origin_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 后续逻辑需要一些参数，也放这里了
    frame_num_and_pos_dict["mx_col"] = hyper_param_dict[MAX_COL_NUM]
    # [10, 10, 3] calc_blocks_size_tuple
    slice_size = get_crop_size(origin_height, origin_width)
    # block_rows, 
    local_height = slice_size[0]
    local_width = slice_size[1]
    # 帧数
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) 
    frame_num_and_pos_dict["num_frames"] = num_frames   
    bg_slices = crop_img(copy.deepcopy(first_bg))
    for f_i in tqdm(range(num_frames)):
        video, frame = get_frame(video)
        # if f_i % 10 != 0:
        #     continue        
        # 记录block位置的 img改成block
        img_pos_list = []
        frame_bbox_list = main_object_list[f_i]

        # 图像切片
        frame_slices = crop_img(frame)
        for slice_index in range(len(bg_slices)):
            local_bg = bg_slices[slice_index]
            slice_bbox = get_coordinates_by_pos_index(slice_index, local_height, local_width)
            has_fg_flag = False
            for fg_bbox_dict in frame_bbox_list:
                fg_bbox = fg_bbox_dict["bbox"]
                fg_bbox = expand_bbox(fg_bbox, origin_height, origin_width, scale=1.3)
                if iou(bbox1=slice_bbox, bbox2=fg_bbox) > 0:
                    has_fg_flag = True
                    break
                
            if has_fg_flag:
                continue
            
            local_img = frame_slices[slice_index]
            # 检测到背景有变化
            if picture_changes(local_bg, local_img):
                # 有变化就更新当前的背景，别只和第一帧图像比，那样存的太多了
                bg_slices[slice_index] = local_img                
                bg_update_list.append(copy.deepcopy(local_img - local_bg))         
                img_pos_list.append(slice_index)
        if len(img_pos_list) > 0:
            frame_num_and_pos_dict[str(f_i)] = img_pos_list 

    bg_update_array = np.array(bg_update_list)
    return bg_update_array, frame_num_and_pos_dict
    

def video_bg_extract_v4(video_path, compress_path):
    # 初始化
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    first_bg = cv2.imread(os.path.join(compress_path, "0.png"))
    
    
    print("获取目标框数据")
    person_box_file_path = os.path.join(compress_path, hyper_param_dict[PERSON_BOX_FILE_NAME])
    vehicle_box_file_path = os.path.join(compress_path, hyper_param_dict[VEHICLE_BOX_FILE_NAME])
    
    main_object_list = get_main_object_list(person_box_file_path, vehicle_box_file_path, num_frames)
    main_object_list = set_hop_count_for_bbox(main_object_list)
    main_object_list = check_leak_detection_and_supplement_bbox(main_object_list, frame_h, frame_w)
    
    print("更新基础背景, 获取背景变化矩阵以及frame_num_and_pos_dict")
    bg_update_array, frame_num_and_pos_dict = get_bg_update_array(
        video, main_object_list, first_bg)
    video.release()

    print("存储所有数据")
    # base_bg_path = os.path.join(compress_path, hyper_param_dict[BASE_BG_NAME])
    ndarray_path = os.path.join(compress_path, hyper_param_dict[NDARRAY_NAME])
    change_json_path = os.path.join(compress_path, hyper_param_dict[CHANGE_DICT_NAME])

    # cv2.imwrite(base_bg_path, first_bg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("PSNR阈值", end=" ")
    print(hyper_param_dict[MIN_PSNR_THRESHOLD])
    print("变化矩阵shape", end=" ")
    print(bg_update_array.shape)
    np.save(ndarray_path, bg_update_array)
    print("frame_num_and_pos_dict的长度(变化帧数)", end=" ")
    print(len(frame_num_and_pos_dict))
    with open(change_json_path, "w") as f:
        json.dump(frame_num_and_pos_dict, f)

def video_bg_extract_v4_psnr(video_path, compress_path, psnr=None):
    if not psnr is None:
        hyper_param_dict[MIN_PSNR_THRESHOLD] = psnr
    video_bg_extract_v4(video_path, compress_path)
         
def video_bg_extract_v4_shiyan(video_path, compress_path, psnr=None, max_row=None, max_col=None):
    if not psnr is None:
        hyper_param_dict[MIN_PSNR_THRESHOLD] = psnr
    if not max_row is None:
        hyper_param_dict[MAX_ROW_NUM] = max_row
    if not max_col is None:
        hyper_param_dict[MAX_COL_NUM] = max_col
    print("设置的参数为 psnr, max_row, max_col", end=" ")
    print(hyper_param_dict[MIN_PSNR_THRESHOLD], hyper_param_dict[MAX_ROW_NUM], hyper_param_dict[MAX_COL_NUM])
    video_bg_extract_v4(video_path, compress_path)

    
# TODO 把帧数也存到json里,  split_list也存进去  
if __name__ == "__main__":
    
    # video_path = "videos/bg_test/video_crop59.mp4"
    # out_path = "videos/bg_test/test_v3_59.png"
    # video_path = "D:/PythonProjects/datas/video_datas/video_session3_right_2k.mp4"
    # compress_path = "D:/PythonProjects/datas/video_datas/video_session3_right_2k.mp4_comp"
    video_path = "D:/PythonProjects/datas/video_datas/brno2016/video_session2_left_2k.mp4"
    compress_path = "D:/PythonProjects/datas/video_datas/video_session2_left_2k.mp4_comp"
    # video_path = "test_psnr1/video_session3_right_2k.mp4"
    # compress_path = "test_psnr1/video_session3_right_2k.mp4_comp"
    # video_path = "video_session3_right_2k.mp4"
    # compress_path = "video_session3_right_2k.mp4_comp"
    # out_path = "videos/bg_test/video_crop5_test_2k5_bg.png"
    print("视频路径: " + str(video_path))
    # video_bg_extract_v4(video_path, compress_path)
    video_bg_extract_v4_shiyan(video_path, compress_path, psnr=18, max_row=108, max_col=192)
    # print("输出路径: " + str(out_path))
    # cv2.imwrite(out_path, bgs[0], [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print('-----Finished Extract ' + video_path + ' -----')
    # cv2.imwrite('./'+args.input.split('/')[-1]+'_bg.png', bg)
