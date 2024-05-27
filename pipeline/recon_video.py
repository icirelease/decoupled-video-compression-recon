import os
import tarfile
import copy
from  tqdm import tqdm
import json
import math
import cv2
import numpy as np
from background.res_bg_extract import bbox_close_to_the_border
from utils.track_util import iou
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils.measure import msssim_fn_single

# 判断输入的字符串是否能转换成int类型，主要用于检测是否为存放视频帧数据的文件夹
def string_convertible_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# 根据索引和图像块宽高获取坐标
def get_coordinates_by_pos_index(pos_index, local_height, local_width, max_col_num):
    s_i, s_j = get_start_i_j_crop_index(pos_index, local_height, local_width, max_col_num)
    x1 = s_j
    x2 = s_j + local_width
    y1 = s_i
    y2 = s_i + local_height
    return (x1, y1, x2, y2)

def get_start_i_j_crop_index(pos_index, local_height, local_width, max_col_num):
    temp_a = pos_index // max_col_num
    if temp_a > 0:
        temp_b = pos_index - (temp_a * max_col_num)
        return int(temp_a * local_height), int(temp_b * local_width)
    else:
        return 0, int(pos_index * local_width)

def init_decomp(comp_dir):
    # 读json
    json_path = os.path.join(comp_dir, "c.json")
    frame_num_and_pos_dict = None
    with open(json_path, "r") as f:
        frame_num_and_pos_dict = json.load(f)

    bg_update_array = np.load(os.path.join(comp_dir, "a.npy"))

    base_bg = cv2.imread(os.path.join(comp_dir, "0.png"))

    return base_bg, bg_update_array, frame_num_and_pos_dict

def get_current_frame_bg(base_bg, bg_update_array, frame_num_and_pos_dict: dict, f_i, nd_index):
    f_i_str = str(f_i)

    try:
        pos_list = frame_num_and_pos_dict[f_i_str]
    except KeyError:
        return base_bg, nd_index   
    mx_col = frame_num_and_pos_dict["mx_col"]
    for pos_index in pos_list:
        img_block = bg_update_array[nd_index, :, :, :]
        local_height = img_block.shape[0]
        local_width = img_block.shape[1]
        bbox_t = get_coordinates_by_pos_index(pos_index, local_height, local_width, mx_col)
        x1 = bbox_t[0]
        y1 = bbox_t[1]
        x2 = bbox_t[2]
        y2 = bbox_t[3]
        try:
            base_bg[y1: y2, x1: x2] += img_block
        except Exception as e:
            print(bbox_t)
            raise e
        nd_index += 1
    
    return base_bg, nd_index

'''
输出视频类
输出一个可视化的视频
'''
class Video:
   def __init__(self,name,fps,width,height):
      self.name = os.path.splitext(name)[0]+'_decomp.mp4'
      self.fps = fps
      self.width = width
      self.height = height
      self.video_writer = cv2.VideoWriter(self.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,(self.width,self.height))
   
   def write_frame(self,frame):
       self.video_writer.write(frame)
 
   def save_video(self):
       self.video_writer.release()

class FrameBbox():  
    def __init__(self, frame_w, frame_h) -> None:
        # 前一帧的前景实例序列
        self.befor_inst_list = None
        # 视频宽高
        self.frame_w = frame_w
        self.frame_h = frame_h
        # iou阈值
        self.iou_th = 0.3
        # 最大前景补回次数
        self.max_hop = 40
    '''
        前景补回过程
        输入当前帧的前景实例序列
        与这个对象中记录的前一帧前景实例序列进行比对，补充缺失实例
        [dict{"bbox": (x1, y1, x2, y2), "inst_img": nd_array, hop: int} ……]
        返回缺失前景实例
    '''
    def get_supplement_list(self, current_inst_list: list):
        if self.befor_inst_list is None:
            self.befor_inst_list = current_inst_list
            return []
        
        supplement_list = []
        for befor_inst_dict in self.befor_inst_list:
            befor_bbox = befor_inst_dict["bbox"]
            if bbox_close_to_the_border(befor_bbox, frame_h=self.frame_h, frame_w=self.frame_w):
                    continue
            if not befor_inst_dict["hop"] < self.max_hop:
                    continue

            has_flag = False
            for current_inst_dict in current_inst_list:
                cur_bbox = current_inst_dict["bbox"]                
                if iou(befor_bbox, cur_bbox) > self.iou_th:
                    has_flag = True

            if not has_flag:
                befor_inst_dict["hop"] += 1
                supplement_list.append(befor_inst_dict)

        current_inst_list.extend(supplement_list)
        self.befor_inst_list = current_inst_list
        return supplement_list

def concat_supplement_fg(fg_concat, supplement_list):
    for supplement_dict in supplement_list:
        bbox = supplement_dict["bbox"]
        inst_img = supplement_dict["inst_img"]
        x1, y1, x2, y2 = bbox
        fg_concat[y1: y2, x1: x2] = inst_img
    return fg_concat



'''
    拼接前景与背景，重建视频
    输入重建前景后的压缩目录以及真实视频目录
    输出视频的重建质量psnr和ms-ssim
'''
def generate_video(decomp_dir, origin_video_path):
    # 初始化背景
    base_bg, bg_update_array, frame_num_and_pos_dict = init_decomp(decomp_dir)
    # 读取原始视频
    video_real = cv2.VideoCapture(origin_video_path)
    num_frames_real = int(video_real.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = video_real.get(cv2.CAP_PROP_FPS)

    print("num_frames_real", end=" ")
    print(num_frames_real)
    # 初始化一个输出
    output = Video(decomp_dir, frames_per_second, base_bg.shape[1], base_bg.shape[0])
    frame_bbox_inst = FrameBbox(frame_h=base_bg.shape[0], frame_w=base_bg.shape[1])

    # 获取目录下的所有内容
    all_items = os.listdir(decomp_dir)

    # 筛选出所有子目录
    subdirectories = [f for f in all_items if os.path.isdir(os.path.join(decomp_dir, f)) and string_convertible_to_int(f)]
    subdirectories.sort(key =lambda x:(len(x),x))
    frame_i = 0
    nd_index = 0
    psnr_value_list = []
    ms_ssim_value_list = []
    background = base_bg
    for sub in tqdm(subdirectories):
        # 重建当前帧背景
        background, nd_index = get_current_frame_bg(background, bg_update_array, frame_num_and_pos_dict, frame_i, nd_index)
        background_temp = copy.deepcopy(background)
        # 初始化前景图层
        fg_concat = np.zeros((background.shape[0],background.shape[1],3))
        # 获取当前帧的前景路径
        frame_dir_path = os.path.join(decomp_dir, sub)
        all_fgs = os.listdir(frame_dir_path)
        cur_fg_paths = [os.path.join(frame_dir_path, f) for f in all_fgs if os.path.isfile(os.path.join(frame_dir_path, f))]
        # 贴重建前景，并记录当前帧前景对象
        cur_inst_list = []
        for i in cur_fg_paths:
            # 从文件名里获取前景实例的位置
            fg_topleft = os.path.splitext(i)[0].split('/')[-1].split('_')
            fg_x0 = int(fg_topleft[0])
            fg_y0 = int(fg_topleft[1])
            fg_x1 = int(fg_topleft[2])
            fg_y1 = int(fg_topleft[3])

            fg = cv2.imread(i)
            need_h = fg_concat[fg_y0:fg_y1, fg_x0:fg_x1].shape[0]
            need_w = fg_concat[fg_y0:fg_y1, fg_x0:fg_x1].shape[1]
            fg = cv2.resize(fg, (need_w, need_h))
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, element)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            fg_height = fg.shape[0]
            fg_width = fg.shape[1]
            cur_inst_list.append({"bbox": (fg_x0, fg_y0, fg_x1, fg_y1), "inst_img": fg, "hop": 0})
            fg_concat[fg_y0:fg_height+fg_y0, fg_x0:fg_width+fg_x0] = fg
        # 补充当前帧的前景 
        supplement_list = frame_bbox_inst.get_supplement_list(cur_inst_list)
        fg_concat = concat_supplement_fg(fg_concat, supplement_list)
        # 重建视频帧
        background_temp[fg_concat!=0] = fg_concat[fg_concat!=0]
        # 输出
        output.write_frame(background_temp)
        # 计算重建视频帧图像质量 
        _, real_img = video_real.read()
        psnr_value = psnr(real_img, background_temp)
        if not math.isinf(psnr_value):
            psnr_value_list.append(psnr_value)
            ms_ssim_value = msssim_fn_single(background_temp, real_img)
            ms_ssim_value_list.append(ms_ssim_value)
        frame_i += 1

    # 计算重建质量的平均值
    psnr_avg = np.average(np.array(psnr_value_list, dtype=np.float32))
    ms_ssim_avg = np.average(np.array(ms_ssim_value_list, dtype=np.float32))
    output.save_video()
    print('Final PSNR:', psnr_avg)
    print('Final MS-SSIM:', ms_ssim_avg)
    return psnr_avg, ms_ssim_avg

if __name__ == "__main__":
    pass
