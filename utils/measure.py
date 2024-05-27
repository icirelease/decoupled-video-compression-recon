import os
import cv2
import numpy as np

import torch
from pytorch_msssim import ms_ssim

def get_bit_size(file_path):
    return os.path.getsize(file_path) * 8

def get_pixels_num(video_origin_path):
    # opencv读取视频对象
    video = cv2.VideoCapture(video_origin_path)
    # 视频宽
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频高
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 总帧数
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()
    
    return width * height * num_frames

def count_bpp(comp_size, pixels_num):
    assert comp_size is not None
    assert pixels_num is not None
    assert pixels_num > 0
    return comp_size / pixels_num

def get_bpp(tar_path, video_path):
    bit_size = get_bit_size(tar_path)
    pixels_num = get_pixels_num(video_path)
    return count_bpp(bit_size, pixels_num)

def msssim_fn_single(fake_numpy, gt_numpy):
    H, W, C = gt_numpy.shape
    f_n = fake_numpy.reshape(1, C, H, W)
    g_n = gt_numpy.reshape(1, C, H, W)
    f_n = np.array(f_n, dtype=np.float32)
    g_n = np.array(g_n, dtype=np.float32)
    fake = torch.from_numpy(f_n)
    gt = torch.from_numpy(g_n)

    msssim = ms_ssim(fake.detach(), gt.detach(), data_range=1, size_average=False)

    msssim_nd = msssim.numpy()
    assert sum(msssim_nd.shape) == 1
    return msssim_nd[0]

