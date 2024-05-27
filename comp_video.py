import os
import sys
import json
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.append("manager")
sys.path.append("background")
sys.path.append("foreground")
sys.path.append("pipeline")
sys.path.append("utils")

import cv2

from background.global_bg_extract import video_bg_extract_v3, set_hyper_param
from background.res_bg_extract import video_bg_extract_v4_shiyan
from utils.measure import get_bpp
from pipeline.fg_handle import fg_extract, fg_clean

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument("-video_path",help="Path to video.")
    parser.add_argument("-fg_detect_switch", default="all")
    parser.add_argument(
        "-psnr_bg_th",
        default="20",
    )
    parser.add_argument(
        "-out_dir",
        default="out_dir_df",
    )
    parser.add_argument(
        "-max_row",
        type=float,
        default=108
    )
    parser.add_argument(
        "-max_col",
        type=float,
        default=192
    )

    return parser


if __name__ == "__main__": 
    args = get_parser().parse_args()
    video_path = args.video_path
    fg_detect_switch = str(args.fg_detect_switch)
    max_row_str = int(args.max_row)
    max_col_str = int(args.max_col)
    out_dir = args.out_dir
    video = cv2.VideoCapture(video_path)
    psnr = int(args.psnr_bg_th)
    # 总帧数
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()

    compress_path = fg_extract(video_path)
    tar_path = fg_clean(compress_path)

    print("本轮参数")
    print(video_path, compress_path, fg_detect_switch)
    print("提取全局平均背景")
    set_hyper_param(fg_detect_switch=fg_detect_switch)
    video_bg_extract_v3(video_path, compress_path)

    print("提取全局变化背景矩阵")
    video_bg_extract_v4_shiyan(video_path, compress_path, psnr=psnr, max_row=int(max_row_str), max_col=int(max_col_str))
    
    out_path = os.path.join(out_dir, str(os.path.basename(video_path)), "psnr_" + str(psnr))
    out_result_dict = {}
    out_comp_json_path = os.path.join(out_path, "comp_result.json")
    print("拷贝数据到", end=" ")
    print(out_path)

    out_dir_path = os.path.join(out_path, os.path.basename(compress_path))
    shutil.copytree(tar_path, out_dir_path)
    bpp = get_bpp(tar_path, video_path)
    print("avg bpp is ", str(bpp))

    out_result_dict["args"] = str(args)
    out_result_dict["psnr_th"] = str(psnr)
    out_result_dict["out_path"] = out_path
    out_result_dict["bpp"] = str(bpp)
    print(out_result_dict)
    with open(out_comp_json_path, "w") as f:
        json.dump(out_result_dict, f)