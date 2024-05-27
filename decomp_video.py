
import os
import sys
sys.path.append("manager")
sys.path.append("background")
sys.path.append("foreground")
sys.path.append("pipeline")
sys.path.append("utils")

import json
import argparse
from pipeline.fg_handle import recon_fg
from pipeline.recon_video import generate_video
from utils.util import copy_decomp_video

def get_parser():
    parser = argparse.ArgumentParser(detarscription="Detectron2 Demo")
    parser.add_argument("-video_path",help="Path to video.")
    parser.add_argument("-tar_path",help="Path to video compression path.")
    parser.add_argument(
        "-psnr_bg_th",
        default="20",
    )
    parser.add_argument(
        "-out_dir",
        default="out_dir",
    )

    return parser

if __name__ == "__main__": 
    args = get_parser().parse_args()
    video_path = args.video_path
    tar_path = args.tar_path
    out_dir = args.out_dir
    psnr = int(args.psnr_bg_th)

    out_path = os.path.join(out_dir, str(os.path.basename(video_path)), "psnr_" + str(psnr))
    out_result_dict = {}
    out_comp_json_path = os.path.join(out_path, "decomp_result.json")
    print("重建前景")
    compress_path = recon_fg(tar_path)
    print("重建视频")
    psnr_avg, ms_ssim_avg = generate_video(compress_path, video_path)
    print("拷贝数据到", end=" ")
    print(out_path)
    copy_decomp_video(compress_path, out_path)

    out_result_dict["args"] = str(args)
    out_result_dict["psnr_th"] = str(psnr)
    out_result_dict["out_path"] = out_path
    out_result_dict["psnr_avg"] = str(psnr_avg)
    out_result_dict["ms_ssim_avg"] = str(ms_ssim_avg)
    with open(out_comp_json_path, "w") as f:
        json.dump(out_result_dict, f)

