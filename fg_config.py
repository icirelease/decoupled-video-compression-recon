import argparse
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from manager.maskdino import add_maskdino_config


# 运行设备
device = "cuda:0"

# 实例分割模型配置
# 模型配置文件
config_file = "path/config_file"
inst_checkpoint = "path/inst_checkpoint"
# Minimum score for instance predictions to be shown
confidence_threshold = 0.3

# 初始化实例分割模型配置
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default=config_file,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("-input",help="Path to video file.")
    parser.add_argument(
        "-confidence-threshold",
        type=float,
        default=confidence_threshold,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', inst_checkpoint],
        nargs=argparse.REMAINDER,
    )
    return parser

inst_args = get_parser().parse_args()
inst_cfg = setup_cfg(inst_args)

# 人体姿态配置
human_pose_config = "path/human_pose_config"
human_pose_checkpoint = "path/human_pose_checkpoint"

# 车辆姿态配置
vehicle_pose_checkpoint = "path/vehicle_pose_checkpoint"

# 重建模型配置
human_generator_model = "path/human_generator_model"
vehicle_generator_model = "path/vehicle_generator_model"