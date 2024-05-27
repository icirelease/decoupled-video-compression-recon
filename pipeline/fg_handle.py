import os
import json
import tqdm
import tarfile
import cv2
import numpy as np
from foreground.predictor import VisualizationDemo
from shutil import rmtree
from manager.HRNet_inference.SimpleHRNet import SimpleHRNet
from fg_config import inst_cfg, human_pose_config, human_pose_checkpoint, vehicle_pose_checkpoint
from foreground.common import init_pose_model, compute_instance_cord_batch, ignore_instance_with_condition, del_imgs
from manager.object_track.iou_tracker import track_iou
from foreground.feature_clean import vehicle_clean,human_clean
from fg_config import human_generator_model, vehicle_generator_model
from utils.comp_decomp_files import decomp_files
from manager.car_pose_inference.vehicle_recon import vehicle_reconstruction
from manager.human_pose_inference.human_recon import human_reconstruction
from foreground.fg_filter_car import filter_key_vehicle
from foreground.fg_filter_key_person import filter_key_person
from utils.comp_decomp_files import comp_files
from utils.util import make_tarfile
'''
    重建前景
    输入压缩视频的目录，利用生成模型对前景目标进行重建。
'''
def recon_fg(decomp_dir):
    if not os.path.isfile(decomp_dir):
        print('输入视频压缩文件不存在！')
        exit()
    print('加载视频压缩文件: ', decomp_dir)
    tarf = tarfile.open(decomp_dir)
    decomp_dir = decomp_dir[:-7]
    tarf.extractall(decomp_dir)

    json_path = os.path.join(decomp_dir, "c.json")
    frame_num_and_pos_dict = None
    with open(json_path, "r") as f:
        frame_num_and_pos_dict = json.load(f)

    decomp_files(decomp_dir, frame_num_and_pos_dict["num_frames"])

    if os.path.exists(os.path.join(decomp_dir,'vehicle_pair.csv')):
        print('重建车辆前景目标...')
        vehicle_reconstruction(decomp_dir, vehicle_generator_model)
    if os.path.exists(os.path.join(decomp_dir,'person_pair.csv')):
        print('重建人体前景目标...')
        human_reconstruction(decomp_dir, human_generator_model)
    return decomp_dir

'''
    人体和车辆的实例分割以及关键点检测步骤
    输入：视频路径
    无输出
'''
def fg_extract(video_path):
    
    cfg = inst_cfg
    if video_path is None or video_path == "":
        raise RuntimeError("请输入视频路径")
    if not os.path.isfile(video_path):
        raise RuntimeError("请确认视频路径" + str(video_path))
    print('开始压缩视频: ' + video_path)
    print('实例分割模型初始化...')
    demo = VisualizationDemo(cfg)
    print('人体抽象特征提取模型初始化...')
    device = "cuda:0"
    human_pose_model = init_pose_model(
        human_pose_config, human_pose_checkpoint, device=device.lower())
    
    print('车辆抽象特征提取模型初始化...')
    vehicle_pose_model = SimpleHRNet(48, 20, vehicle_pose_checkpoint , multiperson=False)

    # opencv读取视频对象
    video = cv2.VideoCapture(video_path)
    # 视频宽
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频高
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 总帧数
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 视频文件名
    basename = os.path.basename(video_path)
    # 创建压缩目录
    compress_path = basename + '_comp/'
    if os.path.exists(compress_path):
        rmtree(compress_path)
    os.makedirs(compress_path)
    # 创建人和车对象的检测框记录文本和关键点记录文本
    person_track_box_file = open(os.path.join(compress_path,'person_box.txt'), 'w')
    vehicle_track_box_file = open(os.path.join(compress_path,'vehicle_box.txt'), 'w')

    human_annotation_file = open(os.path.join(compress_path,'annotation_human.txt'), 'w')
    human_annotation_file.write('name:keypoints_y:keypoints_x'+'\n')
    human_annotation_file.flush()

    vehicle_annotation_file = open(os.path.join(compress_path,'annotation_vehicle.txt'), 'w')
    vehicle_annotation_file.write('name:keypoints_y:keypoints_x'+'\n')
    vehicle_annotation_file.flush()

    # 检测到的车辆数目，每帧一更新
    vehicle_num_detected = 0
    # 检测到的人的数目，每帧一更新
    human_num_detected = 0
    print('提取视频目标抽象特征...')
    frame = 1
    for i in tqdm.tqdm(range(num_frames)):
        # 获取到这一帧的图像和预测box等信息
        vis_frame,predictions = demo.run_on_video(video)
        frame_path = os.path.join(compress_path,str(frame))
        os.makedirs(frame_path)
        final_mask = np.zeros((height,width)).astype(np.uint8)
        boxes_list = []
        human_instance_dict = {}
        vehicle_instance_dict = {}
        classes = predictions.pred_classes.cpu().numpy()
        scores = predictions.scores.cpu().numpy()
        
        instance_num = len(predictions)
        mask_all = predictions.pred_masks.cpu().numpy().astype(np.uint8)
        boxes_all = predictions.pred_boxes.tensor.cpu().numpy().astype(np.uint16)

        # scores低于给定值的去掉
        ignore_scores_low = np.where((scores < 0.30))
        # class不是车和人的都筛掉
        ignore_not_classes = np.where((classes != 0) & (classes != 2) & (classes != 5) & (classes != 7))
        
        ignore_scores_low = np.array(ignore_scores_low[0])
        ignore_not_classes = np.array(ignore_not_classes[0])

        temp_array = np.concatenate((ignore_scores_low, ignore_not_classes))
        ignore = np.unique(temp_array)
        # 筛选当前帧的实例
        classes, scores, mask_all, boxes_all, instance_num = ignore_instance_with_condition(ignore,
                                                                                                classes, scores,  
                                                                                                mask_all, boxes_all, 
                                                                                                instance_num)

        for j in range(instance_num):
            mask = mask_all[j]
            final_mask = np.bitwise_or(mask, final_mask)
            boxes_list.append(boxes_all[j])
        final_mask = final_mask * 255                 
        instance =cv2.add(vis_frame, np.zeros(np.shape(vis_frame), dtype=np.uint8), mask=final_mask)
        boxes_list_new = []
        for item in boxes_list:
            if item.tolist() not in boxes_list_new:
                boxes_list_new.append(item.tolist())
        boxes_list = boxes_list_new
        # 根据实例和class数组的对应关系，将实例按类别保存下来
        for k in range(len(boxes_list)):
            x1, y1, x2, y2 = boxes_list[k][0], boxes_list[k][1], boxes_list[k][2], boxes_list[k][3]
            # 如果前景面积过小则筛掉
            if (abs(x1 - x2) * abs(y1 - y2)) < 1024:
                continue
            instance_crop = instance[y1:y2, x1:x2]
            instance_crop = instance_crop[...,::-1]
            # 之前改过的一个bug，前景的size是0则去掉
            if instance_crop.size ==  0:
                continue
            if classes[k] == 0:
                human_num_detected += 1
                output_img_dir = compress_path+str(frame)+'/'+str(boxes_list[k][0])+'_'+str(boxes_list[k][1])+'_'+str(boxes_list[k][2])+'_'+str(boxes_list[k][3])+'.png'
                cv2.imwrite(output_img_dir,instance_crop,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                human_instance_dict[output_img_dir] = instance_crop
                person_track_info = str(frame)+','+'-1'+','+str(boxes_list[k][0])+','+str(boxes_list[k][1])+','+str(boxes_list[k][2])+','+str(boxes_list[k][3])+','+str(scores[k])+'\n'
                person_track_box_file.write(person_track_info)
                person_track_box_file.flush()
            else:
                vehicle_num_detected += 1
                output_img_dir = compress_path+str(frame)+'/'+str(boxes_list[k][0])+'_'+str(boxes_list[k][1])+'_'+str(boxes_list[k][2])+'_'+str(boxes_list[k][3])+'.jpg'
                cv2.imwrite(output_img_dir,instance_crop,[cv2.IMWRITE_JPEG_QUALITY,100])
                vehicle_instance_dict[output_img_dir] = instance_crop
                vehicle_track_info = str(frame)+','+'-1'+','+str(boxes_list[k][0])+','+str(boxes_list[k][1])+','+str(boxes_list[k][2])+','+str(boxes_list[k][3])+','+str(scores[k])+'\n'
                vehicle_track_box_file.write(vehicle_track_info)
                vehicle_track_box_file.flush()
        # 将实例的关键点保存下来
        if len(human_instance_dict) > 0:
            compute_instance_cord_batch(human_instance_dict,human_annotation_file,human_pose_model, target="human")
        if len(vehicle_instance_dict) > 0:
            compute_instance_cord_batch(vehicle_instance_dict,vehicle_annotation_file,vehicle_pose_model, target="vehicle")
        boxes_list.clear()
        frame += 1
        # 释放资源
        video.release()
        person_track_box_file.flush()
        vehicle_track_box_file.flush()
        person_track_box_file.close()
        vehicle_track_box_file.close()
        os.rename(os.path.join(compress_path,'annotation_vehicle.txt'), os.path.join(compress_path,'annotation_vehicle.csv'))
        os.rename(os.path.join(compress_path,'annotation_human.txt'), os.path.join(compress_path,'annotation_human.csv'))

        return compress_path


def fg_clean(video_path):
    # opencv读取视频对象
    video = cv2.VideoCapture(video_path)
    # 视频宽
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频高
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 视频文件名
    basename = os.path.basename(video_path)
    # 创建压缩目录
    compress_path = basename + '_comp/'
    if not os.path.exists(compress_path):
        raise RuntimeError("path don't exist" + str(compress_path))
    # 对行人和车辆目标进行追踪，超过1帧才追踪
    if os.path.getsize(os.path.join(compress_path,'person_box.txt'))!= 0:
        
        with open(os.path.join(compress_path,'person_box.txt'), 'r') as infile:
            data1 = infile.readlines()
        if len(data1) > 0:
            track_iou(compress_path,'person_box.txt','person_pair.csv','human')
    if os.path.getsize(os.path.join(compress_path,'vehicle_box.txt'))!= 0:
        
        with open(os.path.join(compress_path,'vehicle_box.txt'), 'r') as infile2:
            data2 = infile2.readlines()
        if len(data2) > 0:
            track_iou(compress_path,'vehicle_box.txt','vehicle_pair.csv','vehicle')
            
    # 清洗静态前景目标
    if os.path.exists(compress_path+'vehicle_pair.csv'):
        vehicle_clean(compress_path+'annotation_vehicle.csv',compress_path+'vehicle_pair.csv',(width,height))
        filter_key_vehicle(compress_path)
        if os.path.exists(compress_path+'vehicle_pair.csv'):
            del_imgs(compress_path,compress_path+'vehicle_pair.csv')
    if os.path.exists(compress_path+'person_pair.csv'):
        human_clean(compress_path+'annotation_human.csv',compress_path+'person_pair.csv',(width,height))
        filter_key_person(compress_path)
        if os.path.exists(compress_path+'person_pair.csv'):
            del_imgs(compress_path,compress_path+'person_pair.csv')

    comp_files(compress_path)
    tar_path = basename + '_comp.tar.xz'
    make_tarfile(tar_path, compress_path)
    return tar_path
    
