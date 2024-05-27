import os
import shutil
import tarfile

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:xz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def copy_decomp_video(compress_path, out_folder):
    # 创建目标文件夹，如果不存在  
    if not os.path.exists(out_folder):  
        os.makedirs(out_folder)
    decomp_video_path = os.path.splitext(compress_path)[0]+'_decomp.mp4'

    # 复制文件  
    shutil.copy(decomp_video_path, os.path.join(out_folder, os.path.basename(decomp_video_path)))