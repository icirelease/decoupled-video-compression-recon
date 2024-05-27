# 说明
代码持续更新中......
## 目录结构
``` sh
    background 背景处理相关代码
    foreground 前景处理相关代码
    manager 实例分割、关键点检测等算法代码
    pipeline 视频编解码的流程
    utils 公共代码
    comp_video.py 视频压缩入口
    decomp_video.py 视频解码入口
    requirements.txt 环境配置
```

## 运行指令
``` sh
    // 视频压缩
    python comp_video.py -video_path video_path
    // 视频解码
    python decomp_video.py -video_path video_path -compress_path // 视频压缩文件地址
```
