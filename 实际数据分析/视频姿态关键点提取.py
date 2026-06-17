"""
a.py — 步态视频姿态关键点提取与角度序列导出脚本
================================================
本脚本使用 torchvision 的 Keypoint R-CNN 从步态视频中提取 COCO 17 个人体关键点，
逐帧保存关键点坐标、绘制带骨架的视频，并计算每个关键点相对于髋部参考方向的摆动角度。

主要流程
--------
1. load_pose_model      : 加载预训练 Keypoint R-CNN。
2. collect_all_videos   : 收集待处理视频路径。
3. read_video_frames    : 读取视频帧。
4. process_frames       : 逐帧检测关键点并计算角度序列。
5. save_process_frames  : 保存角度 CSV、关键点坐标 CSV 和标注后的视频。

说明
----
该脚本偏向批处理实验数据，目录和输出路径在 main() 中配置。若迁移到新机器，
优先检查 main() 内的数据目录、输出目录和设备配置。
"""

# 加载姿态估计模型
from json.decoder import JSONArray
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy
import torch
import torchvision
# 读取视频并提取帧
import cv2
# 避免 OpenMP 冲突
import os
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import time
from tqdm import tqdm
# 释放cuda缓存
import gc



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_pose_model():
    """
    加载预训练 Keypoint R-CNN 人体关键点检测模型。

    返回
    ----
    model : torch.nn.Module
        已加载 COCO 关键点权重并切换到 eval 模式的模型。输出包含 boxes、scores、
        labels 和 keypoints，其中 keypoints 的标准形状为 (num_person, 17, 3)。
    """
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model



def read_video_frames(video_path):
    """
    读取视频文件并按原始顺序提取所有帧。

    参数
    ----
    video_path : str
        视频文件路径，通常为 front.mp4、left.mp4 或 right.mp4。

    返回
    ----
    frames : list[np.ndarray]
        OpenCV BGR 格式图像列表，每个元素形状为 (H, W, 3)。若视频无法打开或无帧，
        返回空列表。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# image = frames[0]
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(8,8))
# plt.imshow(image_rgb)
# plt.axis('off')  # 隐藏坐标轴
# plt.show()


# 处理单帧图像，获取2D关键点
import numpy as np
from torchvision import transforms

def process_frame(model, frame, device, video_path):
    """
    对单帧图像运行关键点检测，返回置信度最高人体的 17 个关键点。

    参数
    ----
    model : torch.nn.Module
        Keypoint R-CNN 模型，需处于 eval 模式。
    frame : np.ndarray, shape (H, W, 3)
        OpenCV 读取的 BGR 图像。transforms.ToTensor 会转换为 (C, H, W) 并归一化到 [0, 1]。
    device : str 或 torch.device
        推理设备，例如 "cpu" 或 "cuda"。
    video_path : str
        当前视频路径；保留用于按视角裁剪的兼容逻辑。

    返回
    ----
    keypoints : np.ndarray 或 None, shape (17, 3)
        每行是 (x, y, confidence)。若模型未检测到人体，返回 None。
    """
    transform_fn = transforms.Compose([
        transforms.ToTensor()
    ])
    frame_copy = frame.copy()
    # if os.path.basename(video_path)[:-4] == 'front':
    #     left_cut = 750 # 650 # 650
    #     right_cut = 1200 # 1250 # 1250
    # elif os.path.basename(video_path)[:-4] == 'left':
    #     left_cut = 650 # 650
    #     right_cut = 1100 # 1250
    # elif os.path.basename(video_path)[:-4] == 'right':
    #     left_cut = 750
    #     right_cut = 1250
    #
    # frame_index = list(x for x in range(frame_copy.shape[1]))
    # frame_index = frame_index[:left_cut] + frame_index[right_cut:]
    # frame_copy[:, frame_index, :] = 0

    image = transform_fn(frame_copy).to(device) ###############
    with torch.no_grad():
        outputs = model([image])[0]
    if len(outputs['keypoints']) == 0:
        return None
    keypoints = outputs['keypoints'][0].cpu().numpy()
    return keypoints



# def process_frame2(model, frames):
#     '''
#     对单帧图像进行处理，获取人体关键点的2D坐标。
#     参数：
#         model（torch.nn.Module）:姿态估计模型。
#         frames（list）:所有帧图像组成的列表，列表中的每个元素是一个ndarray:
#         H*W*C，导入torch模型需要用transforms.ToTensor 转换成 C*H*W
#     返回：
#         keypoints(list): 所有关键点的2D坐标组成的列表，列表中的每个元素是一个ndarray: 形状为（17,3）
#     '''
#     transform_fn = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     image = transform_fn(torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float())
#     with torch.no_grad():
#         outputs = model([image])[0]
#     if len(outputs['keypoints']) == 0:
#         return None
#     keypoints = outputs['keypoints'][0].cpu().numpy()
#     return keypoints


# 可视化关键点函数
import matplotlib.pyplot as plt
# Keypoint R-CNN 的标准骨架连接顺序（COCO17）
COCO_SKELETON = [
    (5, 7), (7, 9),     # 左臂
    (6, 8), (8, 10),    # 右臂
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16), # 右腿
    (5, 6),             # 肩
    (11, 12),           # 髋
    (5, 11), (6, 12),   # 躯干连接
    (0, 1), (1, 2), (2, 3), (3, 4)  # 面部
]

def plot_keypoints(image, keypoints, show=True, save_path=None):
    '''
    在原图上绘制关键点和骨架连线。
    参数:
        image (ndarray): 原图，BGR格式（opencv读取的图像）
        keypoints (ndarray): 关键点（17，3），包含（x,y,confidence）
        show (bool): 是否显示图像
        save_path (str): 如设置则保存该路径
    '''
    # left_cut = 750
    # right_cut = 1250
    # image_index = list(x for x in range(image.shape[1]))
    # image_index = image_index[:left_cut]+image_index[right_cut:]
    # image[:, image_index, :] = 0

    # 转化为RGB显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,8))
    plt.imshow(image_rgb)
    ax = plt.gca()
    # 绘制关键点
    for i, (x, y, c) in enumerate(keypoints):
        if c > 0.2: # 阈值过滤
            ax.plot(x, y, 'r*', markersize=3)
            ax.text(x + 2, y - 2, str(i), fontsize=0.5, color='blue')
    # 绘制骨架连接线
    for i, j in COCO_SKELETON:
        if keypoints[i, 2] > 0.2 and keypoints[j, 2] > 0.2:
            x1, y1 = keypoints[i, 0], keypoints[i, 1]
            x2, y2 = keypoints[j, 0], keypoints[j, 1]
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=1.5)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# 可视化关键点函数
def draw_keypoints_on_frame(frame, keypoints, threshold=0.2):
    '''
    使用OpenCV直接在原图上绘制关键点和骨架线，frame将被就地修改。
    参数：
        frame: 原始图像，BGR格式（cv2读入的图）
        keypoints: np.ndarray, (17, 3)，包含(x, y, confidence)
        threshold: 置信度阈值，过滤低质量关键点
    '''
    for i, (x, y, c) in enumerate(keypoints):
        if c > threshold:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # 红点
            cv2.putText(frame, str(i), (int(x)+2, int(y)-2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (255, 0, 0), 1)

    for i, j in COCO_SKELETON:
        if keypoints[i, 2] > threshold and keypoints[j, 2] > threshold:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)  # 绿色骨架线



# 得到可视化关键点标记后的每一帧的图像列表
def process_all_frames_and_draw(model, frames, device, file_info):
    """
    批量检测视频帧并生成带关键点/骨架标注的帧列表。

    参数
    ----
    model : torch.nn.Module
        Keypoint R-CNN 模型。
    frames : list[np.ndarray]
        原始视频帧，OpenCV BGR 格式。
    device : str 或 torch.device
        模型推理设备。
    file_info : list[dict]
        与 frames 对齐的文件信息；每项至少包含 'file' 键，用于传递给 process_frame。

    返回
    ----
    processed_frames : list[np.ndarray]
        绘制了关键点和骨架线的帧。原始 frames 不会被就地修改。
    """
    processed_frames = []
    for idx, frame in enumerate(frames):
        video_path = file_info[idx]['file']
        keypoints = process_frame(model, frame, device, video_path)
        frame_copy = frame.copy()
        if keypoints is not None:
            draw_keypoints_on_frame(frame_copy, keypoints)
        processed_frames.append(frame_copy)
    return processed_frames


# 将关键点标记后的图像列表保存为视频
def save_frames_to_video(frames, output_path, fps=60):
    """
    将帧列表写出为 MP4 视频。

    参数
    ----
    frames : list[np.ndarray]
        待写出的视频帧，要求所有帧大小一致，BGR 格式。
    output_path : str
        输出 MP4 文件路径；父目录需已存在。
    fps : int, 默认 60
        输出视频帧率。

    返回
    ----
    None
        函数通过 cv2.VideoWriter 写文件；当 frames 为空时仅打印提示并返回。
    """
    if len(frames) == 0:
        print("帧列表为空，无法保存视频。")
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'mp4v'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"视频已保存到：{output_path}")



# 获取所有数据集文件夹
def collect_all_videos(dataset_dirs, pattern='*.mp4'):
    """
    从一个或多个目录模式中收集步态视频路径。

    参数
    ----
    dataset_dirs : list[str]
        目录或 glob 模式列表。每个模式会先展开为目录，再在目录内搜索视频文件。
    pattern : str, 默认 '*.mp4'
        视频文件匹配模式。

    返回
    ----
    file_info : list[dict] 或 None
        每个元素形如 {'file': video_path}。未找到视频时返回 None。
    """
    file_info = [] #存储文件信息，用于后续处理
    for dataset_dir in dataset_dirs:
        print(f'从数据集收集数据: {dataset_dir}')
        video_dir = dataset_dir  # 不再假设有 Video_Record 子文件夹
        video_dirs = glob.glob(video_dir)
        for video_dir_path in video_dirs:
            video_files = glob.glob(os.path.join(video_dir_path, pattern))
            for video_file in video_files:
                print(f"收集文件信息: {video_file}")
                file_info.append({'file': video_file})
    if not file_info:
        print(f"没有找到有效数据")
        return None, None
    print(f"总共收集到 {len(file_info)} 个视频文件")
    return file_info


#     dataset_dirs = glob.glob('/root/autodl-tmp/DATA/sp/B*')
# # file_info = collect_all_videos(dataset_dirs)
# # file_path = file_info[0]['file']
# # print(file_path)
#     output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
#     output_dir_2 = "/root/autodl-tmp/DATA/results/CSV_Output/2025-05-20"
#     output_dir_3 = "/root/autodl-tmp/DATA/results/Video_Output/2025-05-20"

def save_process_frames(file_path, output_dir_1, output_dir_2, output_dir_3, KEYPOINTS_all_points, KEYPOINTS_every_points,
                        KEYPOINT_EVERY_NAMES, Angel, processed_frames, idx):
    """
    保存单个视频的角度序列、关键点坐标序列和带标注视频。

    参数
    ----
    file_path : str
        当前处理的视频路径。文件名不含扩展名需为 front、left 或 right，用于判断视角。
    output_dir_1 : str
        摆动角度 CSV 输出目录。
    output_dir_2 : str
        关键点坐标 CSV 输出目录。
    output_dir_3 : str
        绘制关键点后的视频输出目录。
    KEYPOINTS_all_points : np.ndarray
        所有关键点跨帧拼接后的坐标与置信度数组。
    KEYPOINTS_every_points : dict[str, np.ndarray]
        每个关键点单独的时间序列，键名为 point_0 到 point_16。
    KEYPOINT_EVERY_NAMES : dict[str, str]
        关键点编号到中文身体部位名称的映射。
    Angel : dict[str, np.ndarray]
        每个关键点相对于髋部参考方向的摆动角度序列。
    processed_frames : list[np.ndarray]
        已绘制关键点的视频帧。
    idx : int
        当前视频在批处理列表中的序号，仅用于日志输出。
    """

    # 创建结果目录
    if os.path.basename(file_path)[:-4] == 'front':
        face_which = '冠'
        # 保存摆动角度csv文件
        # output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_1 = output_dir_1
        os.makedirs(result_dir_1, exist_ok=True)
        # output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        # result_dir_1 = os.path.join(output_dir_1, relative_path)
        # os.makedirs(result_dir_1, exist_ok=True)
        for _, j in enumerate(KEYPOINT_EVERY_NAMES):
            result_file = os.path.join(result_dir_1, f"{face_which}-{KEYPOINT_EVERY_NAMES[j]}摆动角度.csv")
            result_df = pd.DataFrame({
                f"{j}": Angel[j]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存{KEYPOINT_EVERY_NAMES[j]}冠状面摆动角度csv文件到: {result_file}")
        # 保存位置序列csv文件
        # output_dir_2 = "/root/autodl-tmp/DATA/results/CSV_Output/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_2 = output_dir_2
        os.makedirs(result_dir_2, exist_ok=True)
        for i, j in enumerate(KEYPOINT_EVERY_NAMES):
            result_file = os.path.join(result_dir_2, f"pose_front_front_{KEYPOINT_EVERY_NAMES[j]}.csv")
            result_df = pd.DataFrame({
                f"帧数索引": KEYPOINTS_every_points[f"point_{i}"][:, 0],
                f"point_id": KEYPOINTS_every_points[f"point_{i}"][:, 1],
                f"x": KEYPOINTS_every_points[f"point_{i}"][:, 2],
                f"y": KEYPOINTS_every_points[f"point_{i}"][:, 3],
                f"score": KEYPOINTS_every_points[f"point_{i}"][:, 4]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存冠状面{KEYPOINT_EVERY_NAMES[j]}位置坐标序列csv文件到: {result_file}")
            result_file = os.path.join(result_dir_2, f"pose_front_front.csv")
            result_df = pd.DataFrame({
                f"point_id": KEYPOINTS_all_points[:, 0],
                f"x": KEYPOINTS_all_points[:, 1],
                f"y": KEYPOINTS_all_points[:, 2],
                f"score": KEYPOINTS_all_points[:, 3]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存冠状面所有关键点位置坐标序列csv文件到: {result_file}")
        # 保存视频的地址
        # output_dir_3 = "/root/autodl-tmp/DATA/results/Video_Output/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_3 = output_dir_3
        os.makedirs(result_dir_3, exist_ok=True)
        video_file = os.path.join(result_dir_3, f"pose_front_front.mp4")
        save_frames_to_video(processed_frames, video_file, fps=60)
        print(f"已保存冠状面步态视频")
    elif os.path.basename(file_path)[:-4] == 'left':
        face_which = '矢左'
        # 保存摆动角度csv文件
        # output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_1 = output_dir_1
        os.makedirs(result_dir_1, exist_ok=True)
        # output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        # result_dir_1 = output_dir_1
        # os.makedirs(result_dir_1, exist_ok=True)
        for _, j in enumerate(KEYPOINT_EVERY_NAMES):
            result_file = os.path.join(result_dir_1, f"{face_which}-{KEYPOINT_EVERY_NAMES[j]}摆动角度.csv")
            result_df = pd.DataFrame({
                f"{j}": Angel[j]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存矢状面左{KEYPOINT_EVERY_NAMES[j]}摆动角度csv文件到: {result_file}")
        # 保存位置序列csv文件
        # output_dir_2 = "/root/autodl-tmp/DATA/results/CSV_Output/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_2 = output_dir_2
        os.makedirs(result_dir_2, exist_ok=True)
        for i, j in enumerate(KEYPOINT_EVERY_NAMES):
            result_file = os.path.join(result_dir_2, f"pose_left_left_{KEYPOINT_EVERY_NAMES[j]}.csv")
            result_df = pd.DataFrame({
                f"帧数索引": KEYPOINTS_every_points[f"point_{i}"][:, 0],
                f"point_id": KEYPOINTS_every_points[f"point_{i}"][:, 1],
                f"x": KEYPOINTS_every_points[f"point_{i}"][:, 2],
                f"y": KEYPOINTS_every_points[f"point_{i}"][:, 3],
                f"score": KEYPOINTS_every_points[f"point_{i}"][:, 4]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存矢状面左{KEYPOINT_EVERY_NAMES[j]}位置坐标序列csv文件到: {result_file}")
            result_file = os.path.join(result_dir_2, f"pose_left_left.csv")
            result_df = pd.DataFrame({
                f"point_id": KEYPOINTS_all_points[:, 0],
                f"x": KEYPOINTS_all_points[:, 1],
                f"y": KEYPOINTS_all_points[:, 2],
                f"score": KEYPOINTS_all_points[:, 3]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存矢状面左所有关键点位置坐标序列csv文件到: {result_file}")
        # 保存视频的地址
        # output_dir_3 = "/root/autodl-tmp/DATA/results/Video_Output/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_3 = output_dir_3
        os.makedirs(result_dir_3, exist_ok=True)
        video_file = os.path.join(result_dir_3, f"pose_left_left.mp4")
        save_frames_to_video(processed_frames, video_file, fps=60)
        print(f"已保存矢状面左步态视频")
    elif os.path.basename(file_path)[:-4] == 'right':
        face_which = '矢右'
        # 保存摆动角度csv文件
        # output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_1 = output_dir_1
        os.makedirs(result_dir_1, exist_ok=True)
        # output_dir_1 = "/root/autodl-tmp/DATA/results/Analysis_CSV/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        # result_dir_1 = os.path.join(output_dir_1, relative_path)
        # os.makedirs(result_dir_1, exist_ok=True)
        for _, j in enumerate(KEYPOINT_EVERY_NAMES):
            result_file = os.path.join(result_dir_1, f"{face_which}-{KEYPOINT_EVERY_NAMES[j]}摆动角度.csv")
            result_df = pd.DataFrame({
                f"{j}": Angel[j]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存矢状面右{KEYPOINT_EVERY_NAMES[j]}摆动角度csv文件到: {result_file}")
        # 保存位置序列csv文件
        # output_dir_2 = "/root/autodl-tmp/DATA/results/CSV_Output/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_2 = output_dir_2
        os.makedirs(result_dir_2, exist_ok=True)
        for i, j in enumerate(KEYPOINT_EVERY_NAMES):
            result_file = os.path.join(result_dir_2, f"pose_right_right_{KEYPOINT_EVERY_NAMES[j]}.csv")
            result_df = pd.DataFrame({
                f"帧数索引": KEYPOINTS_every_points[f"point_{i}"][:, 0],
                f"point_id": KEYPOINTS_every_points[f"point_{i}"][:, 1],
                f"x": KEYPOINTS_every_points[f"point_{i}"][:, 2],
                f"y": KEYPOINTS_every_points[f"point_{i}"][:, 3],
                f"score": KEYPOINTS_every_points[f"point_{i}"][:, 4]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存矢状面右{KEYPOINT_EVERY_NAMES[j]}位置坐标序列csv文件到: {result_file}")
            result_file = os.path.join(result_dir_2, f"pose_right_right.csv")
            result_df = pd.DataFrame({
                f"point_id": KEYPOINTS_all_points[:, 0],
                f"x": KEYPOINTS_all_points[:, 1],
                f"y": KEYPOINTS_all_points[:, 2],
                f"score": KEYPOINTS_all_points[:, 3]
            })
            result_df.to_csv(result_file, index=False)
            #print(f"已保存矢状面右所有关键点位置坐标序列csv文件到: {result_file}")
        # 保存视频的地址
        # output_dir_3 = "/root/autodl-tmp/DATA/results/Video_Output/2025-05-20"
        # relative_path = os.path.relpath(os.path.dirname(file_path), "/root/autodl-tmp/DATA/sp")
        result_dir_3 = output_dir_3
        os.makedirs(result_dir_3, exist_ok=True)
        video_file = os.path.join(result_dir_3, f"pose_right_right.mp4")
        save_frames_to_video(processed_frames, video_file, fps=60)
        print(f"已保存矢状面右步态视频")
    print(f"第{idx}个视频的相关输出结果保存完成！")

















import numpy as np
def process_frames(model, frames, idx, device, video_path):
    """
    逐帧提取关键点，并计算每个关键点的摆动角度时间序列。

    参数
    ----
    model : torch.nn.Module
        Keypoint R-CNN 模型。
    frames : list[np.ndarray]
        单个视频的所有帧，OpenCV BGR 格式。
    idx : int
        当前视频/个体序号，仅用于日志和调试。
    device : str 或 torch.device
        推理设备。
    video_path : str
        当前视频路径，传给 process_frame 以保留视角相关兼容逻辑。

    返回
    ----
    KEYPOINTS_all_points : np.ndarray
        所有帧、所有关键点拼接后的数组；列含 point_id、x、y、score。
    KEYPOINTS_every_points : dict[str, np.ndarray]
        每个关键点单独的跨帧时间序列，键为 point_0 ... point_16。
    KEYPOINT_EVERY_NAMES : dict[str, str]
        point_i 到中文关键点名称的映射。
    Angel : dict[str, np.ndarray]
        每个关键点相对初始髋部参考方向的角度序列，单位为弧度。
    processed_frames : list[np.ndarray]
        绘制关键点和骨架后的帧列表。
    """
    KEYPOINT_NAMES = [
        '鼻子', '左眼', '右眼', '左耳', '右耳',
        '左肩', '右肩', '左肘', '右肘',
        '左腕', '右腕', '左髋', '右髋',
        '左膝', '右膝', '左踝', '右踝'
    ]
    row, col = process_frame(model, frames[0], device, video_path).shape
    name = np.array([i for i in range(row)])
    KEYPOINTS_all_points = process_frame(model, frames[0], device, video_path)
    KEYPOINTS_all_points = np.column_stack((name, KEYPOINTS_all_points))
    KEYPOINTS_every_points = {}
    KEYPOINT_EVERY_NAMES = {}
    Angel = {}
    #################
    processed_frames = []
    frame = frames[0]
    frame_copy = frame.copy()
    if KEYPOINTS_all_points[:, 1:] is not None:
        draw_keypoints_on_frame(frame_copy, KEYPOINTS_all_points[:, 1:])
    processed_frames.append(frame_copy)

    for i in range(row):
        key = f"point_{i}"
        KEYPOINTS_every_points[key] = np.array(np.hstack((0 + 1, KEYPOINTS_all_points[i, :])))  # 第一个分量是帧数的指标
        KEYPOINT_EVERY_NAMES[key] = KEYPOINT_NAMES[i]
    point_root = 0.5 * (KEYPOINTS_every_points['point_11'][2:4] + KEYPOINTS_every_points['point_12'][2:4])
    hip_vec = KEYPOINTS_every_points['point_11'][2:4] - KEYPOINTS_every_points['point_12'][2:4]
    ref_vec = np.array([-hip_vec[1], hip_vec[0]])  # 垂直方向（向下）
    for i in range(row):
        key = f"point_{i}"
        # Angel[key] = np.arccos(np.dot(KEYPOINTS_every_points[key][2:4], point_root)
        #                        / (np.linalg.norm(KEYPOINTS_every_points[key][2:4]) * np.linalg.norm(point_root)))
        target_vec = KEYPOINTS_every_points[key][2:4] - point_root
        cross = np.cross(np.append(target_vec,0), np.append(ref_vec,0))[-1]
        dot = np.dot(target_vec, ref_vec)
        # Angel[key] = np.arctan(np.cross(np.append(KEYPOINTS_every_points[key][2:4],0), np.append(point_root,0))[-1]
        #                        / np.dot(KEYPOINTS_every_points[key][2:4], point_root))
        Angel[key] = np.arctan2(cross, dot)
    pbar = tqdm(frames[1:], desc=f"视频 {idx} 帧处理中", ncols=100)
    for i, frame in enumerate(pbar):
        if i > 0:
            keypoints = process_frame(model, frame, device, video_path)
            for j in range(np.shape(keypoints)[0]):
                KEYPOINTS_every_points[f"point_{j}"] = np.vstack(
                    (KEYPOINTS_every_points[f"point_{j}"], np.hstack((np.array([i + 1, j]), keypoints[j, :])))
                )
                # Angel[f"point_{j}"] = np.hstack(
                #     (Angel[f"point_{j}"], np.arccos(np.dot(KEYPOINTS_every_points[f"point_{j}"][i, 2:4], point_root)
                #                                     / (np.linalg.norm(
                #         KEYPOINTS_every_points[f"point_{j}"][i, 2:4]) * np.linalg.norm(point_root))))
                # )
                target_vec = KEYPOINTS_every_points[f"point_{j}"][i, 2:4] - point_root
                cross = np.cross(np.append(target_vec,0), np.append(ref_vec,0))[-1]
                dot = np.dot(target_vec, ref_vec)
                # Angel[f"point_{j}"] = np.hstack(
                #     (Angel[f"point_{j}"], np.arctan(np.cross(np.append(KEYPOINTS_every_points[f"point_{j}"][i, 2:4],0), np.append(point_root,0))[-1]
                #                                     / np.dot(
                #         KEYPOINTS_every_points[f"point_{j}"][i, 2:4], point_root)  )))
                Angel[f"point_{j}"] = np.hstack(
                    (Angel[f"point_{j}"], np.arctan2(cross, dot) ))

            KEYPOINTS_all_points = np.vstack((KEYPOINTS_all_points, np.column_stack((name, keypoints))))
            ########################
            frame_copy = frame.copy()
            if keypoints is not None:
                draw_keypoints_on_frame(frame_copy, keypoints)
            processed_frames.append(frame_copy)
    return (KEYPOINTS_all_points, KEYPOINTS_every_points,
            KEYPOINT_EVERY_NAMES, Angel, processed_frames)

# import glob
# # 获取所有数据集文件夹
# dataset_dirs = glob.glob('/root/autodl-tmp/DATA/sp/B*')
#    dataset_dirs = glob.glob('/sp/B*')

#%%
def main():
    """
    批处理入口：从默认步态视频目录读取视频并导出关键点分析结果。

    执行步骤
    ----
    1. 解析命令行参数并初始化计时器；
    2. 固定 PyTorch/NumPy 随机种子；
    3. 选择 CUDA 或 CPU 设备并加载 Keypoint R-CNN；
    4. 在 base_path/sp/B* 下收集视频；
    5. 对视频逐帧提取关键点、计算摆动角度；
    6. 保存角度 CSV、关键点坐标 CSV 和标注后视频；
    7. 清理缓存并输出运行时间。

    返回
    ----
    None
        结果写入 output_dir_1/output_dir_2/output_dir_3 指定的目录。
    """
    # 解析命令行参数
    parser =argparse.ArgumentParser(description='使用KeypointR-CNN模型提取特征')
    # parser.add_argument('--input_dim', type=int, default=512, help='')
    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()

    # 计算运行时间
    start_time = time.time()  # 记录开始时间

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载预训练的 Keypoint R-CNN 模型，用于人体关键点检测。
    model = load_pose_model().to(device)
    print(f"已加载预训练的 Keypoint R-CNN 模型，用于人体关键点检测。")


    # 获取所有数据集文件夹。公开代码不包含真实步态视频，请在本机设置
    # GAIT_VIDEO_BASE_DIR，指向实际视频数据根目录。
    base_path = os.environ.get("GAIT_VIDEO_BASE_DIR", "").strip()
    if not base_path:
        raise RuntimeError(
            "请先设置环境变量 GAIT_VIDEO_BASE_DIR。真实步态视频数据不随公开代码发布。"
        )
    dataset_dirs = glob.glob(os.path.join(base_path, 'sp/B*'))

    # 收集所有数据所在的位置信息
    file_info = collect_all_videos(dataset_dirs)

    # 输出文件夹名称
    output_dir_1 = os.path.join(base_path, "results/Analysis_CSV/2025-05-23")
    output_dir_2 = os.path.join(base_path, "results/CSV_Output/2025-05-23")
    output_dir_3 = os.path.join(base_path, "results/Video_Output/2025-05-23")
    from tqdm import tqdm
    #for i in tqdm(range(len(file_info)), desc="📦 视频总进度", ncols=100):
    for i in range(1, 2):
        idx = i + 1
        #file_path = file_info[idx]['file']
        video_path = file_info[i]['file']
        frames = read_video_frames(video_path)
        # 使用预训练模型处理第idx个视频，得到关键点空间坐标序列、计算关键点摆动角度，并在视频上标注关键点
        KEYPOINTS_all_points, KEYPOINTS_every_points, KEYPOINT_EVERY_NAMES, Angel, processed_frames\
            = process_frames(model, frames, idx, device, video_path)
        # 保存第idx个样本视频的各项输出结果
        save_process_frames(video_path, output_dir_1, output_dir_2, output_dir_3, KEYPOINTS_all_points,
                            KEYPOINTS_every_points, KEYPOINT_EVERY_NAMES, Angel, processed_frames, idx)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    end_time = time.time()  # 记录结束时间
    print(f'处理完成，运行时间为:{end_time-start_time}秒')









if __name__ == "__main__":
    main()
