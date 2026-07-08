# camera-video.py - 10秒分析系统（优化版 - 提高识别率）
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from collections import Counter
import time
from datetime import datetime

print("=" * 60)
print("10秒分析系统 - 优化版（提高识别率）")
print("=" * 60)

# 检查模型
if not os.path.exists('insect_model.pth'):
    print("错误：找不到模型文件 insect_model.pth")
    print("请先运行 train.py 训练模型")
    input("按回车键退出...")
    exit()

# ============ 使用您指定的文件夹路径 ============
RESULT_FOLDER = r"C:\Users\ABC\Desktop\昆虫识别记录"

# 检查主文件夹
if not os.path.exists(RESULT_FOLDER):
    print(f"错误：找不到文件夹 {RESULT_FOLDER}")
    print("请确认您已经在桌面创建了【昆虫识别记录】文件夹")
    input("按回车键退出...")
    exit()

# 设置子文件夹
images_folder = os.path.join(RESULT_FOLDER, "截图")
records_folder = os.path.join(RESULT_FOLDER, "记录")

# 检查子文件夹
if not os.path.exists(images_folder):
    print(f"错误：找不到子文件夹 {images_folder}")
    print("请在【昆虫识别记录】文件夹内创建【截图】文件夹")
    input("按回车键退出...")
    exit()

if not os.path.exists(records_folder):
    print(f"错误：找不到子文件夹 {records_folder}")
    print("请在【昆虫识别记录】文件夹内创建【记录】文件夹")
    input("按回车键退出...")
    exit()

print(f"\n使用文件夹:")
print(f"   - 截图: {images_folder}")
print(f"   - 记录: {records_folder}")

# ============ 加载模型 ============
print("\n加载模型中...")
device = torch.device('cpu')
checkpoint = torch.load('insect_model.pth', map_location=device)

class_names = checkpoint['class_names']
display_names = []
for name in class_names:
    if '_' in name:
        display_names.append(name.split('_', 1)[-1])
    else:
        display_names.append(name)

print(f"模型加载成功！可识别 {len(class_names)} 种昆虫")
for i, name in enumerate(display_names):
    print(f"   {i+1}. {name}")

# 创建模型
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ============ 优化：增强版图像预处理 ============
# 基础预处理（与训练时一致）
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def enhance_image(img_bgr):
    """
    图像增强：提升昆虫特征清晰度
    - CLAHE 自适应直方图均衡化（增强对比度）
    - 锐化处理（提升边缘细节）
    - 降噪处理（减少摄像头噪点）
    """
    # 转换到 LAB 色彩空间做 CLAHE（只增强亮度通道，保留色彩）
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 轻微高斯降噪（去除摄像头噪点，但不模糊细节）
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 锐化：增强昆虫边缘纹理
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

    return enhanced


def classify_image(img_bgr):
    """
    对单张 BGR 图像进行分类，返回 (类别索引, 类别名, 置信度)
    """
    # 先增强图像
    enhanced = enhance_image(img_bgr)

    # 转 RGB 送入模型
    roi_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    img_tensor = base_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), display_names[pred.item()], conf.item() * 100


def classify_multiscale(img_bgr):
    """
    多尺度分类：在不同裁剪尺寸下分别识别，取最高置信度
    解决昆虫在画面中大小不一的问题
    """
    h, w = img_bgr.shape[:2]
    best_pred, best_name, best_conf = 0, "未知", 0

    # 生成多个裁剪区域：中心区、左上、右上、左下、右下
    crop_configs = []

    # 全图
    crop_configs.append((0, 0, w, h, 1.0))

    # 中心大区（70%）
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    crop_configs.append((margin_x, margin_y, w - margin_x, h - margin_y, 0.8))

    # 中心中区（50%）
    margin_x, margin_y = int(w * 0.25), int(h * 0.25)
    crop_configs.append((margin_x, margin_y, w - margin_x, h - margin_y, 0.6))

    # 九宫格区域（小区域检测远处小昆虫）
    for gy in range(3):
        for gx in range(3):
            x1 = int(w * gx / 3)
            y1 = int(h * gy / 3)
            x2 = int(w * (gx + 1) / 3)
            y2 = int(h * (gy + 1) / 3)
            crop_configs.append((x1, y1, x2, y2, 0.4))

    for x1, y1, x2, y2, weight in crop_configs:
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            continue

        try:
            pred, name, conf = classify_image(crop)
            # 加权置信度（中心区域权重更高）
            weighted_conf = conf * weight
            if weighted_conf > best_conf:
                best_pred = pred
                best_name = name
                best_conf = weighted_conf
                best_raw_conf = conf
        except:
            continue

    return best_pred, best_name, best_raw_conf if 'best_raw_conf' in dir() else best_conf


# ============ 运动检测 ============
def setup_motion_detector():
    """初始化背景减除器"""
    # 使用 MOG2 背景减除器（自适应学习，能适应光线变化）
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,       # 历史帧数，越大越稳定
        varThreshold=50,   # 方差阈值，越小越敏感
        detectShadows=True # 检测阴影（过滤阴影干扰）
    )
    return bg_subtractor


def detect_motion(bg_subtractor, frame):
    """
    检测画面中的运动区域
    返回：是否有运动、运动区域的边界框列表、运动面积
    """
    # 缩小画面做运动检测（提高速度）
    small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 背景减除
    fg_mask = bg_subtractor.apply(gray, learningRate=0.01)

    # 去除阴影（阴影像素值在127-128之间）
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作：去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 找轮廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    total_area = 0
    frame_area = 320 * 240

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤太小的区域（噪点）和太大的区域（整个画面变化）
        if 100 < area < frame_area * 0.6:
            total_area += area
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 坐标映射回原始分辨率
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            boxes.append((
                int(x * scale_x), int(y * scale_y),
                int((x + bw) * scale_x), int((y + bh) * scale_y),
                area
            ))

    motion_ratio = total_area / frame_area if frame_area > 0 else 0
    has_motion = len(boxes) > 0 and motion_ratio > 0.005

    return has_motion, boxes, motion_ratio


# ============ 中文字体设置 ============
def get_font(size=24):
    """获取中文字体"""
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',      # 黑体
        'C:/Windows/Fonts/msyh.ttc',        # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',      # 宋体
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

font_large = get_font(28)
font_medium = get_font(22)
font_small = get_font(16)

def draw_chinese(img, text, position, color=(0, 255, 0), font=font_medium):
    """在OpenCV图像上绘制中文"""
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img


# ============ 打开摄像头 ============
print("\n正在打开摄像头...")
cap = None
for camera_index in range(3):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"摄像头已打开 (索引: {camera_index})")
        break
    else:
        cap.release()

if cap is None or not cap.isOpened():
    print("错误：无法打开摄像头")
    print("请检查摄像头是否连接")
    input("按回车键退出...")
    exit()

# 优化：请求更高分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自动对焦
cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)  # 稍微提亮
cap.set(cv2.CAP_PROP_CONTRAST, 130)   # 提高对比度
cap.set(cv2.CAP_PROP_SATURATION, 120)  # 稍微提高饱和度

# 验证实际分辨率
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头分辨率: {actual_w}x{actual_h}")

print("摄像头已就绪")
print("\n优化内容:")
print("   ✓ 运动检测：只在画面有变化时识别，减少空帧")
print("   ✓ 图像增强：CLAHE对比度增强 + 锐化 + 降噪")
print("   ✓ 多尺度检测：全图+中心+九宫格区域，不遗漏小昆虫")
print("   ✓ 自适应ROI：追踪运动目标，不局限于中心区域")
print("   ✓ 摄像头参数优化：更高分辨率、提亮、增对比度")
print("   ✓ 降低置信度门槛：从50%降到30%，减少漏检")
print("   ✓ 结果投票：多帧投票取众数，提高稳定性")
print("\n控制: Q退出 | S手动识别 | M切换运动检测开/关")
print("-" * 60)

# ============ 参数 ============
ANALYSIS_DURATION = 10
MIN_CONFIDENCE = 30  # 降低置信度阈值（原50%），减少漏检
MOTION_COOLDOWN = 0.5  # 运动检测冷却时间（秒）

analysis_count = 0
insect_stats = {}

# 分析周期状态
current_results = []      # 存储所有有效识别结果
current_confidences = []  # 对应置信度
current_frames = []       # 对应帧
current_boxes = []        # 对应运动框
cycle_start = None
cycle_frame_count = 0
cycle_motion_count = 0    # 本周期运动检测次数
best_result = None
best_confidence = 0

frame_count = 0
is_analyzing = True
running = True

# 运动检测
bg_subtractor = setup_motion_detector()
motion_enabled = True  # 运动检测开关
last_motion_time = 0

# 暖机：让背景减除器学习初始背景
print("摄像头暖机中（3秒）...")
for _ in range(90):
    ret, frame = cap.read()
    if ret:
        bg_subtractor.apply(frame, learningRate=0.01)
print("暖机完成，开始检测！\n")


def save_recognition(insect_name, confidence, screenshot_path, is_manual=False):
    """保存识别记录"""
    record_file = os.path.join(records_folder, "识别记录.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "手动" if is_manual else "自动"

    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] [{mode}] {insect_name} (置信度: {confidence:.1f}%)\n")
        f.write(f"        截图: {screenshot_path}\n")


def get_voting_result(results, confidences, threshold=0):
    """
    投票机制：多帧识别取众数
    如果投票结果置信度均值 > threshold，则接受
    返回 (类别索引, 类别名, 平均置信度, 投票数)
    """
    if not results:
        return None, None, 0, 0

    # 统计每个类别的出现次数和平均置信度
    counter = Counter(results)
    most_common = counter.most_common()

    # 取出现次数最多的类别
    best_class, best_count = most_common[0]

    # 计算该类别的平均置信度
    class_confs = [c for r, c in zip(results, confidences) if r == best_class]
    avg_conf = np.mean(class_confs) if class_confs else 0

    # 投票比例
    vote_ratio = best_count / len(results)

    return best_class, display_names[best_class], avg_conf, vote_ratio


while running:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    frame_count += 1
    h, w = frame.shape[:2]
    display = frame.copy()

    # ============ 运动检测 ============
    has_motion = False
    motion_boxes = []
    motion_ratio = 0

    if motion_enabled:
        has_motion, motion_boxes, motion_ratio = detect_motion(bg_subtractor, frame)

        # 在画面上显示运动检测结果
        if has_motion and motion_boxes:
            for x1, y1, x2, y2, area in motion_boxes:
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 1)

            # 合并运动框为一个大的检测区域
            all_x1 = min(box[0] for box in motion_boxes)
            all_y1 = min(box[1] for box in motion_boxes)
            all_x2 = max(box[2] for box in motion_boxes)
            all_y2 = max(box[3] for box in motion_boxes)

            # 扩展一点边距
            pad = 20
            det_x1 = max(0, all_x1 - pad)
            det_y1 = max(0, all_y1 - pad)
            det_x2 = min(w, all_x2 + pad)
            det_y2 = min(h, all_y2 + pad)

            cv2.rectangle(display, (det_x1, det_y1), (det_x2, det_y2), (0, 255, 255), 2)
            display = draw_chinese(display, "运动目标", (det_x1, det_y1 - 22), (0, 255, 255), font_small)
        else:
            det_x1, det_y1, det_x2, det_y2 = 0, 0, w, h  # 无运动时用全图
    else:
        # 运动检测关闭时，用全图
        has_motion = True  # 始终识别
        det_x1, det_y1, det_x2, det_y2 = 0, 0, w, h

    # ============ 周期管理 ============
    current_time = time.time()

    if cycle_start is None:
        cycle_start = current_time
        current_results = []
        current_confidences = []
        current_frames = []
        current_boxes = []
        cycle_frame_count = 0
        cycle_motion_count = 0
        best_result = None
        best_confidence = 0
        is_analyzing = True
        print("\n" + "=" * 40)
        print("开始新的10秒分析周期...")

    elapsed = current_time - cycle_start
    remaining = max(0, ANALYSIS_DURATION - elapsed)

    # 显示倒计时
    display = draw_chinese(display, f"下次分析: {remaining:.1f}秒", (10, 30), (0, 255, 255), font_medium)

    if best_confidence > 0:
        result_text = f"当前最佳: {best_result} ({best_confidence:.1f}%)"
        display = draw_chinese(display, result_text, (10, 65), (0, 255, 0), font_medium)

    # ============ 识别逻辑 ============
    # 优化：每隔3帧检测一次（比原来更频繁），且只在有运动时识别
    should_analyze = (frame_count % 3 == 0)

    if is_analyzing and should_analyze and (has_motion or not motion_enabled):
        # 防止同一运动目标短时间内重复识别
        if current_time - last_motion_time > MOTION_COOLDOWN:
            last_motion_time = current_time
            cycle_motion_count += 1

            try:
                if motion_enabled and has_motion and motion_boxes:
                    # 有运动目标时：用运动区域做识别（多尺度）
                    # 合并区域
                    all_x1 = max(0, min(box[0] for box in motion_boxes) - 20)
                    all_y1 = max(0, min(box[1] for box in motion_boxes) - 20)
                    all_x2 = min(w, max(box[2] for box in motion_boxes) + 20)
                    all_y2 = min(h, max(box[3] for box in motion_boxes) + 20)

                    roi = frame[all_y1:all_y2, all_x1:all_x2]
                    if roi.size > 0 and roi.shape[0] > 32 and roi.shape[1] > 32:
                        pred, name, conf = classify_multiscale(roi)
                    else:
                        pred, name, conf = 0, "未知", 0
                else:
                    # 无明确运动目标：用全图做多尺度识别
                    pred, name, conf = classify_multiscale(frame)

                if conf >= MIN_CONFIDENCE:
                    current_results.append(pred)
                    current_confidences.append(conf)
                    current_frames.append(frame.copy())
                    current_boxes.append((det_x1, det_y1, det_x2, det_y2))
                    cycle_frame_count += 1

                    if conf > best_confidence:
                        best_confidence = conf
                        best_result = name

            except Exception as e:
                pass

    # ============ 周期结束 ============
    if elapsed >= ANALYSIS_DURATION:
        print(f"\n周期结束！识别次数: {cycle_frame_count}, 运动检测: {cycle_motion_count}")

        if current_results and current_confidences:
            # 优化：投票机制取众数，比单纯取最高置信度更稳定
            vote_class, vote_name, vote_conf, vote_ratio = get_voting_result(
                current_results, current_confidences
            )

            if vote_name and vote_conf >= MIN_CONFIDENCE:
                # 找到投票结果对应的最佳帧
                vote_class_indices = [i for i, r in enumerate(current_results) if r == vote_class]
                best_idx = vote_class_indices[np.argmax([current_confidences[i] for i in vote_class_indices])]
                best_frame = current_frames[best_idx]
                best_box = current_boxes[best_idx]

                print(f"投票结果: {vote_name} (平均置信度: {vote_conf:.1f}%, 投票: {vote_ratio:.0%})")

                # 保存截图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{vote_name}_{vote_conf:.0f}.jpg"
                filepath = os.path.join(images_folder, filename)

                annotated = best_frame.copy()
                bx1, by1, bx2, by2 = best_box
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                # 在截图上标注（用中文 PIL）
                annotated = draw_chinese(annotated, f"{vote_name} {vote_conf:.1f}%",
                                        (bx1, by1 - 30), (0, 255, 0), font_medium)
                cv2.imwrite(filepath, annotated)

                # 保存记录
                save_recognition(vote_name, vote_conf, filename, False)
                print(f"截图已保存: {filename}")

                # 更新统计
                analysis_count += 1
                insect_stats[vote_name] = insect_stats.get(vote_name, 0) + 1

                # 显示统计
                print(f"\n统计 (共{analysis_count}个周期):")
                for insect, count in sorted(insect_stats.items(), key=lambda x: x[1], reverse=True):
                    pct = count / analysis_count * 100
                    print(f"   {insect}: {count}次 ({pct:.1f}%)")
            else:
                print("投票结果置信度过低，本轮无有效识别")
        else:
            print("未检测到有效结果（无运动或置信度不足）")

        print("-" * 40)

        # 重置周期
        cycle_start = None
        best_result = None
        best_confidence = 0
        is_analyzing = False

    # ============ 界面显示 ============
    # 显示统计
    if analysis_count > 0:
        y_pos = h - 140
        display = draw_chinese(display, "统计:", (10, y_pos), (255, 255, 0), font_medium)
        y_pos += 30

        sorted_stats = sorted(insect_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (insect, count) in enumerate(sorted_stats[:4]):
            pct = count / analysis_count * 100
            bar = "█" * int(pct / 5)
            text = f"{insect}: {count}次 ({pct:.1f}%) {bar}"
            display = draw_chinese(display, text, (10, y_pos), (200, 200, 200), font_small)
            y_pos += 22

        total_text = f"总周期: {analysis_count}"
        display = draw_chinese(display, total_text, (10, y_pos + 5), (255, 255, 255), font_small)

    # 控制提示
    motion_status = "开" if motion_enabled else "关"
    display = draw_chinese(display, f"Q:退出|S:手动|M:运动({motion_status})",
                          (w - 210, 30), (200, 200, 200), font_small)

    # 运动状态
    motion_label = "运动:有" if (has_motion and motion_enabled) else ("运动:关" if not motion_enabled else "运动:无")
    motion_color = (0, 255, 0) if (has_motion and motion_enabled) else (0, 0, 255)
    display = draw_chinese(display, motion_label, (w - 100, h - 25), motion_color, font_small)

    # 分析状态
    if is_analyzing:
        status = "[分析中]"
        status_color = (0, 255, 0)
    else:
        status = "[等待]"
        status_color = (0, 0, 255)
    display = draw_chinese(display, status, (10, h - 25), status_color, font_small)

    # 帧计数
    display = draw_chinese(display, f"帧: {frame_count}", (10, 80), (150, 150, 150), font_small)

    # 显示画面
    cv2.imshow('video', display)

    # ============ 按键处理 ============
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n退出程序...")
        running = False
        break

    elif key == ord('s') or key == ord('S'):
        print("\n手动识别...")
        # 手动识别：用全图多尺度
        pred, name, conf = classify_multiscale(frame)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_{timestamp}_{name}_{conf:.0f}.jpg"
        filepath = os.path.join(images_folder, filename)

        annotated = frame.copy()
        cv2.rectangle(annotated, (det_x1, det_y1), (det_x2, det_y2), (0, 255, 0), 3)
        annotated = draw_chinese(annotated, f"{name} {conf:.1f}%",
                                (det_x1, det_y1 - 30), (0, 255, 0), font_medium)
        cv2.imwrite(filepath, annotated)

        save_recognition(name, conf, filename, True)
        print(f"结果: {name} ({conf:.1f}%)")
        print(f"截图已保存: {filename}")

        analysis_count += 1
        insect_stats[name] = insect_stats.get(name, 0) + 1

    elif key == ord('m') or key == ord('M'):
        motion_enabled = not motion_enabled
        status = "开启" if motion_enabled else "关闭"
        print(f"\n运动检测已{status}")

# ============ 最终统计 ============
print("\n" + "=" * 60)
print("最终统计结果")
print("=" * 60)

if analysis_count > 0:
    print(f"\n总分析周期: {analysis_count}")
    print("\n各昆虫出现次数:")

    sorted_stats = sorted(insect_stats.items(), key=lambda x: x[1], reverse=True)
    for insect, count in sorted_stats:
        pct = count / analysis_count * 100
        bar = "#" * int(pct / 2)
        print(f"   {insect}: {count}次 ({pct:.1f}%) {bar}")

    # 保存报告
    report_file = os.path.join(records_folder, f"统计报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("昆虫识别统计报告（优化版）\n")
        f.write("=" * 60 + "\n")
        f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总分析周期: {analysis_count}\n\n")

        for insect, count in sorted_stats:
            pct = count / analysis_count * 100
            f.write(f"{insect}: {count}次 ({pct:.1f}%)\n")

        if sorted_stats:
            f.write(f"\n最常见昆虫: {sorted_stats[0][0]}\n")

    print(f"\n报告已保存: {report_file}")

print(f"\n截图文件夹: {images_folder}")
print(f"记录文件夹: {records_folder}")
print("=" * 60)

cap.release()
cv2.destroyAllWindows()
print("\n程序已退出")
