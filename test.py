# image_detect.py - 修复版
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
from database import init_database, save_record

print("=" * 60)
print("图片昆虫识别系统 - 框选并标注中文名称")
print("=" * 60)
# 初始化数据库
init_database()
# 检查模型
if not os.path.exists('insect_model.pth'):
    print("错误：找不到模型文件 insect_model.pth")
    print("请先运行 train.py 训练模型")
    input("按回车键退出...")
    exit()

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

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============ 中文字体设置 ============
def get_font(size=24):
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

font_large = get_font(32)
font_medium = get_font(24)
font_small = get_font(18)

def draw_chinese(img, text, position, color=(0, 255, 0), font=font_medium):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

# ============ 识别单张图片（简化版，更稳定）============
def recognize_single_image(img, model, transform, display_names):
    """识别单张图片，返回结果"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    insect_name = display_names[pred.item()]
    confidence = conf.item() * 100
    
    # 获取所有类别的概率
    all_probs = probs[0].numpy()
    all_results = []
    for i, idx in enumerate(np.argsort(all_probs)[::-1][:5]):
        all_results.append({
            'name': display_names[idx],
            'confidence': all_probs[idx] * 100
        })
    
    return insect_name, confidence, all_results

# ============ 创建保存文件夹 ============
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
result_folder = os.path.join(desktop, "识别结果")
os.makedirs(result_folder, exist_ok=True)

print(f"\n📁 识别结果将保存到: {result_folder}")

# ============ 选择图片 ============
print("\n请选择要识别的图片：")
image_path = input("图片路径（可直接拖拽）: ").strip().replace('"', '')

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"错误：文件不存在 - {image_path}")
    print("请检查文件路径是否正确")
    input("按回车键退出...")
    exit()

# 尝试用PIL读取图片（更稳定）
try:
    pil_img = Image.open(image_path)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    print(f"✅ 图片读取成功")
    print(f"   图片大小: {img.shape[1]}x{img.shape[0]}")
except Exception as e:
    print(f"错误：无法读取图片 - {e}")
    print("请确保图片格式为 JPG、PNG 或 JPEG")
    input("按回车键退出...")
    exit()

# ============ 识别 ============
print("\n🔍 识别中...")
insect_name, confidence, all_results = recognize_single_image(img, model, transform, display_names)

print(f"\n✅ 识别结果: {insect_name}")
print(f"📊 置信度: {confidence:.1f}%")

print(f"\n其他可能结果:")
for i, result in enumerate(all_results[1:], 1):
    print(f"   {i}. {result['name']}: {result['confidence']:.1f}%")

# ============ 绘制结果 ============
result_img = img.copy()
h, w = result_img.shape[:2]

# 在图片顶部画半透明背景
overlay = result_img.copy()
cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)

# 显示主要识别结果（大号字）
main_text = f"{insect_name} ({confidence:.1f}%)"
result_img = draw_chinese(result_img, main_text, (20, 50), (0, 255, 0), font_large)

# 显示其他可能结果
y_offset = 90
result_img = draw_chinese(result_img, "其他可能:", (20, y_offset), (255, 255, 0), font_small)
y_offset += 25

for i, result in enumerate(all_results[1:3]):  # 只显示前2个
    text = f"  {result['name']}: {result['confidence']:.1f}%"
    result_img = draw_chinese(result_img, text, (20, y_offset), (200, 200, 200), font_small)
    y_offset += 22

# 在图片中央画一个识别框（突出显示）
box_margin = 20
cv2.rectangle(result_img, (box_margin, box_margin), 
              (w - box_margin, h - box_margin), (0, 255, 0), 3)

# 添加水印
time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cv2.putText(result_img, time_str, (w - 180, h - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# ============ 保存结果 ============
filename = os.path.basename(image_path)
name, ext = os.path.splitext(filename)
if ext.lower() not in ['.jpg', '.jpeg', '.png']:
    ext = '.jpg'
output_filename = f"{name}_识别结果{ext}"
output_path = os.path.join(result_folder, output_filename)

# 确保输出文件名不重复
counter = 1
while os.path.exists(output_path):
    output_filename = f"{name}_识别结果_{counter}{ext}"
    output_path = os.path.join(result_folder, output_filename)
    counter += 1

cv2.imwrite(output_path, result_img)
print(f"\n📁 结果已保存: {output_path}")
# ==========================
# 保存到数据库
# ==========================
save_record(
    insect_name=insect_name,
    confidence=confidence,
    source="image",
    image_path=output_path
)

print("✅ 已保存到数据库")

# ============ 保存识别记录 ============
record_file = os.path.join(result_folder, "识别记录.txt")
with open(record_file, 'a', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"图片文件: {filename}\n")
    f.write(f"识别结果: {insect_name} (置信度: {confidence:.1f}%)\n")
    f.write(f"保存路径: {output_filename}\n")
    f.write("\n其他可能结果:\n")
    for i, result in enumerate(all_results[1:4], 1):
        f.write(f"   {i}. {result['name']}: {result['confidence']:.1f}%\n")
    f.write("=" * 50 + "\n\n")

print(f"📝 识别记录已保存: {record_file}")

# ============ 显示图片 ============
# 调整窗口大小以适应屏幕
screen_height = 720
if h > screen_height:
    scale = screen_height / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    display_img = cv2.resize(result_img, (new_w, new_h))
else:
    display_img = result_img

cv2.imshow('Insect identification results', display_img)
print("\n按任意键关闭图片窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n程序退出")