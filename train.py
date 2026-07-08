# train.py - 森林昆虫识别系统（优化版）
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import shutil
import time
import numpy as np
from collections import Counter

print("=" * 50)
print("森林昆虫识别系统 - 模型训练（优化版）")
print("=" * 50)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 数据路径：指向桌面的 insect_data_two（清洗后的数据）
# ============================================================
data_dir = r'C:\Users\ABC\Desktop\insect_data_two'

if not os.path.exists(data_dir):
    print(f"错误：找不到 {data_dir} 文件夹！")
    input("按回车键退出...")
    exit()

# 获取昆虫种类
insect_classes = sorted([d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d))])
print(f"\n数据路径: {data_dir}")
print(f"找到 {len(insect_classes)} 种昆虫:")

# ============================================================
# 自动检测数据结构：
# - 如果有 train/val 子文件夹 → 直接用
# - 如果图片直接平铺在类别文件夹下 → 自动按 80/20 拆分
# ============================================================
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 检查是否已有 train/val 结构
has_train_val = any(
    os.path.exists(os.path.join(data_dir, cls, 'train'))
    for cls in insect_classes
)

if has_train_val:
    # 已有 train/val 结构，直接用
    print("检测到 train/val 结构，直接使用")
    for i, cls in enumerate(insect_classes):
        train_path = os.path.join(data_dir, cls, 'train')
        val_path = os.path.join(data_dir, cls, 'val')
        train_count = len([f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTS)]) if os.path.exists(train_path) else 0
        val_count = len([f for f in os.listdir(val_path) if f.lower().endswith(IMG_EXTS)]) if os.path.exists(val_path) else 0
        print(f"  {i+1}. {cls} - 训练: {train_count}张, 验证: {val_count}张")
else:
    # 图片平铺，需要自动拆分
    print("检测到图片平铺结构，自动按 80/20 拆分 train/val")
    for i, cls in enumerate(insect_classes):
        cls_dir = os.path.join(data_dir, cls)
        all_images = [f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS)]
        random.seed(42)  # 固定随机种子，保证每次拆分一致
        random.shuffle(all_images)

        train_count = int(len(all_images) * 0.8)
        train_images = all_images[:train_count]
        val_images = all_images[train_count:]

        print(f"  {i+1}. {cls} - 总计: {len(all_images)}张 → 训练: {len(train_images)}张, 验证: {len(val_images)}张")

if len(insect_classes) == 0:
    print("错误：数据文件夹中没有找到昆虫子文件夹")
    input("按回车键退出...")
    exit()

# ============================================================
# 数据预处理：增强版
# ============================================================
print("\n准备数据中...")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 创建临时文件夹
temp_train_dir = 'temp_train'
temp_val_dir = 'temp_val'

if os.path.exists(temp_train_dir):
    shutil.rmtree(temp_train_dir)
if os.path.exists(temp_val_dir):
    shutil.rmtree(temp_val_dir)

os.makedirs(temp_train_dir, exist_ok=True)
os.makedirs(temp_val_dir, exist_ok=True)

for cls in insect_classes:
    if has_train_val:
        # 已有 train/val 结构
        src_train = os.path.join(data_dir, cls, 'train')
        dst_train = os.path.join(temp_train_dir, cls)
        if os.path.exists(src_train):
            shutil.copytree(src_train, dst_train)

        src_val = os.path.join(data_dir, cls, 'val')
        dst_val = os.path.join(temp_val_dir, cls)
        if os.path.exists(src_val):
            shutil.copytree(src_val, dst_val)
    else:
        # 平铺结构：按 80/20 拆分
        cls_dir = os.path.join(data_dir, cls)
        all_images = [f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS)]
        random.seed(42)
        random.shuffle(all_images)

        train_count = int(len(all_images) * 0.8)
        train_images = all_images[:train_count]
        val_images = all_images[train_count:]

        dst_train = os.path.join(temp_train_dir, cls)
        dst_val = os.path.join(temp_val_dir, cls)
        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val, exist_ok=True)

        for f in train_images:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(dst_train, f))
        for f in val_images:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(dst_val, f))

try:
    image_datasets = {
        'train': datasets.ImageFolder(temp_train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(temp_val_dir, data_transforms['val'])
    }

    batch_size = 16 if device.type == 'cuda' else 8

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"\n训练集大小: {dataset_sizes['train']} 张图片")
    print(f"验证集大小: {dataset_sizes['val']} 张图片")
    print(f"昆虫种类: {class_names}")

    # 类别不均衡检测和处理
    train_targets = [label for _, label in image_datasets['train'].samples]
    class_counts = Counter(train_targets)
    num_classes = len(class_names)
    class_counts_list = [class_counts.get(i, 0) for i in range(num_classes)]

    print("\n各类别训练样本数:")
    for i, name in enumerate(class_names):
        count = class_counts_list[i]
        print(f"  {name}: {count}张")

    max_count = max(class_counts_list)
    min_count = min(class_counts_list)
    if max_count > 0 and min_count > 0 and max_count / min_count > 2:
        print(f"\n⚠ 类别不均衡（最大/最小 = {max_count/min_count:.1f}倍）")
        print("  → 启用加权采样和加权损失函数")

        sample_weights = [1.0 / class_counts[t] for t in train_targets]
        sample_weights = torch.DoubleTensor(sample_weights)
        sample_weights = sample_weights / sample_weights.sum()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler, num_workers=0)

        class_weights = torch.FloatTensor([
            len(train_targets) / (num_classes * class_counts_list[i]) if class_counts_list[i] > 0 else 1.0
            for i in range(num_classes)
        ])
        class_weights = class_weights / class_weights.mean()
        use_weighted_loss = True
    else:
        print("\n✓ 类别分布较均衡")
        use_weighted_loss = False

except Exception as e:
    print(f"数据加载错误: {e}")
    shutil.rmtree(temp_train_dir, ignore_errors=True)
    shutil.rmtree(temp_val_dir, ignore_errors=True)
    input("按回车键退出...")
    exit()

# ============================================================
# 原代码问题 4（最关键）：全部参数冻结，只训练最后一层
# ResNet18 预训练在 ImageNet（猫狗汽车等），跟昆虫差距很大
# 冻结整个 backbone = 让模型用"看猫看狗"的特征来识别昆虫
# 解决方案：解冻最后几层（layer3 + layer4），让它们学习昆虫特征
# ============================================================
print("\n创建模型中...")

model = models.resnet18(pretrained=True)

# 解冻最后两个 block（layer3 + layer4）+ fc
# layer1 和 layer2 保持冻结（底层边缘/纹理特征通用，不需要重新学）
for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 统计可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total:.1%})")
print(f"解冻层: layer3, layer4, fc")
print(f"冻结层: conv1, layer1, layer2, bn")

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)
print(f"输出类别数: {len(class_names)}")

# ============================================================
# 原代码问题 5：没有学习率调度器
# 固定 lr=0.001，后期学习太粗导致精度上不去
# 解决方案：不同层用不同学习率 + CosineAnnealing 调度
# ============================================================
# 分层学习率：新层（fc）用大学习率，微调层用小学习率
optimizer = optim.AdamW([
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 3e-4},
], weight_decay=1e-4)

# 余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# 损失函数（支持加权）
if use_weighted_loss:
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("使用加权交叉熵损失函数")
else:
    criterion = nn.CrossEntropyLoss()

# ============================================================
# 原代码问题 6：只有 15 个 epoch，没有早停
# 解冻更多层后需要更多 epoch 收敛
# ============================================================
NUM_EPOCHS = 30
PATIENCE = 8  # 早停：连续8轮验证精度不提升就停止

print(f"\n训练配置:")
print(f"  Epochs: {NUM_EPOCHS}（早停 patience={PATIENCE}）")
print(f"  Batch size: {batch_size}")
print(f"  优化器: AdamW (分层学习率)")
print(f"  调度器: CosineAnnealingLR")


def train_model(model, criterion, optimizer, scheduler, num_epochs, patience):
    print("\n开始训练...")
    print("-" * 50)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    no_improve_count = 0

    for epoch in range(num_epochs):
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}  LR: {current_lr:.2e}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        # 梯度裁剪，防止解冻层梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'  {phase:5s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    no_improve_count = 0
                    print(f'  ✓ 新最佳！准确率: {epoch_acc:.4f}')
                else:
                    no_improve_count += 1

        # 更新学习率
        scheduler.step()

        # 早停检查
        if no_improve_count >= patience:
            print(f'\n早停触发：连续 {patience} 轮无提升，停止训练')
            break

        print()

    print(f'训练完成！最佳验证准确率: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, best_acc


# 开始训练
start_time = time.time()
model, best_acc = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, PATIENCE)
training_time = time.time() - start_time
print(f"训练耗时: {training_time:.1f} 秒")

# 保存模型（保存到桌面）
print("\n保存模型中...")
model_save_dir = r'C:\Users\ABC\Desktop'
model_path = os.path.join(model_save_dir, 'insect_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
}, model_path)
print(f"模型已保存到: {model_path}")

# 清理临时文件
print("\n清理临时文件...")
shutil.rmtree(temp_train_dir, ignore_errors=True)
shutil.rmtree(temp_val_dir, ignore_errors=True)
print("临时文件已清理")

print("\n" + "=" * 50)
print("训练完成！")
print(f"共训练了 {len(class_names)} 种昆虫")
print(f"最佳准确率: {best_acc:.2%}")
print(f"训练耗时: {training_time:.1f} 秒")
print("=" * 50)
