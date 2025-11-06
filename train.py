import math
import torch.utils.model_zoo as model_zoo
import timm
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import time
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import alexnet

"""
蘑菇分类模型训练脚本
支持知识蒸馏、多种模型架构和CBAM注意力机制
"""

# ================================
# 导入必要的库
# ================================
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import models as model_zoo
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# 可选导入
# import timm

# ================================
# 全局配置
# ================================
# 处理截断图像
Image.MAX_IMAGE_PIXELS = None
ImageFile = Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 全局标志
saved = False

# ================================
# 超参数配置
# ================================
# 数据预处理参数（ImageNet 标准参数,与预训练模型匹配)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 训练参数
BATCH_SIZE = 32
IMAGE_SIZE = 400
NUM_CLASSES = 11
EPOCHS = 300
LEARNING_RATE = 1e-4  # 或 3e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.003
PRETRAINED_WEIGHTS = 'Yes'

# 知识蒸馏参数
TEMPERATURE = 2.0
ALPHA = 0.5

# 数据路径
TRAIN_DATA_PATH = 'Mushrooms'
TEST_DATA_PATH = 'Test'
WEIGHTS_SAVE_PATH = 'weights'

# 早停参数
EARLY_STOPPING_PATIENCE = 10

# 模型URL
MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam']


# ================================
# 模型架构定义
# ================================

# ---------- CBAM注意力机制 ----------
class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ---------- ResNet + CBAM 基础模块 ----------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock with CBAM"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck with CBAM"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------- ResNet with CBAM ----------
class ResNet(nn.Module):
    """ResNet backbone with CBAM attention"""

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ---------- 模型构建函数 ----------
def resnet18_cbam(pretrained=True, **kwargs):
    """
    构建 ResNet-18 + CBAM 模型
    Args:
        pretrained (bool): 是否使用 ImageNet 预训练权重
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(MODEL_URLS['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=True, **kwargs):
    """
    构建 ResNet-34 + CBAM 模型
    Args:
        pretrained (bool): 是否使用 ImageNet 预训练权重
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(MODEL_URLS['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=True, **kwargs):
    """
    构建 ResNet-50 + CBAM 模型
    Args:
        pretrained (bool): 是否使用 ImageNet 预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(MODEL_URLS['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


# ================================
# 损失函数定义
# ================================
class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    结合 KL 散度损失（蒸馏损失）和交叉熵损失（监督学习）
    """

    def __init__(self, temperature=3.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # KL 散度损失（蒸馏损失）
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # 交叉熵损失（监督学习）
        ce_loss = self.ce_loss(student_logits, labels)

        # 总损失
        return self.alpha * distillation_loss + (1 - self.alpha) * ce_loss


# ================================
# 工具类和辅助函数
# ================================
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


def pil_loader(path):
    """自定义图像加载器,处理损坏的图像"""
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
            return img.convert('RGB')
        except OSError as e:
            print(f"Error loading image at {path}: {e}")
            return None


def set_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


def get_mean_and_std(loader):
    """
    计算数据集的均值和标准差
    每次更改图像大小或训练变换时使用
    """
    mean = 0
    std = 0
    total_images_count = 0

    for images, _ in loader:
        image_count_in_batch = images.size(0)
        images = images.view(image_count_in_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std


# ================================
# 数据加载
# ================================
def get_data_loaders(batch_size, image_size, mean, std,
                     train_path, test_path, use_augmentation=False):
    """
    创建训练和测试数据加载器
    Args:
        batch_size: 批次大小
        image_size: 图像尺寸
        mean: 归一化均值
        std: 归一化标准差
        train_path: 训练数据路径
        test_path: 测试数据路径
        use_augmentation: 是否使用数据增强
    """
    # 覆盖默认的图像加载器
    torchvision.datasets.folder.pil_loader = pil_loader

    # 训练数据变换
    if use_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.RandomRotation(15),  # 随机旋转 ±15°
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 随机裁剪后resize
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std))
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std))
        ])

    # 测试数据变换
    test_transforms = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

    # 创建数据集
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transforms)

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ================================
# 训练和评估函数
# ================================
def evaluate_model_on_test_set(model, test_loader, threshold=0):
    """
    在测试集上评估模型
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        threshold: 概率阈值(仅当最大概率大于阈值时才认为预测有效)
    Returns:
        epoch_acc: 测试准确率
    """
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    start_time = time.time()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)  # 转换为概率分布
            max_probs, predicted = torch.max(probabilities, 1)  # 获取最大概率及其索引

            # 仅当最大概率大于阈值时才认为预测有效
            valid_predictions = (max_probs > threshold) & (predicted == labels)
            predicted_correctly_on_epoch += valid_predictions.sum().item()

    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    end_time = time.time()
    evaluation_duration = end_time - start_time

    print(f'- Testing Data: Got {predicted_correctly_on_epoch} out of {total} images. Accuracy: {epoch_acc:.2f}%')
    print(f'Evaluation completed in {evaluation_duration:.2f} seconds')

    return epoch_acc


def train_nn(model, train_loader, test_loader, criterion, optimizer, epochs,
             lr, momentum, weight_decay, weights, batch_size, image_size):
    """
    标准神经网络训练函数
    Args:
        model: 待训练模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        其他参数用于保存模型文件名
    """
    global saved
    device = set_device()

    for epoch in range(epochs):
        start_time = time.time()
        print(f'Starting epoch: {epoch + 1}')
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        # 使用 tqdm 显示进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for data in train_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                total += labels.size(0)

                optimizer.zero_grad()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_correct += (labels == predicted).sum().item()

                # 更新进度条
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.00 * running_correct / total

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f'- Training Data: Got {running_correct} out of {total} images. '
              f'Accuracy: {epoch_accuracy:.2f}% Loss: {epoch_loss:.4f}')
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds')

        # 测试集评估
        test_acc = evaluate_model_on_test_set(model, test_loader)

        # 如果训练和测试准确率都达到97%,保存模型并停止训练
        if test_acc >= 97.00 and epoch_accuracy >= 97.00:
            saved = True
            save_path = (f'{WEIGHTS_SAVE_PATH}/model_weights(lr={lr},mom={momentum},'
                         f'wd={weight_decay},pretr={weights},bs={batch_size},'
                         f'ep={epochs},size={image_size},trainacc={epoch_accuracy:.2f},'
                         f'testacc={test_acc:.2f}).pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            print("Test accuracy reached 97%. Stopping training.")
            break

    print("Finished")
    return model


def train_with_distillation(student_model, teacher_model, train_loader, test_loader,
                            optimizer, epochs, temperature, alpha):
    """
    使用知识蒸馏训练学生模型
    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        optimizer: 优化器
        epochs: 训练轮数
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
    """
    device = set_device()
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        # 使用 tqdm 显示进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # 获取教师模型的预测
                with torch.no_grad():
                    teacher_logits = teacher_model(images)

                # 获取学生模型的预测
                student_logits = student_model(images)

                # 计算蒸馏损失
                loss = criterion(student_logits, teacher_logits, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)

        # 计算测试集准确率
        test_acc = evaluate_model_on_test_set(student_model, test_loader)

        # 打印并记录日志
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


# ================================
# 模型初始化函数
# ================================
def initialize_model(model_name='resnet50', num_classes=11, pretrained=True):
    """
    初始化模型
    Args:
        model_name: 模型名称
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    Returns:
        model: 初始化后的模型
    """
    if model_name == 'resnet18_cbam':
        # ResNet18 + CBAM  终端1
        model = resnet18_cbam(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet18':
        # ResNet18  终端2
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet50':
        # ResNet50  终端3
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'alexnet':
        # AlexNet  终端4
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'mobilenet_v3_small':
        # MobileNetV3-Small  终端5
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == 'densenet121':
        # DenseNet121
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # elif model_name == 'vit_tiny':
    #     # Vision Transformer (ViT-Tiny)
    #     model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


# ================================
# 主程序
# ================================
def main():
    """主训练流程"""
    global saved

    # 创建权重保存目录
    os.makedirs(WEIGHTS_SAVE_PATH, exist_ok=True)

    # 设置设备
    device = set_device()
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        mean=MEAN,
        std=STD,
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        use_augmentation=False  # 终端2删掉数据增强
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # 每次更改图像大小或类/方法时使用此函数计算均值和标准差
    # print(get_mean_and_std(train_loader))

    # ========================================
    # 方案1: 标准训练
    # ========================================
    # print("\n=== Standard Training ===")
    # model = initialize_model(model_name='resnet50', num_classes=NUM_CLASSES, pretrained=True)
    # model = model.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
    #                             momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #
    # model = train_nn(model, train_loader, test_loader, criterion, optimizer, EPOCHS,
    #                  LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, PRETRAINED_WEIGHTS,
    #                  BATCH_SIZE, IMAGE_SIZE)

    # ========================================
    # 方案2: 知识蒸馏训练
    # ========================================
    print("\n=== Knowledge Distillation Training ===")

    # 加载教师模型（ResNet50）  终端2
    print("Loading teacher model...")
    teacher_model = models.resnet50(pretrained=True)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, NUM_CLASSES)
    teacher_model.load_state_dict(torch.load(
        'weights/resnet50_model_weights(lr=0.01,mom=0.9,wd=0.003,pretr=Yes,bs=32,ep=100,size=400).pth'
    ))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 设置为评估模式
    print("Teacher model loaded successfully")

    # 加载学生模型（MobileNetV3-Small）
    print("Initializing student model...")
    student_model = models.mobilenet_v3_small(pretrained=True)
    student_model.classifier[3] = nn.Linear(student_model.classifier[3].in_features, NUM_CLASSES)
    student_model = student_model.to(device)
    print("Student model initialized")

    # 优化器
    optimizer = torch.optim.Adam(student_model.parameters(), LEARNING_RATE)

    # 知识蒸馏训练
    # 终端3 新权重; 终端4再调整temp1.0; 终端5回调temp到2
    train_with_distillation(
        student_model, teacher_model, train_loader, test_loader,
        optimizer, EPOCHS, temperature=TEMPERATURE, alpha=ALPHA
    )

    # 保存最终模型
    if not saved:
        save_path = (f'{WEIGHTS_SAVE_PATH}/model_weights(lr={LEARNING_RATE},'
                     f'mom={MOMENTUM},wd={WEIGHT_DECAY},pretr={PRETRAINED_WEIGHTS},'
                     f'bs={BATCH_SIZE},ep={EPOCHS},size={IMAGE_SIZE}).pth')
        torch.save(student_model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()