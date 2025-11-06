# 野生食用菌智能分类系统

基于深度学习的云南省野生食用菌智能分类项目，采用知识蒸馏技术实现高精度轻量化模型部署。

## 演示效果

![野生食用菌分类演示](ezgif-2c671108347e4c.gif)

*将训练好的模型由PyTorch转换成ONNX框架支持的文件，利用开源Demo完成移动端部署*

## 特点

- 🍄 识别11类常见野生食用菌
- 📱 支持Android移动端实时分类
- ⚡ 轻量化模型，推理速度<18ms
- 🎯 验证准确率达94.64%

## 支持的菌类

羊肚菌、黑牛肝菌、红牛肝菌、美味牛肝菌、黄皮疣柄牛肝菌、鸡枞菌、青头菌、奶浆菌、干巴菌、竹荪、松茸

## 技术栈

- **框架**: PyTorch 1.13.1 + CUDA 11.6
- **教师模型**: ResNet50 (94.45%)
- **学生模型**: MobileNetV3-small (94.64%)
- **部署**: ONNX Runtime Mobile

## 模型性能

| 模型 | 精度 | 参数量 | 备注 |
|------|------|--------|------|
| ResNet50 | 94.45% | ~25M | 教师模型 |
| MobileNetV3 (原始) | 92.82% | ~5.5M | 学生模型 |
| MobileNetV3 (蒸馏) | **94.64%** | ~5.5M | 最优模型 |

## 快速开始
```bash
# 安装依赖
pip install torch torchvision onnx onnxruntime opencv-python

# 训练教师模型
python train_teacher.py --model resnet50

# 知识蒸馏
python distillation.py --teacher resnet50 --student mobilenetv3

# 转换ONNX
python convert_onnx.py --model_path checkpoints/best.pth
```

## 数据集

- 总计: 6000+ 张图像
- 来源: Kaggle开源数据 + 网络爬虫自建
- 划分: 训练集:验证集:测试集 = 7:2:1
