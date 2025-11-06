import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from colorama import Fore, Style
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
model = models.mobilenet_v3_small()

number_of_classes = 11
model.classifier[3] = nn.Linear(model.classifier[3].in_features, number_of_classes)


script_dir = os.path.dirname(os.path.abspath(__file__))
weight_file = "best.pth"
weight_path = os.path.join(script_dir, "weights", weight_file)  # 动态拼接路径

# 检查文件是否存在
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"权重文件 {weight_path} 不存在！")

# 加载权重（修复反斜杠和路径问题）
model.load_state_dict(
    torch.load(
        weight_path,
        map_location=torch.device('cpu'), 
        weights_only=True
    )
)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize([400, 400]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

def predict_image(image_path, class_names,threshold=0):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, 1)
        if max_prob.item() < threshold:
            return "未知类别"  # 或 "不认识"
        else:
            return class_names[predicted.item()]

test_data_dir = os.path.join(script_dir, 'example')
if not os.path.exists(test_data_dir):
    raise FileNotFoundError(f"Test data directory '{test_data_dir}' does not exist.")

class_names = sorted([d for d in os.listdir(test_data_dir) if d != "未知类别"])


test_data = {}
for class_name in class_names:
    class_dir = os.path.join(test_data_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, file_name)
            test_data[image_path] = class_name
# 添加对 Unknown 文件夹的处理
unknown_folder = os.path.join(test_data_dir, "未知类别")
if os.path.exists(unknown_folder):
    for file_name in os.listdir(unknown_folder):
        image_path = os.path.join(unknown_folder, file_name)
        test_data[image_path] = "未知类别"  # 这里将 Unknown 图片标记为 Unknown 类别

correct_predictions = 0
total_predictions = 0

for image_path, actual_class in test_data.items():
    predicted_class = predict_image(image_path, class_names)
    
    if actual_class == predicted_class:
        print(f"{Fore.GREEN}Actual: {actual_class}, Predicted: {predicted_class}{Style.RESET_ALL}")
        correct_predictions += 1
    else:
        print(f"{Fore.RED}Actual: {actual_class}, Predicted: {predicted_class}{Style.RESET_ALL}")
    
    total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {total_predictions - correct_predictions}")
print(f"Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2%})")