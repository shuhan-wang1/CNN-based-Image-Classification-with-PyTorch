import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# CNN模型定义
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 数据增强和预处理
def get_transform(img_size=64):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# 自定义类标签映射
def custom_class_label_transform(class_name):
    if class_name.isdigit():  
        return class_name  
    elif len(class_name) == 1 and class_name.islower():  
        return class_name  
    elif len(class_name) == 2 and class_name.islower():  
        return class_name.upper()  
    else:
        raise ValueError(f"Unexpected class name: {class_name}")

# 自定义 ImageFolder 类来处理分类文件夹名
class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()  
        class_to_idx = {custom_class_label_transform(c): i for i, c in enumerate(classes)}
        print(f"Class to index mapping: {class_to_idx}")
        return classes, class_to_idx

# 加载自定义数据集（合并两个数据集）
def load_custom_data(custom_data_dir, custom_data_dir_1, batch_size=128):
    custom_transform = get_transform()

    # 加载第一个数据集
    custom_train = CustomImageFolder(root=os.path.join(custom_data_dir, 'train'), transform=custom_transform)
    custom_test = CustomImageFolder(root=os.path.join(custom_data_dir, 'test'), transform=custom_transform)

    # 加载第二个数据集
    custom_train_1 = CustomImageFolder(root=os.path.join(custom_data_dir_1, 'train_1'), transform=custom_transform)
    custom_test_1 = CustomImageFolder(root=os.path.join(custom_data_dir_1, 'test_1'), transform=custom_transform)

    # 合并训练和测试数据集
    combined_train_dataset = ConcatDataset([custom_train, custom_train_1])
    combined_test_dataset = ConcatDataset([custom_test, custom_test_1])

    # 创建DataLoader
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, custom_train.class_to_idx

# 初始化GradScaler
scaler = GradScaler()

# 绘制混淆矩阵
def plot_confusion_matrix(true_labels, pred_labels, class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# 验证模型并展示部分预测结果
def validate_model_and_show_images(model, val_loader, criterion, class_to_idx):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    val_loss /= len(val_loader)
    return val_loss, all_preds, all_labels

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, discrepancy, epochs=20, class_to_idx=None):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        val_loss, val_preds, val_labels = validate_model_and_show_images(model, val_loader, criterion, class_to_idx)
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')
        # 绘制混淆矩阵
        #plot_confusion_matrix(val_labels, val_preds, class_to_idx)

        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")

        scheduler.step(val_loss)

        if early_stopping.best_loss is None or (early_stopping.best_loss - val_loss) > discrepancy:
            early_stopping(val_loss, model)
        else:
            print(f"Validation loss did not decrease by more than {discrepancy}. Model not saved.")

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    

# 测试模型
def test_model(model, test_loader, class_to_idx):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    # 绘制测试集的混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, class_to_idx)

    return accuracy

class EarlyStopping:
    def __init__(self, patience=10, delta=0, save_path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.save_path)

# 主函数
if __name__ == '__main__':
    custom_data_dir = r"C:\Users\jackw\OneDrive - The University of Manchester\桌面\CNN_file\CNN_mixed\data"
    custom_data_dir_1 = r"C:\Users\jackw\OneDrive - The University of Manchester\桌面\CNN_file\CNN_mixed\data"

    train_loader, test_loader, class_to_idx = load_custom_data(custom_data_dir, custom_data_dir_1)

    num_classes = 62  
    model = CNNModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3)
    early_stopping = EarlyStopping(patience=10)

    discrepancy = 0.001  

    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))

        print("Validating loaded model...")
        initial_val_loss, _, _ = validate_model_and_show_images(model, test_loader, criterion, class_to_idx)
        print(f"Initial Validation Loss from loaded model: {initial_val_loss:.4f}")
        early_stopping.best_loss = initial_val_loss  

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, early_stopping, discrepancy, epochs=1000, class_to_idx=class_to_idx)

    print("Testing model after training...")
    final_accuracy = test_model(model, test_loader, class_to_idx)
    print(f"Final Accuracy after training: {final_accuracy:.2f}%")
