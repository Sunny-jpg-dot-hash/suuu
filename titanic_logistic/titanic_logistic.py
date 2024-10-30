import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
test_data = pd.read_csv(r"C:\Users\user\Desktop\titanic_data\test.csv")
train_data = pd.read_csv(r"C:\Users\user\Desktop\titanic_data\train.csv")

# 数据预览
print("Test Data Head:")
print(test_data.head())
print("\nTrain Data Head:")
print(train_data.head())

# 合并数据集
total_data = pd.concat([train_data, test_data], ignore_index=True)

# 数据处理和特征工程
labelencoder = LabelEncoder()
total_data['Sex'] = labelencoder.fit_transform(total_data['Sex'])
total_data['Embarked'] = total_data['Embarked'].fillna('S')
total_data['Embarked'] = labelencoder.fit_transform(total_data['Embarked'])
total_data['Fare'] = total_data['Fare'].fillna(total_data['Fare'].median())
total_data['Age'] = total_data['Age'].fillna(total_data['Age'].median())

# 准备训练和测试数据
train_x = total_data[total_data['Survived'].notnull()][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
train_y = total_data[total_data['Survived'].notnull()]['Survived'].values
test_x = total_data[total_data['Survived'].isnull()][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values

# 数据标准化
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# 转换为张量
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)
test_x = torch.tensor(test_x, dtype=torch.float32)

# 划分训练和验证集
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 自定义数据集类
class TitanicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

train_set = TitanicDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(train_x.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
best_acc = 0

for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算训练集和验证集的准确率
    with torch.no_grad():
        train_pred = model(train_x).round()
        train_acc = (train_pred == train_y).sum().item() / len(train_y)

        valid_pred = model(valid_x).round()
        valid_acc = (valid_pred == valid_y).sum().item() / len(valid_y)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), "model.pth")

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")

# 加载最佳模型
best_model = Model()
best_model.load_state_dict(torch.load("model.pth"))

# 使用最佳模型生成预测
with torch.no_grad():
    test_pred = best_model(test_x).round().view(-1).numpy().astype(int)
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_pred})
    submission.to_csv(r"C:\Users\user\Desktop\IT_submission.csv", index=False)
    print("Submission file created: IT_submission.csv")



