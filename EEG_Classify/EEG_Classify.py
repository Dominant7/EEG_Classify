import numpy
import EA
import CSP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from scipy.fft import fft
import random
import csv

FILTER_COUNT = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED) 
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

'''
CSP选择的滤波器数量的二分之一
'''

'''
trainDataDictxxx的结构如下
字典    |   key     value
        |  文件名    元组   |    data_X                    data_y
        |                   |  (200,59,300) nd.array    (200) nd.array
        |                   |  (trail*channel*sample)
        |                   |
'''
# 读取训练数据
trainDataDictRaw = {} # 未经对齐的训练集数据
trainDataDictEA = {} # 经过EA对齐的训练集数据
dataList = []
fileNames = ['s1.npz', 's2.npz', 's3.npz', 's4.npz']
for fileName in fileNames:
    with numpy.load('./Data\\train\\' + fileName) as data:
        data_X = data['X']
        data_y = data['y']
        # print(data_X.shape)
        # print(data_y.shape)
        dataList.append((data_X, data_y))
# print(trainDataDict)
trainDataDictRaw.update(dict(zip(fileNames, dataList)))


# Xmean = []
# for key in trainDataDictRaw:
#     Xmean.append(numpy.mean(numpy.mean(trainDataDictRaw[key][0], axis=2), axis=0))
# print(Xmean)

# 进行欧拉对齐
dataList = []
for key in trainDataDictRaw:
    data_X, data_y = trainDataDictRaw[key]
    data_X = EA.EA(data_X)
    dataList.append((data_X, data_y))
trainDataDictEA.update(dict(zip(fileNames, dataList)))

# 进行CSP滤波
labels = set()
for value in trainDataDictEA.values():
    labels.update(y for y in value[1])

CSPFilteredData = []
filtersList = []
for subject in trainDataDictEA:
    trainDataLabelDict = {} # 通过字典，将每个trail按照y标签分类
    for trail in range(0, len(trainDataDictEA[subject][1])): # 取y的值作为键
        labelKey = trainDataDictEA[subject][1][trail]
        if labelKey not in trainDataLabelDict:
            trainDataLabelDict[labelKey] = [trainDataDictEA[subject][0][trail]] # 若键(标签)不存在则创建一个列表作为值
        else:
            trainDataLabelDict[labelKey].append(trainDataDictEA[subject][0][trail]) # 若存在则向列表加入
    filters = CSP.CSP(trainDataLabelDict) # 可取W前n行与X相乘，则将数据降维，类似PCA
    filtersList.append(filters)
    for key in trainDataDictEA:
        data_X, data_y = trainDataDictEA[key]
        if len(trainDataLabelDict) == 2:
            data_X = numpy.einsum('ij,kjl->kil', numpy.concatenate((filters[0][:FILTER_COUNT], filters[0][-FILTER_COUNT:]), axis=0), data_X)
        else:
            for aFilter in filters:
                pass
        CSPFilteredData.append((data_X, data_y))

trainDataDictCSP = dict(zip(fileNames, CSPFilteredData))
'''
'''

'''
# 保留序列使用LSTM

# 分类器设计
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# 加载数据
data_X = []
data_y = []
for subject in trainDataDictCSP:
    data_X.append(trainDataDictCSP[subject][0])
    data_y.append(trainDataDictCSP[subject][1])

# 将列表中的数组连接起来
data_X = numpy.concatenate([data for data in data_X], axis=0)
data_y = numpy.concatenate([data for data in data_y], axis=0)

train_data = torch.Tensor(data_X)
train_labels = torch.LongTensor(data_y)

# 合并数据和标签为 TensorDataset
dataset = TensorDataset(train_data, train_labels)

# 随机划分数据集
split_ratio = 0.9
split_sizes = [int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio)]
train_dataset, test_dataset = random_split(dataset, split_sizes)

# 然后可以根据 train_dataset 和 test_dataset 构建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型

# 超参数
input_size = 300 # 采样点数
hidden_size = 16 # 隐藏层维度
num_layers = 1
num_classes = 2
learning_rate = 0.0001
num_epochs = 200
fc_hidden_sizes = [5, 10]

# 定义模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, fc_hidden_sizes, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 添加多个全连接层和Dropout层
        self.fc_layers = nn.ModuleList()
        prev_size = hidden_size
        for fc_size in fc_hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_layers.append(nn.ReLU())  # 添加激活函数
            self.fc_layers.append(nn.Dropout(dropout_prob))  # 添加Dropout层
            prev_size = fc_size
        
        self.fc_layers.append(nn.Linear(prev_size, num_classes))
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # 展平 LSTM 输出的最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过多个全连接层传递
        for layer in self.fc_layers:
            out = layer(out)
        
        return out

# 设置设备为CUDA（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型并移动到设备
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes, fc_hidden_sizes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Acc: {test_acc:.4f}')

# 最后可以保存模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'lstm_classifier_checkpoint.pth')
'''
# 提取log特征
''
# 加载数据
data_X = []
data_y = []
for subject in trainDataDictCSP:
    data_X.append(trainDataDictCSP[subject][0])
    data_y.append(trainDataDictCSP[subject][1])

# 将列表中的数组连接起来
data_X = numpy.concatenate([data for data in data_X], axis=0)
data_y = numpy.concatenate([data for data in data_y], axis=0)

logFeature = numpy.zeros(data_X.shape[:2])
i = 0
for trail in data_X:
    logFeature[i] = numpy.log10(numpy.diag(trail @ trail.T) / numpy.trace(trail @ trail.T))
    i += 1
    

'''
# 分类器

# 加载数据
data_X = []
data_y = []
for subject in trainDataDictCSP:
    data_X.append(trainDataDictCSP[subject][0])
    data_y.append(trainDataDictCSP[subject][1])

# 将列表中的数组连接起来
data_X = numpy.concatenate([data for data in data_X], axis=0)
data_y = numpy.concatenate([data for data in data_y], axis=0)

logFeature = numpy.zeros(data_X.shape[:2])
i = 0
for trail in data_X:
    logFeature[i] = numpy.log10(numpy.diag(trail @ trail.T) / numpy.trace(trail @ trail.T))
    i += 1

# 超参数
input_size = 2 * FILTER_COUNT  # 输入特征维度，根据数据集的特征数确定
hidden_size = [16, 32]  # 隐藏层大小
output_size = 1  # 输出大小，二分类任务只需一个输出
dropout_prob = 0.5 # dropout概率
learning_rate = 0.001 # 学习率
num_epochs = 4000

# 转换数据为 Tensor，并将标签转换为浮点数类型
train_data = torch.Tensor(logFeature)
train_labels = torch.FloatTensor(data_y)  # 将 LongTensor 转换为 FloatTensor

dataset = TensorDataset(train_data, train_labels)

# 随机划分训练集和验证集
split_ratio = 0.8
split_sizes = [int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio)]
train_dataset, val_dataset = random_split(dataset, split_sizes)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 定义全连接神经网络类
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # 构建隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        out = self.fc_out(x)
        out = self.sigmoid(out)
        return out

# 实例化模型和定义损失函数与优化器
model = NeuralNet(input_size, hidden_size, output_size, dropout_prob)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_preds = []
    train_targets = []
    for i, (inputs, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(inputs)
        # 将标签转换为 FloatTensor
        loss = criterion(outputs, labels.unsqueeze(1).float())  # labels需要添加一个维度以匹配输出

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练集准确率
        preds = (outputs > 0.5).float()  # 将输出概率转换为类别预测
        train_preds.extend(preds.squeeze().tolist())
        train_targets.extend(labels.tolist())

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}'
                  .format(epoch+1, num_epochs, loss.item()))

    train_acc = accuracy_score(train_targets, train_preds)

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels.unsqueeze(1).float()).item()
            preds = (outputs > 0.5).float()  # 将输出概率转换为类别预测
            val_preds.extend(preds.squeeze().tolist())
            val_targets.extend(labels.tolist())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)
    print('Epoch [{}/{}], Train Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'
          .format(epoch+1, num_epochs, train_acc, val_loss, val_acc))

torch.save(model, 'model\\FCmodel.mdl')

'''

# SVM实现

data_X = logFeature
data_y = data_y

# 划分训练集和测试集
#X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, test_size=0.2)
X_train, y_train = data_X, data_y

svm_model = SVC(kernel='linear', C=1.0)

# 训练模型
svm_model.fit(X_train, y_train)

# 在验证集上进行预测
TRAIL_NUM = 200
X_val = [data_X[i * TRAIL_NUM:(i + 1) * TRAIL_NUM] for i in range(len(fileNames))]
y_val = [data_y[i * TRAIL_NUM:(i + 1) * TRAIL_NUM] for i in range(len(fileNames))]
for i in range(len(fileNames)):
    val_preds = svm_model.predict(X_val[i])
    val_acc = accuracy_score(y_val[i], val_preds)
    print(fileNames[i] + f' Validation Accuracy: {val_acc:.4f}')
    print(classification_report(y_val[i], val_preds))


# 预测剩余标签

# 读取test数据

testDataList = []
fileNames = ['s5.npz', 's6.npz', 's7.npz']
for fileName in fileNames:
    with numpy.load('./Data\\test\\' + fileName) as data:
        data_X = data['X']
        testDataList.append(data_X)

# 欧拉对齐
testDataListEA = []
for testData in testDataList:
    data_X = EA.EA(testData)
    testDataListEA.append(data_X)

# 提取CSP特征 (这里有问题，因为CSP需要标签，这里只能使用之前的CSP滤波器，严格意义上说是不正确的)
Test_Data = []
for testData in testDataListEA:
    if len(trainDataLabelDict) == 2:
        data_X = numpy.einsum('ij,kjl->kil', numpy.concatenate((filters[0][:FILTER_COUNT], filters[0][-FILTER_COUNT:]), axis=0), testData)
    else:
        for aFilter in filters:
            pass
    Test_Data.append(data_X.real)

testLogFeature = numpy.zeros((data_X.shape[0] * len(Test_Data), data_X.shape[1]))
i = 0
for subject in Test_Data:
    for trail in subject:
        testLogFeature[i] = numpy.log10(numpy.diag(trail @ trail.T) / numpy.trace(trail @ trail.T))
        i += 1
    

# 使用模型预测
prediction = svm_model.predict(testLogFeature)

chunk_size = len(prediction) // len(fileNames)
chunks = [prediction[i * chunk_size:(i + 1) * chunk_size] for i in range(len(fileNames))]
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    headers = fileNames
    csvwriter.writerow(headers)
    for row in numpy.array(chunks).T:
        csvwriter.writerow(row)
'''        
with open('11.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    headers = trainDataDictRaw.keys()
    csvwriter.writerow(headers)
    data_y = []
    for key in trainDataDictRaw:
        data_y.append(trainDataDictRaw[key][1])
        
    for row in numpy.array(data_y).T:
        csvwriter.writerow(row)
'''