import numpy
import EA

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

pass
