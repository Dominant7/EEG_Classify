import numpy as np
import scipy.linalg as la

def CSP(labelDict):
    '''
    输入以标签作为词典的键，数据列表作为词典的值的词典
    应保证数据列表中每个trail有相同的形状
    返回CSP滤波变换矩阵(np.array)
    取前n行即可得到特征值前n大的成分
    '''
    if len(labelDict) < 2:
        print('至少有2类才能CSP空间滤波')
        return (None,) * len(labelDict)
    else:
        filters = []
        # CSP算法
        # 对于每个标签x，找到均值协方差矩阵Rx和not_Rx，用于计算空间滤波器SFx
        RxList = []
        for x in labelDict:
            # 计算Rx
            Rx = []
            for trail in labelDict[x]: # 200*59*300 array
                Rx.append(covarianceMatrix(trail))
            RxSum = np.sum(Rx, axis=0)
            RxList.append(RxSum / len(labelDict[x])) # 平均归一化协方差矩阵
            # 只有两个标签，不需要计算任何其他均值协方差矩阵
        if len(labelDict) == 2:
            filters.append(spatialFilter(RxList[0], RxList[1]))
        # 多分类
        else:
            pass
    return filters

def covarianceMatrix(A):
    '''
    返回按方差缩放后的协方差矩阵,即归一化协方差矩阵
    '''
    covA = np.cov(A, rowvar=True)
    covA = covA / np.trace(covA)
    return covA

def spatialFilter(Ra, Rb):
    '''
    输入均值协方差矩阵Ra和Rb
    返回CSP滤波器SFa
    '''
    R = Ra + Rb
    E, U = la.eig(R)

    # CSP要求特征值E和特征向量U按降序排序
    ord = np.argsort(E)
    ord = ord[::-1]  # argsort给出升序，翻转以获得降序
    E = E[ord]
    U = U[:, ord]

    # 计算白化变换矩阵
    P = np.dot(np.sqrt(la.inv(np.diag(E))), np.transpose(U))

    # 对均值协方差矩阵施加变换
    Sa = np.dot(P, np.dot(Ra, np.transpose(P)))
    Sb = np.dot(P, np.dot(Rb, np.transpose(P)))

    # 计算并排序广义特征值和特征向量
    E1, U1 = la.eig(Sa, Sb)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:, ord1]

    # 计算投影矩阵（即空间滤波器）
    SFa = np.dot(np.transpose(U1), P)
    return SFa