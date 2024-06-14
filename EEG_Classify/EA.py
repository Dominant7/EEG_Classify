import numpy
from scipy.linalg import sqrtm, inv

def EA(rawArray):
    '''
    输入要处理的numpy array(格式为[trails * 通道* 采样点])
    返回进行欧拉对齐后的numpy array
    '''
    transMatrixList = []
    covarianceMatrixList = []
    for trail in rawArray: # 遍历trail
        # covarianceMatrixList = []
        # for trail in rawArray:
        #     covarianceMatrix = numpy.cov(trail[i])
        #     covarianceMatrixList.append(covarianceMatrix)
        covarianceMatrix = numpy.cov(trail, rowvar=True)
        covarianceMatrixList.append(covarianceMatrix)
    covarianceMatrixMean = numpy.mean(covarianceMatrixList, axis=0)
    transMatrix = inv(sqrtm(covarianceMatrixMean))
    processedArray = numpy.empty_like(rawArray)
    for i in range(processedArray.shape[0]): # 遍历trail
        processedArray[i] = transMatrix @ rawArray[i]
            
    return processedArray