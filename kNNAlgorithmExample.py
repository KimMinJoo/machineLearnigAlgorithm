from numpy import *
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import bookkNNAlgorithm

def file2matrix(filename, k):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,k))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:k]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

'''
정규화가 필요한 이유
각 피쳐마다 크기가 다른데 그럴경우 유클리드거리를 사용하는 현재의 kNN알고리즘에서 크기가 큰 피쳐가 비중이 너무커짐.(각 피쳐별 형식통일)
책에서는 여러값을 반환하지만 정규화만 시키는것이 맞는것같아 정규화된 매트릭스만 반환.
정규화식
normaliztionValue = (originValue - minValue) / (maxValue - minValue)
'''
def autoNormaliztionFunc(dataSet):
    #min, max와 min-max를 구한다
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    range = maxValue - minValue

    #dataSet모양과 같은 0으로 찬 매트릭스 셍성
    normaliztionDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #tile을 이용하여 먼저 빼주고 나눠준다.
    normaliztionDataSet = dataSet - tile(minValue, (m,1))
    normaliztionDataSet = normaliztionDataSet / tile(range, (m,1))
    return normaliztionDataSet

def datingClassTest():
    hoRatio = 0.10
    datingDataMat ,datingLabels = file2matrix('datingTestSet2.txt', 3)
    datingDataMat = autoNormaliztionFunc(datingDataMat)
    m= datingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
     classifierResult = bookkNNAlgorithm.kNNAlgorithm(datingDataMat[i, :], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
     if (classifierResult != datingLabels[i]):
         print("the classifier came back with : %d, the real answer is: %d"%(classifierResult, datingLabels[i]))
         errorCount += 1.0
    print("the total error rate is %f"% (errorCount/float(numTestVecs)))

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt', 3)
print("정규화 전")
print(datingDataMat)

datingDataMat = autoNormaliztionFunc(datingDataMat)
print("정규화 후")
print(datingDataMat)

print(datingLabels)

fig = plt.figure()
#figure의 구분값( 1 1 1 의 경우 1x1의 첫번째  212 의 경우 2x1의 두번째.....)
ax = fig.add_subplot(111)
#(y축, x축 크기, 색(각 점마다 색을 주기위해 배열)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 30.0*array(datingLabels), array(datingLabels))
plt.show()

datingClassTest()