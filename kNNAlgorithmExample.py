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
각 피쳐마다 크기가 다른데 그럴경우 유클리드거리를 사용하는 현재의 kNN알고리즘에서 절대적인 크기가 큰 피쳐가 비중이 너무커짐.(각 피쳐별 형식통일)
책에서는 여러값을 반환하지만 정규화만 시키는것이 맞는것같아 정규화된 매트릭스만 반환.
정규화식
normaliztionValue = (originValue - minValue) / (maxValue - minValue)
'''
def autoNormalizationFunc(normalizationDataSet, standardDataSet):
    #min, max와 min-max를 구한다
    minValue = standardDataSet.min(0)
    maxValue = standardDataSet.max(0)
    range = maxValue - minValue

    #dataSet모양과 같은 0으로 찬 매트릭스 셍성
    resultDataSet = zeros(shape(normalizationDataSet))
    m = normalizationDataSet.shape[0]
    #tile을 이용하여 먼저 빼주고 나눠준다.
    resultDataSet = normalizationDataSet - tile(minValue, (m,1))
    resultDataSet = resultDataSet / tile(range, (m,1))
    return resultDataSet

def datingClassTest():
    hoRatio = 0.10
    datingDataMat ,datingLabels = file2matrix('datingTestSet2.txt', 3)
    datingDataMat = autoNormalizationFunc(datingDataMat, datingDataMat)
    m= datingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
     classifierResult = bookkNNAlgorithm.kNNAlgorithm(datingDataMat[i, :], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
     if (classifierResult != datingLabels[i]):
         print("the classifier came back with : %d, the real answer is: %d"%(classifierResult, datingLabels[i]))
         errorCount += 1.0
    print("the total error rate is %f"% (errorCount/float(numTestVecs)))


def getPersonInfo():
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    return percentTats, ffMiles, iceCream

def learnNewData():
    resultList = ['안좋음', '보통', '좋음']
    percentTats, ffMiles, iceCream = getPersonInfo()
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt', 3)
    datingDataMat = autoNormalizationFunc(datingDataMat, datingDataMat)
    inArr = array([[ffMiles, percentTats, iceCream]])
    inArr = autoNormalizationFunc(inArr, datingDataMat)
    print(inArr)
    result = bookkNNAlgorithm.kNNAlgorithm(inArr, datingDataMat, datingLabels, 3)
    print("You will probably like this person: %s" % (resultList[result-1]))


'''
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt', 3)
print("정규화 전")
print(datingDataMat)

datingDataMat = autoNormalizationFunc(datingDataMat, datingDataMat)
print("정규화 후")
print(datingDataMat)

print(datingLabels[670])
print(datingLabels[87])
print(datingLabels[778])

fig = plt.figure()
#figure의 구분값( 1 1 1 의 경우 1x1의 첫번째  212 의 경우 2x1의 두번째.....)
ax = fig.add_subplot(111)
#(y축, x축 크기, 색(각 점마다 색을 주기위해 배열)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 30.0*array(datingLabels), array(datingLabels))
plt.show()

datingClassTest()
'''
learnNewData()