from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def kNNAlgorithm(inX, dataSet, labels, k):
    #shape함수의 경우 array의 모양을 나타냄 위의 createDataSet에 따르면 4,2의 행렬이므로 [0]은 4, [1]은 2를 나타낸다.
    dataSetSize = dataSet.shape[0]
    #inX의 값으로 (dataSetSize,1)사이즈로 만들고 dataSet만큼 빼 dataSet과의 X축거리, Y축거리를 확보한다
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #거리를 구하기위해 제곱을 한다.
    sqDiffMat = diffMat ** 2
    print(sqDiffMat)
    #axis = 0은 행  1은 열을 의미함. 행별로 더하는게 아니라 열별로 더한다.
    sqDistances = sqDiffMat.sum(axis = 1)
    #루트를 씌워 최종 거리를 구한다.
    distances = sqDistances ** 0.5
    #거리별로 index들을 정렬한다.
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        #가장 가까운 인덱스들의 라벨을 꺼낸다.
        voteIlabel = labels[sortedDistIndicies[i]]
        #classCount dictionary에서 각 라벨의 횟수를 가져와 1을 올려준다. 다만 없을경우 0으로 가져온다.
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #dictionary 정렬 itemgetter가 0이면 키기준, 1이면 value기준
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    #정렬된 dictionary에서 가장 첫밸류 리턴.
    return sortedClassCount[0][0]