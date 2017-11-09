from numpy import *

'''
fileName을 지정해주면 그 파일의 데이터를 읽어오는 함수이다.
데이터 포맷은 DATA1 DATA2 LABEL이다.
반환값은 읽은 데이터 매트릭스와 라벨 매트릭스이다.
'''
def loadDataSet(fileName):
    dataMat = [];
    labelMat = [];
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
'''
Sigmoid를 간단히 구현한 함수.
'''
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

'''
기울기 상승 함수.
dataMatIn의 경우 두개의 서로다른 속성을 두개의 컬럼으로 가진 배열이며 loadDataSet함수에서 0번째 인자를 1.0으로 넣어주므로 100 x 3의 행렬이다.
classLabels의 경우100 x 1 행렬이다.
'''
def gradAscent(dataMatIn, classLabels):
    #들어온 배열을 매트릭스로 변환시킨다.
    dataMatrix = mat(dataMatIn)
    #들어온 배열을 매트릭스로 변환시키고 트랜스포즈시켜 100 x 1에서 1 x 100 로우백터로 변환시킨다.
    labelMat = mat(classLabels).transpose()
    # row 수와 column 수를 구한다.
    rowCount , columnCount = shape(dataMatrix)
    #이 변수는 목적을 향하도록 하기 위한 단계의 크기.
    alpha = 0.001
    #반복 횟수
    maxCycles = 500
    #3 x 1로 초기화
    weights = ones((columnCount, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei
    dataMat, labelMat = loadDataSet("testSet.txt")
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

dataArr, labelMat = loadDataSet("testSet.txt")
weights = gradAscent(dataArr, labelMat)
plotBestFit(weights.getA())