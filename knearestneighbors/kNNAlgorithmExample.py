import matplotlib.pyplot as plt
from numpy import *
from numpy import array

from knearestneighbors import bookkNNAlgorithm
'''
file을 matrix로 변환시켜주는 함수
filename 파일이름
k 파일 한 줄에 들어가있는 데이터의 수
'''
def file2matrix(filename, k):
    #파일을 읽는다.
    fr = open(filename)
    #파일의 각 줄로 배열을 만든다.
    arrayOfLines = fr.readlines()
    #파일의 전체 줄 수.
    numberOfLines = len(arrayOfLines)
    #0으로된 (파일줄수, k)의 matrix를 만든다.
    returnMat = zeros((numberOfLines,k))
    #라벨을 따로 저장할 라벨 배열.
    classLabelVector = []
    index = 0
    #각 라인의 데이터를 매트릭스와 라벨베열에 넣어준다.
    for line in arrayOfLines:
        #문자 양끝 공백제거. (필요에 따라 생략가능)
        line = line.strip()
        #라인을 tab기준으로 스플릿하여 리스트로 만든다.
        listFromLine = line.split('\t')
        #매트릭스의 index번째에 리스트를 넣어준다.
        returnMat[index,:] = listFromLine[0:k]
        #데이터 마지막에 있는 라벨을 라벨 리스트에 저장해준다.
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
def autoNormalizationFunc(dataSet):
    #min, max와 max-min을 구한다.
    #min, max의 경우 각 열에서 min, max를 구한 1행짜리 list가 나온다.
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    range = maxValue - minValue

    m = dataSet.shape[0]
    #tile을 이용하여 먼저 빼주고 나눠준다.
    resultDataSet = dataSet - tile(minValue, (m, 1))
    resultDataSet = resultDataSet / tile(range, (m,1))
    return resultDataSet

def datingClassTest():
    #비율
    hoRatio = 0.10

    #데이터 읽기
    datingDataMat ,datingLabels = file2matrix('datingTestSet2.txt', 3)

    #데이터 정규화
    datingDataMat = autoNormalizationFunc(datingDataMat)

    #총 갯수 구하기
    m= datingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)

    #에러 카운트 초기화
    errorCount = 0.0

    #검증 시작.
    for i in range(numTestVecs):
     classifierResult = bookkNNAlgorithm.kNNAlgorithm(datingDataMat[i, :], datingDataMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)

     #오류 상황
     if (classifierResult != datingLabels[i]):
         print("the classifier came back with : %d, the real answer is: %d"%(classifierResult, datingLabels[i]))
         errorCount += 1.0

    #결과 출력
    print("the total error rate is %f"% (errorCount/float(numTestVecs)))


def getPersonInfo():
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    return percentTats, ffMiles, iceCream

def learnNewData():
    #라벨이 정수인데 한글로 보여주기위한 리스트
    resultList = ['안좋음', '보통', '좋음']

    #데이트 상대 정보 입력
    percentTats, ffMiles, iceCream = getPersonInfo()

    #데이터 읽어오기
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt', 3)

    #정규화
    datingDataMat = autoNormalizationFunc(datingDataMat)

    #인풋값 리스트로 만들기
    inArr = array([[ffMiles, percentTats, iceCream]])

    #인풋값 정규화
    inArr = autoNormalizationFunc(inArr)
    print(inArr)

    #알고리즘 실행
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
learnNewData()
'''
#데이터 읽기.
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt', 3)
print("정규화 전")
print(datingDataMat)



#정규화!
'''
datingDataMat = autoNormalizationFunc(datingDataMat)
print("정규화 후")
print(datingDataMat)
'''


#데이터 분석!
'''
fig = plt.figure()
#figure의 구분값( 1 1 1 의 경우 1x1의 첫번째  212 의 경우 2x1의 두번째.....)
ax = fig.add_subplot(221)
ax.set(title='mileage and video game', xlabel='mileage', ylabel='video game')
#(y축, x축 크기, 색(각 점마다 색을 주기위해 배열)
#연간 항송 마일리지에 비례한 비디오 게임으로 보내는 시간의 비율
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 30.0*array(datingLabels), array(datingLabels))
ax2 = fig.add_subplot(222)
ax2.set(title='video game and ice cream', xlabel='video game', ylabel='ice cream')
ax2.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 30.0*array(datingLabels), array(datingLabels))
ax3 = fig.add_subplot(223)
ax3.set(title='mileage and ice cream', xlabel='mileage', ylabel='ice cream')
ax3.scatter(datingDataMat[:, 0], datingDataMat[:, 2], 30.0*array(datingLabels), array(datingLabels))
plt.show()
'''


#데이터 테스트!
#datingClassTest()


#데이터 입력!
#learnNewData()