'''
다층 퍼셉트론 Example
Xor
AND 퍼셉트론과 OR 퍼셉트론으로 구현.
'''

class Perceptron() :
    def __init__(self):
        self.__threshold = 0
        self.__weightArr = [0.3, 0.4, 0.1]
        self.__bias = -1
        self.__learningRate = 0.05

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @property
    def weightArr(self):
        return self.__weightArr

    @weightArr.setter
    def weightArr(self, weightArr):
        self.__weightArr = weightArr

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    @property
    def learningRate(self):
        return self.__learningRate

    @learningRate.setter
    def learningRate(self, learningRate):
        self.__learningRate = learningRate

    def learning(self, dataArr):
        for i in range(0 , len(dataArr)) :
            result  = self.calculate(dataArr[i][0:-1])
            if result == dataArr[i][-1] :
                continue
            else:
                #그 뭐시기냐 러닝레이트 다시 설정하고 러닝 다시 ㄱㄱ!
                self.updateLearningRate(dataArr[i], result)
                self.learning(dataArr)
                break
        return

    def updateLearningRate(self, dataRow, result):
        dataRow = [self.bias] + dataRow
        for i in range(0, len(self.weightArr)) :
            self.weightArr[i] += (self.learningRate * dataRow[i] * (dataRow[-1] - result))

    def calculate(self, coordinate):
        result = self.weightArr[0] * self.bias
        for i in range(0, len(coordinate)) :
            result += coordinate[i] * self.weightArr[i+1]

        result = 1 if result >= self.threshold else 0
        return result

class MultiLayerPerceptron() :
    def __init__(self):
        self.__perceptrons = list()

    @property
    def perceptrons(self):
        return self.__perceptrons

    def appendPerceptron(self, perceptron):
        self.perceptrons.append(perceptron)

    def calculate(self, coordinate):
        before = self.perceptrons[0].calculate(coordinate)
        for i in range(1, len(self.perceptrons)):
            current = self.perceptrons[i].calculate(coordinate)
            if before == current:
                return 0

            before = current

        return 1

import sys
sys.setrecursionlimit(10000)

andLearningDataArr = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0.7, 0.8, 1], [0.7, 0.79, 0], [0.9, 0.6, 1], [0.9, 0.59, 0], [0.75, 0.75, 1], [0.6, 0.9, 1]]

print(andLearningDataArr[1][0:-1])
andPerceptron = Perceptron()
andPerceptron.learning(andLearningDataArr)
coordinate = [0,1]
print("===============and 결과===============")
print("데이터 :")
print(andLearningDataArr)
print("가중치 :")
print(andPerceptron.weightArr)
print("좌표 :")
print(coordinate)
print("결과 :")
print(andPerceptron.calculate(coordinate))


orLearningDataArr = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0, 1], [0, 0.5, 1], [0.25, 0.25, 1], [0.2, 0.2, 0], [0.3, 0.19, 0]]
orPerceptron = Perceptron()
orPerceptron.learning(orLearningDataArr)
print("===============or 결과===============")
print("데이터 :")
print(orLearningDataArr)
print("가중치 :")
print(orPerceptron.weightArr)
print("좌표 :")
print(coordinate)
print("결과 :")
print(orPerceptron.calculate(coordinate))

xorPerceptron = MultiLayerPerceptron()
xorPerceptron.appendPerceptron(andPerceptron)
xorPerceptron.appendPerceptron(orPerceptron)

print("============Xor 결과=============")
print("좌표 :")
print(coordinate)
print("결과 :")
print(xorPerceptron.calculate(coordinate))