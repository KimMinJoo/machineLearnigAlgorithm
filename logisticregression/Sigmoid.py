import numpy
import matplotlib.pyplot as plt

class Sigmoid:
    _sigmoid = []
    _x=[]
    _pointNumber = 10000

    def __init__(self):
        self._sigmoid = []
        self._x = []
        self._pointNumber = 10000;

    def createSigmoid(self, minX, maxX):
        self._x = numpy.linspace(minX, maxX, self._pointNumber)
        for itr in self._x:
            self._sigmoid.append(1/(1 + numpy.exp(-itr)))

    def view(self):
        plt.plot(self._x, self._sigmoid)
        plt.show()

    @property
    def sigmoid(self):
        return self._sigmoid

    @sigmoid.setter
    def sigmoid(self, sigmoid):
        self._sigmoid = sigmoid

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def pointNumber(self):
        return self._pointNumber

    @pointNumber.setter
    def pointNumber(self, pointNumber):
        self._pointNumber = pointNumber


sigmoid1 = Sigmoid()
sigmoid1.createSigmoid(-5, 5)
#sigmoid1.view()

sigmoid2 = Sigmoid()
sigmoid2.createSigmoid(-100, 100)
#sigmoid2.view()

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.set(title='-5~5', xlabel='x', ylabel='value')
ax1.plot(sigmoid1.x, sigmoid1.sigmoid, linewidth = 2.0, linestyle = "-")

ax2 = fig.add_subplot(212)
ax2.set(title='-100~100', xlabel='x', ylabel='value')
ax2.plot(sigmoid2.x, sigmoid2.sigmoid, linewidth = 2.0, linestyle = "-")

plt.show()