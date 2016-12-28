import bookkNNAlgorithm

group, labels = bookkNNAlgorithm.createDataSet()

print(bookkNNAlgorithm.kNNAlgorithm([0, 0], group, labels, 3))